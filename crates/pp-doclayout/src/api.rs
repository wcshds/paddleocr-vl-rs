use std::{path::Path, str::FromStr};

use burn::{prelude::Backend, tensor::DType};
use image::DynamicImage;
use serde::Serialize;

use crate::{
    PPDocLayoutV2Config, PPDocLayoutV2ForObjectDetection, PPDocLayoutV3Config,
    PPDocLayoutV3ForObjectDetection,
    postprocessing::{
        DetectionResult, DetectionResultV3, post_process_object_detection,
        post_process_object_detection_v3,
    },
    preprocessing::{DocLayoutPreprocessConfig, preprocess_image},
};

/// Official class-id-to-label mapping used by PP-DocLayoutV2/V3.
///
/// Some labels intentionally appear more than once because the upstream model
/// distinguishes separate class IDs that are rendered with the same public
/// label (for example multiple header/footer/text categories). Do not dedupe
/// this table: downstream task mapping and JSON output should preserve the
/// original class ID.
pub const ID2LABEL: &[&str] = &[
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer",
    "footnote",
    "formula_number",
    "header",
    "header",
    "image",
    "formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "text",
    "vision_footnote",
];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DocLayoutVersion {
    #[default]
    V2,
    V3,
}

impl DocLayoutVersion {
    pub fn default_model_path(self) -> &'static str {
        match self {
            Self::V2 => "./model1.0/PP-DocLayoutV2/model.safetensors",
            Self::V3 => "./model1.5/PP-DocLayoutV3/model.safetensors",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::V2 => "PP-DocLayoutV2",
            Self::V3 => "PP-DocLayoutV3",
        }
    }
}

impl FromStr for DocLayoutVersion {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "2" | "v2" | "2.0" => Ok(Self::V2),
            "3" | "v3" | "3.0" => Ok(Self::V3),
            other => Err(format!(
                "unsupported DocLayout version '{other}', expected v2 or v3"
            )),
        }
    }
}

pub fn label_name(id: usize) -> &'static str {
    ID2LABEL.get(id).copied().unwrap_or("unknown")
}

#[derive(Clone, Debug, Serialize)]
pub struct LayoutBlock {
    pub id: usize,
    pub label_id: usize,
    pub label: String,
    pub score: f32,
    pub bbox: [f32; 4],
    pub polygon: Option<Vec<[f32; 2]>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LayoutResult {
    pub width: u32,
    pub height: u32,
    pub blocks: Vec<LayoutBlock>,
}

enum DocLayoutInner<B: Backend> {
    V2(PPDocLayoutV2ForObjectDetection<B>),
    V3(PPDocLayoutV3ForObjectDetection<B>),
}

pub struct DocLayout<B: Backend> {
    device: B::Device,
    version: DocLayoutVersion,
    dtype: DType,
    threshold: f32,
    preprocess_cfg: DocLayoutPreprocessConfig,
    inner: DocLayoutInner<B>,
}

impl<B: Backend> DocLayout<B> {
    pub fn load_from_safetensors(
        version: DocLayoutVersion,
        model_path: impl AsRef<Path>,
        device: B::Device,
        threshold: f32,
    ) -> Self {
        Self::try_load_from_safetensors(version, model_path, device, threshold)
            .expect("failed to load PP-DocLayout weights")
    }

    pub fn load_from_safetensors_with_dtype(
        version: DocLayoutVersion,
        model_path: impl AsRef<Path>,
        device: B::Device,
        threshold: f32,
        target_dtype: DType,
    ) -> Self {
        Self::try_load_from_safetensors_with_dtype(
            version,
            model_path,
            device,
            threshold,
            target_dtype,
        )
        .expect("failed to load PP-DocLayout weights")
    }

    pub fn try_load_from_safetensors(
        version: DocLayoutVersion,
        model_path: impl AsRef<Path>,
        device: B::Device,
        threshold: f32,
    ) -> Result<Self, String> {
        Self::try_load_from_safetensors_with_dtype(
            version,
            model_path,
            device,
            threshold,
            DType::F32,
        )
    }

    pub fn try_load_from_safetensors_with_dtype(
        version: DocLayoutVersion,
        model_path: impl AsRef<Path>,
        device: B::Device,
        threshold: f32,
        target_dtype: DType,
    ) -> Result<Self, String> {
        let model_path = model_path.as_ref();
        let inner = match version {
            DocLayoutVersion::V2 => {
                let model = PPDocLayoutV2Config::new().try_init_from_safetensors_with_dtype::<B>(
                    model_path,
                    &device,
                    target_dtype,
                )?;
                DocLayoutInner::V2(model)
            }
            DocLayoutVersion::V3 => {
                let model = PPDocLayoutV3Config::new().try_init_from_safetensors_with_dtype::<B>(
                    model_path,
                    &device,
                    target_dtype,
                )?;
                DocLayoutInner::V3(model)
            }
        };

        Ok(Self {
            device,
            version,
            dtype: target_dtype,
            threshold,
            preprocess_cfg: DocLayoutPreprocessConfig::default(),
            inner,
        })
    }

    pub fn version(&self) -> DocLayoutVersion {
        self.version
    }

    pub fn detect(&self, image: &DynamicImage) -> LayoutResult {
        let width = image.width();
        let height = image.height();
        let (pixel_values, original_size) =
            preprocess_image::<B>(image, &self.preprocess_cfg, &self.device, self.dtype);
        let blocks = match &self.inner {
            DocLayoutInner::V2(model) => {
                let output = model.forward(pixel_values);
                let results = post_process_object_detection::<B>(
                    &output.logits,
                    &output.pred_boxes,
                    &output.order_logits,
                    &[original_size],
                    self.threshold,
                );
                blocks_from_v2(&results[0])
            }
            DocLayoutInner::V3(model) => {
                let output = model.forward(pixel_values);
                let results = post_process_object_detection_v3::<B>(
                    &output.logits,
                    &output.pred_boxes,
                    &output.order_logits,
                    Some(&output.out_masks),
                    &[original_size],
                    self.threshold,
                );
                blocks_from_v3(&results[0])
            }
        };

        LayoutResult {
            width,
            height,
            blocks,
        }
    }
}

fn blocks_from_v2(result: &DetectionResult) -> Vec<LayoutBlock> {
    result
        .scores
        .iter()
        .zip(result.labels.iter())
        .zip(result.boxes.iter())
        .enumerate()
        .map(|(id, ((score, label_id), bbox))| LayoutBlock {
            id,
            label_id: *label_id,
            label: label_name(*label_id).to_string(),
            score: *score,
            bbox: *bbox,
            polygon: None,
        })
        .collect()
}

fn blocks_from_v3(result: &DetectionResultV3) -> Vec<LayoutBlock> {
    result
        .scores
        .iter()
        .zip(result.labels.iter())
        .zip(result.boxes.iter())
        .zip(result.polygon_points.iter())
        .enumerate()
        .map(|(id, (((score, label_id), bbox), polygon))| LayoutBlock {
            id,
            label_id: *label_id,
            label: label_name(*label_id).to_string(),
            score: *score,
            bbox: *bbox,
            polygon: Some(polygon.clone()),
        })
        .collect()
}
