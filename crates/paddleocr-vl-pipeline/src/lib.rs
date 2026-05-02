use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    str::FromStr,
};

use burn::{prelude::Backend, tensor::DType};
use image::DynamicImage;
use paddleocr_vl::{OcrTask, PaddleOcrVersion, PaddleOcrVl};
use pp_doclayout::{DocLayout, DocLayoutVersion, LayoutBlock};
use serde::Serialize;

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp", "gif"];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PipelineVersion {
    V1,
    #[default]
    V1_5,
}

impl PipelineVersion {
    /// PaddleOCR-VL-1.0 uses the V1 pipeline with PP-DocLayoutV2. Version 1.5
    /// upgrades the layout stage to PP-DocLayoutV3, but the recognizer remains
    /// a direct element recognizer; text spotting is intentionally not part of
    /// this page-level pipeline.
    pub fn vl_version(self) -> PaddleOcrVersion {
        match self {
            Self::V1 => PaddleOcrVersion::V1_0,
            Self::V1_5 => PaddleOcrVersion::V1_5,
        }
    }

    pub fn layout_version(self) -> DocLayoutVersion {
        match self {
            Self::V1 => DocLayoutVersion::V2,
            Self::V1_5 => DocLayoutVersion::V3,
        }
    }

    pub fn default_layout_model_path(self) -> &'static str {
        self.layout_version().default_model_path()
    }

    pub fn default_vl_model_dir(self) -> &'static str {
        self.vl_version().default_model_dir()
    }

    pub fn default_layout_threshold(self) -> f32 {
        match self {
            Self::V1 => 0.5,
            Self::V1_5 => 0.3,
        }
    }
}

impl FromStr for PipelineVersion {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "v1" | "1" | "1.0" => Ok(Self::V1),
            "v1.5" | "1.5" => Ok(Self::V1_5),
            other => Err(format!(
                "unsupported pipeline version '{other}', expected v1 or v1.5"
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Pipeline preset: V1 maps to PP-DocLayoutV2 + PaddleOCR-VL-1.0, while
    /// V1.5 maps to PP-DocLayoutV3 + PaddleOCR-VL-1.5.
    pub version: PipelineVersion,
    pub layout_model_path: PathBuf,
    pub vl_model_dir: PathBuf,
    pub dtype: DType,
    pub layout_threshold: f32,
    pub max_new_tokens: Option<usize>,
    pub use_chart_recognition: bool,
    pub use_seal_recognition: bool,
    pub use_ocr_for_image_block: bool,
    /// Labels ignored only when rendering Markdown.
    pub markdown_ignore_labels: Vec<String>,
}

impl PipelineConfig {
    pub fn for_version(version: PipelineVersion) -> Self {
        Self {
            version,
            layout_model_path: PathBuf::from(version.default_layout_model_path()),
            vl_model_dir: PathBuf::from(version.default_vl_model_dir()),
            dtype: DType::F32,
            layout_threshold: version.default_layout_threshold(),
            max_new_tokens: None,
            use_chart_recognition: false,
            use_seal_recognition: version == PipelineVersion::V1_5,
            use_ocr_for_image_block: false,
            markdown_ignore_labels: vec![
                "number".into(),
                "footnote".into(),
                "header".into(),
                "header_image".into(),
                "footer".into(),
                "footer_image".into(),
                "aside_text".into(),
            ],
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self::for_version(PipelineVersion::default())
    }
}

pub struct PaddleOcrVlPipelineBuilder<B: Backend> {
    device: B::Device,
    config: PipelineConfig,
}

impl<B: Backend> PaddleOcrVlPipelineBuilder<B>
where
    B::Device: Clone,
{
    pub fn version(mut self, version: PipelineVersion) -> Self {
        self.config = PipelineConfig::for_version(version);
        self
    }

    pub fn layout_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.layout_model_path = path.into();
        self
    }

    pub fn vl_model_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.vl_model_dir = dir.into();
        self
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.config.dtype = dtype;
        self
    }

    pub fn layout_threshold(mut self, threshold: f32) -> Self {
        self.config.layout_threshold = threshold;
        self
    }

    pub fn max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.config.max_new_tokens = Some(max_new_tokens);
        self
    }

    pub fn use_chart_recognition(mut self, enabled: bool) -> Self {
        self.config.use_chart_recognition = enabled;
        self
    }

    pub fn use_seal_recognition(mut self, enabled: bool) -> Self {
        self.config.use_seal_recognition = enabled;
        self
    }

    pub fn use_ocr_for_image_block(mut self, enabled: bool) -> Self {
        self.config.use_ocr_for_image_block = enabled;
        self
    }

    pub fn markdown_ignore_labels(mut self, labels: Vec<String>) -> Self {
        self.config.markdown_ignore_labels = labels;
        self
    }

    pub fn build(self) -> Result<PaddleOcrVlPipeline<B>, String> {
        let layout = DocLayout::<B>::try_load_from_safetensors_with_dtype(
            self.config.version.layout_version(),
            &self.config.layout_model_path,
            self.device.clone(),
            self.config.layout_threshold,
            self.config.dtype,
        )?;
        let mut recognizer = PaddleOcrVl::<B>::builder(self.device)
            .version(self.config.version.vl_version())
            .model_dir(self.config.vl_model_dir.clone())
            .dtype(self.config.dtype);
        if let Some(max_new_tokens) = self.config.max_new_tokens {
            recognizer = recognizer.max_new_tokens(max_new_tokens);
        }
        let recognizer = recognizer.build()?;

        Ok(PaddleOcrVlPipeline {
            layout,
            recognizer,
            config: self.config,
        })
    }
}

pub struct PaddleOcrVlPipeline<B: Backend> {
    layout: DocLayout<B>,
    recognizer: PaddleOcrVl<B>,
    config: PipelineConfig,
}

#[derive(Clone, Debug)]
pub enum PipelineProgress {
    LayoutDetected {
        total_blocks: usize,
    },
    BlockStarted {
        current: usize,
        total: usize,
        label: String,
        will_recognize: bool,
    },
    BlockFinished {
        current: usize,
        total: usize,
        label: String,
        recognized: bool,
    },
}

impl<B: Backend> PaddleOcrVlPipeline<B>
where
    B::Device: Clone,
{
    pub fn builder(device: B::Device) -> PaddleOcrVlPipelineBuilder<B> {
        PaddleOcrVlPipelineBuilder {
            device,
            config: PipelineConfig::default(),
        }
    }

    pub fn predict(&self, input: impl AsRef<Path>) -> Result<Vec<PipelineResult>, String> {
        let input = input.as_ref();
        let paths = collect_images(input)?;
        if paths.is_empty() {
            return Err(format!("no image files found in {}", input.display()));
        }
        paths
            .into_iter()
            .map(|path| self.predict_image_path(path))
            .collect()
    }

    pub fn predict_image_path(&self, path: PathBuf) -> Result<PipelineResult, String> {
        let image =
            image::open(&path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        self.predict_image(Some(path), image)
    }

    pub fn predict_image_path_with_progress<F>(
        &self,
        path: PathBuf,
        progress: F,
    ) -> Result<PipelineResult, String>
    where
        F: FnMut(PipelineProgress),
    {
        let image =
            image::open(&path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        self.predict_image_with_progress(Some(path), image, progress)
    }

    pub fn predict_image(
        &self,
        input_path: Option<PathBuf>,
        image: DynamicImage,
    ) -> Result<PipelineResult, String> {
        self.predict_image_with_progress(input_path, image, |_| {})
    }

    pub fn predict_image_with_progress<F>(
        &self,
        input_path: Option<PathBuf>,
        image: DynamicImage,
        mut progress: F,
    ) -> Result<PipelineResult, String>
    where
        F: FnMut(PipelineProgress),
    {
        let layout = self.layout.detect(&image);
        let total_blocks = layout.blocks.len();
        let width = layout.width;
        let height = layout.height;
        let page_index = None;
        let mut blocks = Vec::new();
        let mut markdown_order = 1;
        let markdown_ignored = self
            .config
            .markdown_ignore_labels
            .iter()
            .map(|label| label.as_str())
            .collect::<HashSet<_>>();

        progress(PipelineProgress::LayoutDetected { total_blocks });

        for (index, block) in layout.blocks.into_iter().enumerate() {
            let current = index + 1;
            let will_recognize = task_for_label(block.label.as_str(), &self.config).is_some();
            progress(PipelineProgress::BlockStarted {
                current,
                total: total_blocks,
                label: block.label.clone(),
                will_recognize,
            });
            let content = self.recognize_block(&image, &block)?;
            let recognized = content.is_some();
            let content = content.unwrap_or_default();
            progress(PipelineProgress::BlockFinished {
                current,
                total: total_blocks,
                label: block.label.clone(),
                recognized,
            });
            let id = blocks.len();
            let bbox = bbox_to_u32(block.bbox);
            let order = if !content.trim().is_empty()
                && !markdown_ignored.contains(block.label.as_str())
                && block.label != "image"
            {
                let order = Some(markdown_order);
                markdown_order += 1;
                order
            } else {
                None
            };
            blocks.push(ParsedBlock {
                id,
                label: block.label,
                content,
                bbox,
                order,
                group_id: id,
                block_polygon_points: polygon_points(block.polygon, bbox),
                source_bbox: block.bbox,
            });
        }

        Ok(PipelineResult {
            input_path: input_path.as_deref().and_then(input_file_name),
            source_path: input_path,
            page_index,
            page_count: None,
            width,
            height,
            model_settings: PipelineModelSettings::from_config(&self.config),
            blocks,
        })
    }

    fn recognize_block(
        &self,
        image: &DynamicImage,
        block: &LayoutBlock,
    ) -> Result<Option<String>, String> {
        let Some(task) = task_for_label(block.label.as_str(), &self.config) else {
            return Ok(None);
        };
        // PP-DocLayoutV3 can return polygonal regions for warped/skewed pages.
        // This first local pipeline version keeps cropping deliberately simple:
        // the polygon is preserved in the structured output, while recognition
        // uses the axis-aligned bounding box. Perspective rectification should
        // be added as a separate, tested step rather than hidden in the crop.
        let crop = crop_bbox(image, block.bbox)?;
        let result = self.recognizer.recognize_image(crop, task)?;
        Ok(Some(result.text))
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PipelineModelSettings {
    pub use_doc_preprocessor: bool,
    pub use_layout_detection: bool,
    pub use_chart_recognition: bool,
    pub use_seal_recognition: bool,
    pub use_ocr_for_image_block: bool,
    pub format_block_content: bool,
    pub merge_layout_blocks: bool,
    pub markdown_ignore_labels: Vec<String>,
    pub return_layout_polygon_points: bool,
}

impl PipelineModelSettings {
    fn from_config(config: &PipelineConfig) -> Self {
        Self {
            use_doc_preprocessor: false,
            use_layout_detection: true,
            use_chart_recognition: config.use_chart_recognition,
            use_seal_recognition: config.use_seal_recognition,
            use_ocr_for_image_block: config.use_ocr_for_image_block,
            format_block_content: false,
            merge_layout_blocks: true,
            markdown_ignore_labels: config.markdown_ignore_labels.clone(),
            return_layout_polygon_points: true,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ParsedBlock {
    #[serde(rename = "block_label")]
    pub label: String,
    #[serde(rename = "block_content")]
    pub content: String,
    #[serde(rename = "block_bbox")]
    pub bbox: [u32; 4],
    #[serde(rename = "block_id")]
    pub id: usize,
    #[serde(rename = "block_order")]
    pub order: Option<usize>,
    pub group_id: usize,
    pub block_polygon_points: Vec<[f32; 2]>,
    #[serde(skip)]
    pub source_bbox: [f32; 4],
}

#[derive(Clone, Debug, Serialize)]
pub struct PipelineResult {
    pub input_path: Option<PathBuf>,
    #[serde(skip)]
    pub source_path: Option<PathBuf>,
    pub page_index: Option<usize>,
    pub page_count: Option<usize>,
    pub width: u32,
    pub height: u32,
    pub model_settings: PipelineModelSettings,
    #[serde(rename = "parsing_res_list")]
    pub blocks: Vec<ParsedBlock>,
}

impl PipelineResult {
    pub fn print(&self) -> Result<(), String> {
        println!(
            "{}",
            serde_json::to_string_pretty(self).map_err(|e| e.to_string())?
        );
        Ok(())
    }

    pub fn to_json_value(&self) -> Result<serde_json::Value, String> {
        serde_json::to_value(self).map_err(|e| e.to_string())
    }

    pub fn save_to_json(&self, save_path: impl AsRef<Path>) -> Result<(), String> {
        let path = output_path(save_path.as_ref(), self.input_path.as_deref(), "json");
        if let Some(parent) = path.parent() {
            create_dir_all_if_needed(parent)?;
        }
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&path, text).map_err(|e| format!("failed to write {}: {e}", path.display()))
    }

    pub fn to_markdown(&self) -> String {
        let ignored = self
            .model_settings
            .markdown_ignore_labels
            .iter()
            .map(|label| label.as_str())
            .collect::<HashSet<_>>();
        let mut out = String::new();
        for block in &self.blocks {
            if ignored.contains(block.label.as_str()) {
                continue;
            }
            let content = block.content.trim();
            match block.label.as_str() {
                "image" => {
                    out.push_str(&image_block_markdown(block, self.width));
                    out.push_str("\n\n");
                }
                "doc_title" => {
                    if content.is_empty() {
                        continue;
                    }
                    out.push_str("# ");
                    out.push_str(content);
                    out.push_str("\n\n");
                }
                "paragraph_title" | "figure_title" => {
                    if content.is_empty() {
                        continue;
                    }
                    out.push_str("## ");
                    out.push_str(content);
                    out.push_str("\n\n");
                }
                "table" | "chart" | "formula" => {
                    if content.is_empty() {
                        continue;
                    }
                    out.push_str(content);
                    out.push_str("\n\n");
                }
                _ => {
                    if content.is_empty() {
                        continue;
                    }
                    out.push_str(content);
                    out.push_str("\n\n");
                }
            }
        }
        out
    }

    pub fn save_to_markdown(&self, save_path: impl AsRef<Path>) -> Result<(), String> {
        let path = output_path(save_path.as_ref(), self.input_path.as_deref(), "md");
        if let Some(parent) = path.parent() {
            create_dir_all_if_needed(parent)?;
        }
        self.save_image_crops_for_markdown(&path)?;
        std::fs::write(&path, self.to_markdown())
            .map_err(|e| format!("failed to write {}: {e}", path.display()))
    }

    fn save_image_crops_for_markdown(&self, markdown_path: &Path) -> Result<(), String> {
        let image_blocks = self
            .blocks
            .iter()
            .filter(|block| block.label == "image")
            .collect::<Vec<_>>();
        if image_blocks.is_empty() {
            return Ok(());
        }
        let Some(source_path) = &self.source_path else {
            return Ok(());
        };
        let source = image::open(source_path).map_err(|e| {
            format!(
                "failed to open {} for markdown images: {e}",
                source_path.display()
            )
        })?;
        let base_dir = markdown_path.parent().unwrap_or_else(|| Path::new("."));
        let imgs_dir = base_dir.join("imgs");
        create_dir_all_if_needed(&imgs_dir)?;
        for block in image_blocks {
            let crop = crop_bbox(&source, block.source_bbox)?;
            let path = imgs_dir.join(image_block_file_name(block));
            crop.save(&path)
                .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result(source_path: Option<PathBuf>) -> PipelineResult {
        let input_path = Some(PathBuf::from("sample.png"));
        PipelineResult {
            input_path: input_path.clone(),
            source_path,
            page_index: None,
            page_count: None,
            width: 100,
            height: 200,
            model_settings: PipelineModelSettings {
                use_doc_preprocessor: false,
                use_layout_detection: true,
                use_chart_recognition: false,
                use_seal_recognition: true,
                use_ocr_for_image_block: false,
                format_block_content: false,
                merge_layout_blocks: true,
                markdown_ignore_labels: vec!["header".into()],
                return_layout_polygon_points: true,
            },
            blocks: vec![
                ParsedBlock {
                    label: "header".into(),
                    content: "ignored header".into(),
                    bbox: [0, 0, 100, 20],
                    id: 0,
                    order: None,
                    group_id: 0,
                    block_polygon_points: vec![
                        [0.0, 0.0],
                        [100.0, 0.0],
                        [100.0, 20.0],
                        [0.0, 20.0],
                    ],
                    source_bbox: [0.0, 0.0, 100.0, 20.0],
                },
                ParsedBlock {
                    label: "text".into(),
                    content: "kept text".into(),
                    bbox: [0, 20, 100, 80],
                    id: 1,
                    order: Some(1),
                    group_id: 1,
                    block_polygon_points: vec![
                        [0.0, 20.0],
                        [100.0, 20.0],
                        [100.0, 80.0],
                        [0.0, 80.0],
                    ],
                    source_bbox: [0.0, 20.0, 100.0, 80.0],
                },
                ParsedBlock {
                    label: "image".into(),
                    content: String::new(),
                    bbox: [10, 90, 30, 110],
                    id: 2,
                    order: None,
                    group_id: 2,
                    block_polygon_points: vec![
                        [10.0, 90.0],
                        [30.0, 90.0],
                        [30.0, 110.0],
                        [10.0, 110.0],
                    ],
                    source_bbox: [10.0, 90.0, 30.0, 110.0],
                },
            ],
        }
    }

    #[test]
    fn json_uses_reference_fields_and_keeps_ignored_blocks() {
        let value = serde_json::to_value(sample_result(None)).unwrap();

        assert!(value.get("model_settings").is_some());
        assert!(value.get("layout_det_res").is_none());
        assert!(value.get("blocks").is_none());
        assert_eq!(value["page_count"], serde_json::Value::Null);
        assert_eq!(value["width"], 100);
        assert_eq!(value["height"], 200);
        assert_eq!(value["model_settings"]["merge_layout_blocks"], true);
        assert_eq!(
            value["model_settings"]["return_layout_polygon_points"],
            true
        );
        let parsing = value["parsing_res_list"].as_array().unwrap();
        assert_eq!(parsing.len(), 3);
        assert_eq!(parsing[0]["block_label"], "header");
        assert_eq!(parsing[0]["block_content"], "ignored header");
        assert_eq!(parsing[0]["block_order"], serde_json::Value::Null);
        assert_eq!(parsing[0]["group_id"], 0);
        assert!(parsing[0].get("block_label_id").is_none());
        assert!(parsing[0].get("block_score").is_none());
        assert!(parsing[0].get("block_polygon").is_none());
        assert!(parsing[0].get("block_polygon_points").is_some());
    }

    #[test]
    fn markdown_ignore_labels_do_not_affect_json() {
        let result = sample_result(None);
        let markdown = result.to_markdown();
        let json = result.to_json_value().unwrap();

        assert!(!markdown.contains("ignored header"));
        assert!(markdown.contains("kept text"));
        assert!(markdown.contains("imgs/img_in_image_box_10_90_30_110.jpg"));
        assert_eq!(json["parsing_res_list"][0]["block_label"], "header");
    }

    #[test]
    fn output_paths_and_markdown_image_crops_match_reference_style() {
        let base =
            std::env::temp_dir().join(format!("paddleocr_vl_pipeline_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let source_path = base.join("sample.png");
        image::DynamicImage::new_rgb8(40, 140)
            .save(&source_path)
            .unwrap();
        let result = sample_result(Some(source_path));

        result.save_to_json(&base).unwrap();
        assert!(base.join("sample_res.json").exists());

        result.save_to_markdown(&base).unwrap();
        let markdown_path = base.join("sample.md");
        let image_path = base.join("imgs/img_in_image_box_10_90_30_110.jpg");
        assert!(markdown_path.exists());
        assert!(image_path.exists());
        let markdown = std::fs::read_to_string(markdown_path).unwrap();
        assert!(markdown.contains(
            "<div style=\"text-align: center;\"><img src=\"imgs/img_in_image_box_10_90_30_110.jpg\""
        ));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn official_text_like_layout_labels_are_recognized() {
        let config = PipelineConfig::default();

        assert_eq!(task_for_label("header", &config), Some(OcrTask::Ocr));
        assert_eq!(task_for_label("number", &config), Some(OcrTask::Ocr));
        assert_eq!(task_for_label("footnote", &config), Some(OcrTask::Ocr));
        assert_eq!(
            task_for_label("formula_number", &config),
            Some(OcrTask::Ocr)
        );
        assert_eq!(
            task_for_label("vision_footnote", &config),
            Some(OcrTask::Ocr)
        );
        assert_eq!(task_for_label("image", &config), None);
    }
}

fn task_for_label(label: &str, config: &PipelineConfig) -> Option<OcrTask> {
    // Keep text spotting out of the page parser. Spotting is a direct VLM task
    // exposed by the `paddleocr-vl` crate. Official PaddleOCR-VL still fills
    // JSON content for non-text labels like headers, footers, and page numbers,
    // so default layout text regions to OCR unless a specialized task applies.
    match label {
        "table" => Some(OcrTask::Table),
        "chart" => config.use_chart_recognition.then_some(OcrTask::Chart),
        "formula" => Some(OcrTask::Formula),
        "seal" => config.use_seal_recognition.then_some(OcrTask::Seal),
        "image" => config.use_ocr_for_image_block.then_some(OcrTask::Ocr),
        _ => Some(OcrTask::Ocr),
    }
}

fn crop_bbox(image: &DynamicImage, bbox: [f32; 4]) -> Result<DynamicImage, String> {
    let width = image.width();
    let height = image.height();
    let x1 = bbox[0].floor().clamp(0.0, width.saturating_sub(1) as f32) as u32;
    let y1 = bbox[1].floor().clamp(0.0, height.saturating_sub(1) as f32) as u32;
    let x2 = bbox[2].ceil().clamp((x1 + 1) as f32, width as f32) as u32;
    let y2 = bbox[3].ceil().clamp((y1 + 1) as f32, height as f32) as u32;
    if x2 <= x1 || y2 <= y1 {
        return Err(format!("invalid crop bbox: {bbox:?}"));
    }
    Ok(image.crop_imm(x1, y1, x2 - x1, y2 - y1))
}

fn bbox_to_u32(bbox: [f32; 4]) -> [u32; 4] {
    [
        bbox[0].round().max(0.0) as u32,
        bbox[1].round().max(0.0) as u32,
        bbox[2].round().max(0.0) as u32,
        bbox[3].round().max(0.0) as u32,
    ]
}

fn polygon_points(polygon: Option<Vec<[f32; 2]>>, bbox: [u32; 4]) -> Vec<[f32; 2]> {
    polygon.unwrap_or_else(|| {
        vec![
            [bbox[0] as f32, bbox[1] as f32],
            [bbox[2] as f32, bbox[1] as f32],
            [bbox[2] as f32, bbox[3] as f32],
            [bbox[0] as f32, bbox[3] as f32],
        ]
    })
}

fn input_file_name(path: &Path) -> Option<PathBuf> {
    path.file_name()
        .map(|name| PathBuf::from(name.to_os_string()))
}

fn create_dir_all_if_needed(path: &Path) -> Result<(), String> {
    if path.as_os_str().is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(path).map_err(|e| format!("failed to create {}: {e}", path.display()))
}

fn output_path(save_path: &Path, input_path: Option<&Path>, extension: &str) -> PathBuf {
    if save_path.extension().is_some() {
        return save_path.to_path_buf();
    }

    let stem = input_path
        .and_then(|path| path.file_stem())
        .and_then(|stem| stem.to_str())
        .unwrap_or("output");
    let file_name = if extension == "json" {
        format!("{stem}_res.{extension}")
    } else {
        format!("{stem}.{extension}")
    };
    save_path.join(file_name)
}

fn image_block_file_name(block: &ParsedBlock) -> String {
    let [x1, y1, x2, y2] = block.bbox;
    format!("img_in_image_box_{x1}_{y1}_{x2}_{y2}.jpg")
}

fn image_block_markdown(block: &ParsedBlock, page_width: u32) -> String {
    let image_width = block.bbox[2].saturating_sub(block.bbox[0]);
    let percent = if page_width == 0 {
        100
    } else {
        ((image_width as f32 / page_width as f32) * 100.0)
            .round()
            .clamp(1.0, 100.0) as u32
    };
    format!(
        "<div style=\"text-align: center;\"><img src=\"imgs/{}\" alt=\"Image\" width=\"{}%\" /></div>",
        image_block_file_name(block),
        percent
    )
}

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| IMAGE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
}

fn collect_images(input: &Path) -> Result<Vec<PathBuf>, String> {
    if input.is_dir() {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(input)
            .map_err(|e| format!("failed to read directory {}: {e}", input.display()))?
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                (path.is_file() && is_image_file(&path)).then_some(path)
            })
            .collect();
        paths.sort();
        Ok(paths)
    } else if input.is_file() {
        if is_image_file(input) {
            Ok(vec![input.to_path_buf()])
        } else {
            Err(format!("{} is not a supported image file", input.display()))
        }
    } else {
        Err(format!(
            "{} is not a valid file or directory",
            input.display()
        ))
    }
}
