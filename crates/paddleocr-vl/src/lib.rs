#![recursion_limit = "256"]

use std::{
    path::{Path, PathBuf},
    str::FromStr,
    sync::OnceLock,
};

use burn::{
    Tensor,
    prelude::Backend,
    tensor::{DType, TensorData},
};
use image::{DynamicImage, imageops::FilterType};
use regex::Regex;
use serde::Deserialize;
use tokenizers::Tokenizer;

pub mod load_adapter;
pub mod paddleocr_vl;

pub use paddleocr_vl::{PaddleOcrVlModel, PaddleOcrVlModelConfig};

const V1_0_MIN_PIXELS: usize = 147_384;
const V1_0_MAX_PIXELS: usize = 2_822_400;
const V1_5_MIN_PIXELS: usize = 112_896;
const V1_5_MAX_PIXELS: usize = 1_003_520;
const SPOTTING_MAX_PIXELS: usize = 2_048 * 28 * 28;
const SPOTTING_UPSCALE_THRESHOLD: u32 = 1_500;
const CLS_TOKEN: &str = "<|begin_of_sentence|>";
const EOS_TOKEN: &str = "</s>";
const IMAGE_START_TOKEN: &str = "<|IMAGE_START|>";
const IMAGE_PLACEHOLDER_TOKEN: &str = "<|IMAGE_PLACEHOLDER|>";
const IMAGE_END_TOKEN: &str = "<|IMAGE_END|>";
const PADDLEOCR_VL_SPOTTING_MAX_NEW_TOKENS: usize = 8192;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PaddleOcrVersion {
    #[default]
    V1_0,
    V1_5,
}

impl PaddleOcrVersion {
    pub fn default_model_dir(self) -> &'static str {
        match self {
            Self::V1_0 => "./model1.0",
            Self::V1_5 => "./model1.5",
        }
    }
}

impl FromStr for PaddleOcrVersion {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "1.0" | "v1.0" | "vl-1.0" => Ok(Self::V1_0),
            "1.5" | "v1.5" | "vl-1.5" => Ok(Self::V1_5),
            other => Err(format!(
                "unsupported model version '{other}', expected 1.0 or 1.5"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OcrTask {
    #[default]
    Ocr,
    Table,
    Chart,
    Formula,
    Spotting,
    Seal,
}

impl OcrTask {
    pub fn prompt(self) -> &'static str {
        match self {
            Self::Ocr => "OCR:",
            Self::Table => "Table Recognition:",
            Self::Chart => "Chart Recognition:",
            Self::Formula => "Formula Recognition:",
            Self::Spotting => "Spotting:",
            Self::Seal => "Seal Recognition:",
        }
    }

    pub fn is_spotting(self) -> bool {
        self == Self::Spotting
    }
}

impl FromStr for OcrTask {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "ocr" => Ok(Self::Ocr),
            "table" => Ok(Self::Table),
            "chart" => Ok(Self::Chart),
            "formula" => Ok(Self::Formula),
            "spotting" => Ok(Self::Spotting),
            "seal" => Ok(Self::Seal),
            other => Err(format!(
                "unsupported task '{other}', expected one of: ocr, table, chart, formula, spotting, seal"
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RecognitionResult {
    pub task: OcrTask,
    pub text: String,
    pub spotting: Option<SpottingResult>,
    pub generated_tokens: Vec<u32>,
    pub input_size: (u32, u32),
    pub processed_size: (u32, u32),
    pub image_grid_hw: [usize; 2],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpottingResult {
    pub rec_polys: Vec<[[f32; 2]; 4]>,
    pub rec_texts: Vec<String>,
}

impl RecognitionResult {
    pub fn print(&self) {
        println!("{}", self.text);
    }

    pub fn as_text(&self) -> &str {
        &self.text
    }
}

pub struct PaddleOcrVlBuilder<B: Backend> {
    device: B::Device,
    version: PaddleOcrVersion,
    model_dir: Option<PathBuf>,
    dtype: DType,
    max_new_tokens: Option<usize>,
}

impl<B: Backend> PaddleOcrVlBuilder<B> {
    pub fn version(mut self, version: PaddleOcrVersion) -> Self {
        self.version = version;
        self
    }

    pub fn model_dir(mut self, model_dir: impl Into<PathBuf>) -> Self {
        self.model_dir = Some(model_dir.into());
        self
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = Some(max_new_tokens);
        self
    }

    pub fn build(self) -> Result<PaddleOcrVl<B>, String> {
        let model_dir = self
            .model_dir
            .unwrap_or_else(|| PathBuf::from(self.version.default_model_dir()));
        let cfg = load_processor_cfg(&model_dir, self.version)?;
        let model = PaddleOcrVlModelConfig::for_version(self.version)
            .try_init_from_safetensors_with_dtype::<B>(
                model_dir.join("model.safetensors"),
                &self.device,
                self.dtype,
            )?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| format!("failed to load tokenizer: {e}"))?;

        Ok(PaddleOcrVl {
            device: self.device,
            version: self.version,
            dtype: self.dtype,
            cfg,
            model,
            tokenizer,
            max_new_tokens: self.max_new_tokens,
        })
    }
}

pub struct PaddleOcrVl<B: Backend> {
    device: B::Device,
    version: PaddleOcrVersion,
    dtype: DType,
    cfg: PaddleOcrCfg,
    model: PaddleOcrVlModel<B>,
    tokenizer: Tokenizer,
    max_new_tokens: Option<usize>,
}

impl<B: Backend> PaddleOcrVl<B> {
    pub fn builder(device: B::Device) -> PaddleOcrVlBuilder<B> {
        PaddleOcrVlBuilder {
            device,
            version: PaddleOcrVersion::default(),
            model_dir: None,
            dtype: DType::F32,
            max_new_tokens: None,
        }
    }

    pub fn recognize_path(
        &self,
        path: impl AsRef<Path>,
        task: OcrTask,
    ) -> Result<RecognitionResult, String> {
        let img = image::open(path.as_ref())
            .map_err(|e| format!("failed to open {}: {e}", path.as_ref().display()))?;
        self.recognize_image(img, task)
    }

    pub fn recognize_image(
        &self,
        img: DynamicImage,
        task: OcrTask,
    ) -> Result<RecognitionResult, String> {
        self.recognize_image_with_max_new_tokens(
            img,
            task,
            self.default_max_new_tokens_for_task(task),
        )
    }

    pub fn recognize_image_with_max_new_tokens(
        &self,
        img: DynamicImage,
        task: OcrTask,
        max_new_tokens: usize,
    ) -> Result<RecognitionResult, String> {
        if task.is_spotting() && self.version != PaddleOcrVersion::V1_5 {
            return Err("spotting is only supported by the PaddleOCR-VL 1.5 recipe".into());
        }
        let input_size = (img.width(), img.height());
        let processed = prepare_image_for_task(img, task);
        let processed_size = (processed.width(), processed.height());
        let cfg = self.cfg.clone().with_task(task);

        let ImageOutput {
            pixel_values,
            image_grid_hw,
        } = preprocess_image::<B>(&processed, &cfg, &self.device, self.dtype)?;
        let pixel_values = pixel_values.unsqueeze_dim::<5>(0);
        let [grid_h, grid_w] = image_grid_hw;

        let messages = [Message {
            role: "user",
            content: vec![Content::Image(&processed), Content::Text(task.prompt())],
        }];
        let prompt = render_chat(&messages, true, &cfg);
        let expanded = expand_image_placeholders(&prompt, &[image_grid_hw], &cfg)?;
        let encoding = self
            .tokenizer
            .encode(expanded.as_str(), false)
            .map_err(|e| format!("tokenization failed: {e}"))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();

        let generated = self.model.generate(
            pixel_values,
            &[(grid_h, grid_w)],
            &input_ids,
            max_new_tokens,
        );
        let raw_text = self.tokenizer.decode(&generated, false).unwrap_or_default();
        let (text, spotting) = if task.is_spotting() {
            let (text, spotting) =
                post_process_for_spotting(&raw_text, processed.width(), processed.height());
            (text, Some(spotting))
        } else {
            (raw_text, None)
        };

        Ok(RecognitionResult {
            task,
            text,
            spotting,
            generated_tokens: generated,
            input_size,
            processed_size,
            image_grid_hw,
        })
    }

    pub fn recognize(
        &self,
        path: impl AsRef<Path>,
        task: OcrTask,
    ) -> Result<RecognitionResult, String> {
        self.recognize_path(path, task)
    }

    pub fn ocr(&self, path: impl AsRef<Path>) -> Result<RecognitionResult, String> {
        self.recognize_path(path, OcrTask::Ocr)
    }

    pub fn table(&self, path: impl AsRef<Path>) -> Result<RecognitionResult, String> {
        self.recognize_path(path, OcrTask::Table)
    }

    pub fn chart(&self, path: impl AsRef<Path>) -> Result<RecognitionResult, String> {
        self.recognize_path(path, OcrTask::Chart)
    }

    pub fn formula(&self, path: impl AsRef<Path>) -> Result<RecognitionResult, String> {
        self.recognize_path(path, OcrTask::Formula)
    }

    pub fn seal(&self, path: impl AsRef<Path>) -> Result<RecognitionResult, String> {
        self.recognize_path(path, OcrTask::Seal)
    }

    pub fn spotting(&self, path: impl AsRef<Path>) -> Result<RecognitionResult, String> {
        self.recognize_path(path, OcrTask::Spotting)
    }

    pub fn default_max_new_tokens(&self) -> usize {
        self.max_new_tokens.unwrap_or(match self.version {
            PaddleOcrVersion::V1_0 => 1024,
            PaddleOcrVersion::V1_5 => 512,
        })
    }

    pub fn default_max_new_tokens_for_task(&self, task: OcrTask) -> usize {
        self.max_new_tokens.unwrap_or_else(|| {
            if task.is_spotting() {
                PADDLEOCR_VL_SPOTTING_MAX_NEW_TOKENS
            } else {
                self.default_max_new_tokens()
            }
        })
    }
}

pub fn load_processor_cfg(
    model_dir: impl AsRef<Path>,
    version: PaddleOcrVersion,
) -> Result<PaddleOcrCfg, String> {
    let model_dir = model_dir.as_ref();
    let base = PaddleOcrCfg::for_version(version);
    let preprocessor_path = model_dir.join("preprocessor_config.json");
    if preprocessor_path.exists() {
        base.apply_preprocessor_config_file(&preprocessor_path)
    } else {
        Ok(base)
    }
}

#[derive(Clone, Debug)]
pub struct PaddleOcrCfg {
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub rescale_factor: f32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub image_token: &'static str,
    pub image_token_id: u32,
    pub cls_token: &'static str,
    pub eos_token: &'static str,
    pub assistant_prefix: &'static str,
}

impl Default for PaddleOcrCfg {
    fn default() -> Self {
        Self::for_version(PaddleOcrVersion::V1_0)
    }
}

impl PaddleOcrCfg {
    pub fn for_version(version: PaddleOcrVersion) -> Self {
        let (min_pixels, max_pixels, assistant_prefix) = match version {
            PaddleOcrVersion::V1_0 => (V1_0_MIN_PIXELS, V1_0_MAX_PIXELS, "Assistant: "),
            // PaddleOCR-VL-1.5 changed the processor default resolution and the
            // generation prompt in chat_template.jinja to "Assistant:\n".
            PaddleOcrVersion::V1_5 => (V1_5_MIN_PIXELS, V1_5_MAX_PIXELS, "Assistant:\n"),
        };

        Self {
            min_pixels,
            max_pixels,
            patch_size: 14,
            merge_size: 2,
            temporal_patch_size: 1,
            rescale_factor: 1.0 / 255.0,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_token: IMAGE_PLACEHOLDER_TOKEN,
            image_token_id: 100295,
            cls_token: CLS_TOKEN,
            eos_token: EOS_TOKEN,
            assistant_prefix,
        }
    }

    pub fn with_task(mut self, task: OcrTask) -> Self {
        if task.is_spotting() {
            self.max_pixels = SPOTTING_MAX_PIXELS;
        }
        self
    }

    pub fn apply_preprocessor_config_file(
        mut self,
        path: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        let parsed: PreprocessorConfigFile = serde_json::from_str(&text)
            .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;
        parsed.apply_to(&mut self);
        Ok(self)
    }
}

#[derive(Debug, Deserialize)]
struct PreprocessorConfigFile {
    min_pixels: Option<usize>,
    max_pixels: Option<usize>,
    size: Option<PreprocessorSize>,
    patch_size: Option<usize>,
    temporal_patch_size: Option<usize>,
    merge_size: Option<usize>,
    rescale_factor: Option<f32>,
    image_mean: Option<[f32; 3]>,
    image_std: Option<[f32; 3]>,
}

impl PreprocessorConfigFile {
    fn apply_to(self, cfg: &mut PaddleOcrCfg) {
        if let Some(min_pixels) = self
            .min_pixels
            .or_else(|| self.size.as_ref().and_then(|size| size.shortest_edge))
            .or_else(|| self.size.as_ref().and_then(|size| size.min_pixels))
        {
            cfg.min_pixels = min_pixels;
        }
        if let Some(max_pixels) = self
            .max_pixels
            .or_else(|| self.size.as_ref().and_then(|size| size.longest_edge))
            .or_else(|| self.size.as_ref().and_then(|size| size.max_pixels))
        {
            cfg.max_pixels = max_pixels;
        }
        if let Some(patch_size) = self.patch_size {
            cfg.patch_size = patch_size;
        }
        if let Some(temporal_patch_size) = self.temporal_patch_size {
            cfg.temporal_patch_size = temporal_patch_size;
        }
        if let Some(merge_size) = self.merge_size {
            cfg.merge_size = merge_size;
        }
        if let Some(rescale_factor) = self.rescale_factor {
            cfg.rescale_factor = rescale_factor;
        }
        if let Some(image_mean) = self.image_mean {
            cfg.image_mean = image_mean;
        }
        if let Some(image_std) = self.image_std {
            cfg.image_std = image_std;
        }
    }
}

#[derive(Debug, Deserialize)]
struct PreprocessorSize {
    shortest_edge: Option<usize>,
    longest_edge: Option<usize>,
    min_pixels: Option<usize>,
    max_pixels: Option<usize>,
}

pub enum Content<'a> {
    Text(&'a str),
    Image(&'a DynamicImage),
}

pub struct Message<'a> {
    pub role: &'a str, // "user" | "assistant" | "system"
    pub content: Vec<Content<'a>>,
}

pub fn render_chat(
    messages: &[Message],
    add_generation_prompt: bool,
    cfg: &PaddleOcrCfg,
) -> String {
    // Manual rendering of PaddleOCR-VL's chat_template.jinja. Keep the two-pass
    // image-then-text order and exact assistant prefixes: v1.0 uses
    // "Assistant: ", while v1.5 uses "Assistant:\n".
    let mut out = String::from(cfg.cls_token);
    for msg in messages {
        match msg.role {
            "user" => {
                out.push_str("User: ");
                for c in &msg.content {
                    if matches!(c, Content::Image(_)) {
                        out.push_str(IMAGE_START_TOKEN);
                        out.push_str(IMAGE_PLACEHOLDER_TOKEN);
                        out.push_str(IMAGE_END_TOKEN);
                    }
                }
                for c in &msg.content {
                    if let Content::Text(t) = c {
                        out.push_str(t);
                    }
                }
                out.push('\n');
            }
            "assistant" => {
                out.push_str(cfg.assistant_prefix);
                for c in &msg.content {
                    if let Content::Text(t) = c {
                        out.push_str(t);
                    }
                }
                out.push_str(cfg.eos_token);
            }
            "system" => {
                for c in &msg.content {
                    if let Content::Text(t) = c {
                        out.push_str(t);
                        out.push('\n');
                    }
                }
            }
            _ => panic!("unsupported role"),
        }
    }
    if add_generation_prompt {
        out.push_str(cfg.assistant_prefix);
    }
    out
}

// Python round(): bankers rounding. HuggingFace's image processor relies on
// this behavior when snapping resized dimensions to the patch/merge factor.
fn py_round(x: f64) -> usize {
    let floor = x.floor();
    let frac = x - floor;
    if frac < 0.5 {
        floor as usize
    } else if frac > 0.5 {
        (floor + 1.0) as usize
    } else {
        let f = floor as i64;
        if f % 2 == 0 {
            f as usize
        } else {
            (f + 1) as usize
        }
    }
}

pub fn prepare_image_for_task(img: DynamicImage, task: OcrTask) -> DynamicImage {
    let (orig_w, orig_h) = (img.width(), img.height());

    // PaddleOCR-VL-1.5's spotting recipe doubles small images before normal
    // processor resizing. Keep this outside `preprocess_image` so other tasks
    // and PaddleOCR-VL-1.0 keep their existing preprocessing behavior.
    if task.is_spotting()
        && orig_w < SPOTTING_UPSCALE_THRESHOLD
        && orig_h < SPOTTING_UPSCALE_THRESHOLD
    {
        return img.resize_exact(orig_w * 2, orig_h * 2, FilterType::Lanczos3);
    }

    img
}

fn annot_text_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)<\|TEXT_START\|>(.*?)<\|TEXT_END\|>").unwrap())
}

fn loc_block_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)<\|LOC_BEGIN\|>(.*?)<\|LOC_END\|>").unwrap())
}

fn loc_token_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<\|LOC_(\d+)\|>").unwrap())
}

pub fn post_process_for_spotting(input: &str, width: u32, height: u32) -> (String, SpottingResult) {
    fn points_from_tokens(vals: &[u32], width: u32, height: u32) -> [[f32; 2]; 4] {
        let mut pts = [[0.0; 2]; 4];
        for (point, xy) in pts.iter_mut().zip(vals.chunks_exact(2)) {
            point[0] = xy[0] as f32 / 1000.0 * width as f32;
            point[1] = xy[1] as f32 / 1000.0 * height as f32;
        }
        pts
    }

    let texts: Vec<_> = annot_text_re()
        .captures_iter(input)
        .map(|caps| caps[1].trim().to_owned())
        .collect();
    let loc_blocks: Vec<_> = loc_block_re()
        .captures_iter(input)
        .map(|caps| caps[1].to_owned())
        .collect();

    let mut rec_texts = Vec::new();
    let mut rec_polys = Vec::new();

    for (text, block) in texts.into_iter().zip(loc_blocks.iter()) {
        let vals: Vec<u32> = loc_token_re()
            .captures_iter(block)
            .filter_map(|caps| caps[1].parse().ok())
            .take(8)
            .collect();
        if vals.len() < 8 {
            continue;
        }
        rec_texts.push(text);
        rec_polys.push(points_from_tokens(&vals, width, height));
    }

    if rec_polys.is_empty() || rec_texts.is_empty() {
        let matches: Vec<_> = loc_token_re().find_iter(input).collect();
        let mut last_end = 0;
        let mut i = 0;
        while i + 7 < matches.len() {
            let group = &matches[i..i + 8];
            let vals: Vec<u32> = group
                .iter()
                .filter_map(|m| {
                    loc_token_re()
                        .captures(m.as_str())
                        .and_then(|caps| caps[1].parse().ok())
                })
                .collect();
            if vals.len() == 8 {
                rec_texts.push(input[last_end..group[0].start()].trim().to_owned());
                rec_polys.push(points_from_tokens(&vals, width, height));
                last_end = group[7].end();
            }
            i += 8;
        }
    }

    let text = rec_texts.join("\n\n");
    (
        text,
        SpottingResult {
            rec_polys,
            rec_texts,
        },
    )
}

pub fn smart_resize(
    mut height: usize,
    mut width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    if height < factor {
        width = py_round((width as f64 * factor as f64) / height as f64);
        height = factor;
    }
    if width < factor {
        height = py_round((height as f64 * factor as f64) / width as f64);
        width = factor;
    }
    let aspect = (height.max(width) as f64) / (height.min(width) as f64);
    if aspect > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {aspect}"
        ));
    }

    let mut h_bar = py_round(height as f64 / factor as f64) * factor;
    let mut w_bar = py_round(width as f64 / factor as f64) * factor;

    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = factor.max(((height as f64 / beta / factor as f64).floor() as usize) * factor);
        w_bar = factor.max(((width as f64 / beta / factor as f64).floor() as usize) * factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
        w_bar = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
    }
    Ok((h_bar, w_bar))
}

pub struct ImageOutput<B: Backend> {
    pub pixel_values: Tensor<B, 4>, // flattened [num_patches, 3, patch, patch]
    pub image_grid_hw: [usize; 2],  // [h, w]
}

pub fn preprocess_image<B: Backend>(
    img: &DynamicImage,
    cfg: &PaddleOcrCfg,
    device: &B::Device,
    target_dtype: DType,
) -> Result<ImageOutput<B>, String> {
    if cfg.temporal_patch_size != 1 {
        return Err(
            "HF PaddleOCRVLImageProcessor currently errors when temporal_patch_size != 1".into(),
        );
    }

    let channel = 3;
    let image_mean = Tensor::<B, 1>::from_data(
        TensorData::new(cfg.image_mean.to_vec(), [cfg.image_mean.len()]).convert_dtype(DType::F32),
        (device, DType::F32),
    )
    .reshape([1, 1, -1]);
    let image_std = Tensor::<B, 1>::from_data(
        TensorData::new(cfg.image_std.to_vec(), [cfg.image_std.len()]).convert_dtype(DType::F32),
        (device, DType::F32),
    )
    .reshape([1, 1, -1]);

    let rgb = img.to_rgb8();
    let (origin_width, origin_height) = rgb.dimensions();
    let factor = cfg.patch_size * cfg.merge_size;
    let (resized_height, resized_width) = smart_resize(
        origin_height as usize,
        origin_width as usize,
        factor,
        cfg.min_pixels,
        cfg.max_pixels,
    )?;

    let resized = image::imageops::resize(
        &rgb,
        resized_width as u32,
        resized_height as u32,
        FilterType::CatmullRom,
    );
    let img_data = resized
        .into_vec()
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    let img_tensor = Tensor::<B, 1>::from_data(
        TensorData::new(img_data, [resized_height * resized_width * 3]).convert_dtype(DType::F32),
        (device, DType::F32),
    )
    .reshape([resized_height, resized_width, 3]);
    let rescaled_tensor = img_tensor * cfg.rescale_factor;
    let rescaled_tensor = ((rescaled_tensor - image_mean) / image_std).cast(target_dtype);
    let channel_first_tensor = rescaled_tensor.permute([2, 0, 1]);

    let grid_h = resized_height / cfg.patch_size;
    let grid_w = resized_width / cfg.patch_size;
    let patches =
        channel_first_tensor.reshape([channel, grid_h, cfg.patch_size, grid_w, cfg.patch_size]);
    let patches = patches.permute([1, 3, 0, 2, 4]);
    let flatten_patches =
        patches.reshape([grid_h * grid_w, channel, cfg.patch_size, cfg.patch_size]);

    Ok(ImageOutput {
        pixel_values: flatten_patches,
        image_grid_hw: [grid_h, grid_w],
    })
}

pub fn expand_image_placeholders(
    text: &str,
    image_grids: &[[usize; 2]],
    cfg: &PaddleOcrCfg,
) -> Result<String, String> {
    let mut out = text.to_string();
    for grid in image_grids {
        let n = grid[0] * grid[1] / (cfg.merge_size * cfg.merge_size);
        let replacement = cfg.image_token.repeat(n);
        if !out.contains(cfg.image_token) {
            return Err("image placeholder count < image count".into());
        }
        out = out.replacen(cfg.image_token, &replacement, 1);
    }
    Ok(out)
}

pub fn build_mm_token_type_ids(input_ids: &[u32], image_token_id: u32) -> Vec<u8> {
    input_ids
        .iter()
        .map(|&id| if id == image_token_id { 1 } else { 0 })
        .collect()
}

/// Result of computing M-RoPE 3D position IDs
pub struct RopeIndex {
    pub temporal: Vec<i64>,
    pub height: Vec<i64>,
    pub width: Vec<i64>,
    /// max(all positions) + 1 − seq_len, used as the position offset during generation
    pub rope_deltas: i64,
}

/// Compute the 3D position_ids for M-RoPE (corresponds to the Python-side `get_rope_index`)
///
/// - `input_ids`: token ID sequence (batch_size=1)
/// - `image_grid_thw`: (temporal, grid_h, grid_w) for each image, where grid_h/grid_w are the vision-layer output grid dimensions
/// - `image_token_id`: image placeholder token (100295)
/// - `vision_start_token_id`: `<|IMAGE_START|>` token (101305)
/// - `spatial_merge_size`: spatial merge factor used by the projector (2)
pub fn get_rope_index(
    input_ids: &[u32],
    image_grid_thw: &[(usize, usize, usize)],
    image_token_id: u32,
    vision_start_token_id: u32,
    spatial_merge_size: usize,
) -> RopeIndex {
    let seq_len = input_ids.len();

    if image_grid_thw.is_empty() {
        let ids: Vec<i64> = (0..seq_len as i64).collect();
        return RopeIndex {
            temporal: ids.clone(),
            height: ids.clone(),
            width: ids,
            rope_deltas: 0,
        };
    }

    let image_nums = input_ids
        .windows(2)
        .filter(|w| w[0] == vision_start_token_id && w[1] == image_token_id)
        .count();
    assert_eq!(
        image_nums,
        image_grid_thw.len(),
        "image count ({image_nums}) != image_grid_thw count ({})",
        image_grid_thw.len()
    );

    let mut st = 0usize;
    let mut chunks_t: Vec<Vec<i64>> = Vec::new();
    let mut chunks_h: Vec<Vec<i64>> = Vec::new();
    let mut chunks_w: Vec<Vec<i64>> = Vec::new();

    let max_prev = |ct: &[Vec<i64>], ch: &[Vec<i64>], cw: &[Vec<i64>]| -> i64 {
        let mt = ct
            .last()
            .and_then(|v| v.iter().max().copied())
            .unwrap_or(-1);
        let mh = ch
            .last()
            .and_then(|v| v.iter().max().copied())
            .unwrap_or(-1);
        let mw = cw
            .last()
            .and_then(|v| v.iter().max().copied())
            .unwrap_or(-1);
        mt.max(mh).max(mw)
    };

    for (image_index, _) in (0..image_nums).enumerate() {
        let ed = input_ids[st..]
            .iter()
            .position(|&id| id == image_token_id)
            .unwrap()
            + st;

        let (t, h, w) = image_grid_thw[image_index];
        let llm_grid_t = t;
        let llm_grid_h = h / spatial_merge_size;
        let llm_grid_w = w / spatial_merge_size;

        let text_len = ed - st;
        let st_idx = max_prev(&chunks_t, &chunks_h, &chunks_w) + 1;

        // text before this image
        let text_ids: Vec<i64> = (0..text_len as i64).map(|i| i + st_idx).collect();
        chunks_t.push(text_ids.clone());
        chunks_h.push(text_ids.clone());
        chunks_w.push(text_ids);

        // image tokens: 3D spatial positions
        let n_vis = llm_grid_t * llm_grid_h * llm_grid_w;
        let mut ti = Vec::with_capacity(n_vis);
        let mut hi = Vec::with_capacity(n_vis);
        let mut wi = Vec::with_capacity(n_vis);
        let offset = text_len as i64 + st_idx;

        for _t in 0..llm_grid_t {
            for h_i in 0..llm_grid_h {
                for w_i in 0..llm_grid_w {
                    // For single-frame images, the temporal coordinate is zero
                    // before the text offset is applied. This matches the HF
                    // M-RoPE convention where text and vision positions share
                    // one monotonically increasing coordinate space.
                    ti.push(offset);
                    hi.push(h_i as i64 + offset);
                    wi.push(w_i as i64 + offset);
                }
            }
        }
        chunks_t.push(ti);
        chunks_h.push(hi);
        chunks_w.push(wi);

        st = ed + n_vis;
    }

    // remaining text after last image
    if st < seq_len {
        let st_idx = max_prev(&chunks_t, &chunks_h, &chunks_w) + 1;
        let text_len = seq_len - st;
        let text_ids: Vec<i64> = (0..text_len as i64).map(|i| i + st_idx).collect();
        chunks_t.push(text_ids.clone());
        chunks_h.push(text_ids.clone());
        chunks_w.push(text_ids);
    }

    let flat = |chunks: &[Vec<i64>]| -> Vec<i64> {
        let mut v = Vec::with_capacity(seq_len);
        for c in chunks {
            v.extend_from_slice(c);
        }
        v
    };
    let temporal = flat(&chunks_t);
    let height = flat(&chunks_h);
    let width = flat(&chunks_w);

    assert_eq!(temporal.len(), seq_len);

    let max_pos = *temporal
        .iter()
        .chain(height.iter())
        .chain(width.iter())
        .max()
        .unwrap();
    let rope_deltas = max_pos + 1 - seq_len as i64;

    RopeIndex {
        temporal,
        height,
        width,
        rope_deltas,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_presets_keep_processor_and_template_differences() {
        let v1_0 = PaddleOcrCfg::for_version(PaddleOcrVersion::V1_0);
        let v1_5 = PaddleOcrCfg::for_version(PaddleOcrVersion::V1_5);

        assert_eq!(v1_0.min_pixels, V1_0_MIN_PIXELS);
        assert_eq!(v1_0.max_pixels, V1_0_MAX_PIXELS);
        assert_eq!(v1_0.assistant_prefix, "Assistant: ");

        assert_eq!(v1_5.min_pixels, V1_5_MIN_PIXELS);
        assert_eq!(v1_5.max_pixels, V1_5_MAX_PIXELS);
        assert_eq!(v1_5.assistant_prefix, "Assistant:\n");
    }

    #[test]
    fn render_chat_matches_official_template_for_generation_prompt() {
        let image = DynamicImage::new_rgb8(8, 8);
        let messages = [Message {
            role: "user",
            content: vec![
                Content::Image(&image),
                Content::Text(OcrTask::Seal.prompt()),
            ],
        }];

        let v1_0 = render_chat(
            &messages,
            true,
            &PaddleOcrCfg::for_version(PaddleOcrVersion::V1_0),
        );
        let v1_5 = render_chat(
            &messages,
            true,
            &PaddleOcrCfg::for_version(PaddleOcrVersion::V1_5),
        );

        assert_eq!(
            v1_0,
            "<|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>Seal Recognition:\nAssistant: "
        );
        assert_eq!(
            v1_5,
            "<|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>Seal Recognition:\nAssistant:\n"
        );
    }

    #[test]
    fn render_chat_matches_official_template_for_assistant_message() {
        let messages = [Message {
            role: "assistant",
            content: vec![Content::Text("result")],
        }];

        let v1_0 = render_chat(
            &messages,
            false,
            &PaddleOcrCfg::for_version(PaddleOcrVersion::V1_0),
        );
        let v1_5 = render_chat(
            &messages,
            false,
            &PaddleOcrCfg::for_version(PaddleOcrVersion::V1_5),
        );

        assert_eq!(v1_0, "<|begin_of_sentence|>Assistant: result</s>");
        assert_eq!(v1_5, "<|begin_of_sentence|>Assistant:\nresult</s>");
    }

    #[test]
    fn spotting_upscales_small_images_only() {
        let small = prepare_image_for_task(DynamicImage::new_rgb8(640, 480), OcrTask::Spotting);
        let large = prepare_image_for_task(DynamicImage::new_rgb8(1800, 480), OcrTask::Spotting);
        let ocr = prepare_image_for_task(DynamicImage::new_rgb8(640, 480), OcrTask::Ocr);

        assert_eq!((small.width(), small.height()), (1280, 960));
        assert_eq!((large.width(), large.height()), (1800, 480));
        assert_eq!((ocr.width(), ocr.height()), (640, 480));
    }

    #[test]
    fn post_process_for_spotting_parses_official_blocks() {
        let (text, spotting) = post_process_for_spotting(
            "<|TEXT_START|>你好<|TEXT_END|><|LOC_BEGIN|><|LOC_0|><|LOC_0|><|LOC_500|><|LOC_0|><|LOC_500|><|LOC_250|><|LOC_0|><|LOC_250|><|LOC_END|>",
            200,
            100,
        );

        assert_eq!(text, "你好");
        assert_eq!(spotting.rec_texts, vec!["你好"]);
        assert_eq!(
            spotting.rec_polys,
            vec![[[0.0, 0.0], [100.0, 0.0], [100.0, 25.0], [0.0, 25.0]]]
        );
    }

    #[test]
    fn post_process_for_spotting_falls_back_to_loc_tokens() {
        let (text, spotting) = post_process_for_spotting(
            "fallback text <|LOC_100|><|LOC_200|><|LOC_300|><|LOC_200|><|LOC_300|><|LOC_400|><|LOC_100|><|LOC_400|>",
            1000,
            500,
        );

        assert_eq!(text, "fallback text");
        assert_eq!(spotting.rec_texts, vec!["fallback text"]);
        assert_eq!(
            spotting.rec_polys,
            vec![[
                [100.0, 100.0],
                [300.0, 100.0],
                [300.0, 200.0],
                [100.0, 200.0]
            ]]
        );
    }
}
