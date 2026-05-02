use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{DType, Int, Tensor, activation, backend::Backend, module::attention, s},
};

#[derive(Module, Debug)]
pub struct PaddleOcrVisionMLP<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: Backend> PaddleOcrVisionMLP<B> {
    /// MLP forward: fc1 → GELU → fc2.
    pub fn forward(&self, hidden_states: Tensor<B, 2>) -> Tensor<B, 2> {
        let output = self.fc1.forward(hidden_states);
        let output = activation::gelu(output);
        self.fc2.forward(output)
    }
}

#[derive(Config, Debug)]
pub struct PaddleOcrVisionMLPConfig {
    #[config(default = 1152)]
    pub hidden_size: usize,
    #[config(default = 4304)]
    pub intermediate_size: usize,
}

impl PaddleOcrVisionMLPConfig {
    /// Build a `PaddleOcrVisionMLP` layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PaddleOcrVisionMLP<B> {
        PaddleOcrVisionMLP {
            fc1: LinearConfig::new(self.hidden_size, self.intermediate_size).init(device),
            fc2: LinearConfig::new(self.intermediate_size, self.hidden_size).init(device),
        }
    }
}

pub struct VisionPositionEmbeddings<B: Backend> {
    pub cos: Tensor<B, 2>,
    pub sin: Tensor<B, 2>,
}

pub struct PaddleOCRVisionEncoderOutput<B: Backend> {
    pub last_hidden_state: Tensor<B, 2>,
}

#[derive(Module, Debug)]
pub struct PaddleOCRVisionRotaryEmbedding<B: Backend> {
    pub inv_freq: Tensor<B, 1>,
    pub dim: usize,
}

impl<B: Backend> PaddleOCRVisionRotaryEmbedding<B> {
    /// Construct a vision rotary embedding with the given half-head dimension.
    pub fn new(dim: usize, device: &B::Device) -> Self {
        let theta = 10_000.0f32;
        // equals to 1 / theta^(index / dim)
        let inv_freq = Tensor::<B, 1, Int>::arange_step(0..dim as i64, 2, device)
            .float()
            .cast(DType::F32)
            .div_scalar(dim as f32)
            .mul_scalar(-(theta.ln()))
            .exp();

        Self { inv_freq, dim }
    }

    /// Compute rotary frequency matrix for the given sequence length.
    ///
    /// Returns a `[seq_length, dim/2]` tensor of position frequencies.
    pub fn forward(&self, seq_length: usize) -> Tensor<B, 2> {
        let device = &self.inv_freq.device();
        let seq = Tensor::<B, 1, Int>::arange(0..seq_length as i64, device)
            .float()
            .cast(DType::F32)
            .unsqueeze_dim::<2>(1);

        seq.matmul(self.inv_freq.clone().unsqueeze_dim::<2>(0))
    }
}

#[derive(Module, Debug)]
pub struct PaddleOcrVisionAttention<B: Backend> {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub attn_dropout: Dropout,
    pub scaling: f64,
}

impl<B: Backend> PaddleOcrVisionAttention<B> {
    /// Vision self-attention forward pass with per-segment processing.
    ///
    /// Each segment (one image's patch tokens) is processed independently
    /// with full bidirectional attention + rotary position embeddings.
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 2>,
        segment_lengths: &[usize],
        position_embeddings: &VisionPositionEmbeddings<B>,
    ) -> Tensor<B, 2> {
        let [seq_length, _] = hidden_states.dims();

        let query_states = self.q_proj.forward(hidden_states.clone()).reshape([
            seq_length,
            self.num_heads,
            self.head_dim,
        ]);
        let key_states = self.k_proj.forward(hidden_states.clone()).reshape([
            seq_length,
            self.num_heads,
            self.head_dim,
        ]);
        let value_states =
            self.v_proj
                .forward(hidden_states)
                .reshape([seq_length, self.num_heads, self.head_dim]);

        let (query_states, key_states) = apply_rotary_pos_emb_vision(
            query_states,
            key_states,
            &position_embeddings.cos,
            &position_embeddings.sin,
        );

        let mut start = 0usize;
        let mut attn_outputs = Vec::with_capacity(segment_lengths.len());

        for &segment_len in segment_lengths {
            let q = query_states
                .clone()
                .slice(s![start..(start + segment_len), .., ..])
                .swap_dims(0, 1)
                .unsqueeze_dim::<4>(0);
            let k = key_states
                .clone()
                .slice(s![start..(start + segment_len), .., ..])
                .swap_dims(0, 1)
                .unsqueeze_dim::<4>(0);
            let v = value_states
                .clone()
                .slice(s![start..(start + segment_len), .., ..])
                .swap_dims(0, 1)
                .unsqueeze_dim::<4>(0);

            let device = q.device();
            let attn_output = if self.attn_dropout.prob == 0.0 || !B::ad_enabled(&device) {
                attention(q, k, v, None, None, Default::default())
            } else {
                let scores = q.clone().matmul(k.swap_dims(2, 3)).mul_scalar(self.scaling);
                let probs = self.attn_dropout.forward(activation::softmax(scores, 3));
                probs.matmul(v)
            };

            let attn_output = attn_output
                .squeeze_dim::<3>(0)
                .swap_dims(0, 1)
                .reshape([segment_len, self.embed_dim]);
            attn_outputs.push(attn_output);
            start += segment_len;
        }

        let attn_output = Tensor::cat(attn_outputs, 0);
        self.out_proj.forward(attn_output)
    }
}

fn apply_rotary_pos_emb_vision<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    cos: &Tensor<B, 2>,
    sin: &Tensor<B, 2>,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let dtype = q.dtype();
    let cos = cos.clone().cast(dtype).unsqueeze_dim::<3>(1);
    let sin = sin.clone().cast(dtype).unsqueeze_dim::<3>(1);

    let q_embed = q.clone() * cos.clone() + rotate_half(q) * sin.clone();
    let k_embed = k.clone() * cos + rotate_half(k) * sin;

    (q_embed, k_embed)
}

fn rotate_half<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [_seq_length, _num_heads, head_dim] = x.dims();
    let half_dim = head_dim / 2;

    let x1 = x.clone().slice(s![.., .., 0..half_dim]);
    let x2 = x.slice(s![.., .., half_dim..head_dim]);

    Tensor::cat(vec![x2.neg(), x1], 2)
}

#[derive(Config, Debug)]
pub struct PaddleOcrVisionAttentionConfig {
    #[config(default = 1152)]
    pub embed_dim: usize,
    #[config(default = 16)]
    pub num_attention_heads: usize,
    #[config(default = 0.0)]
    attn_dropout: f64,
}

impl PaddleOcrVisionAttentionConfig {
    /// Build a `PaddleOcrVisionAttention` module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PaddleOcrVisionAttention<B> {
        let head_dim = self.embed_dim / self.num_attention_heads;

        PaddleOcrVisionAttention {
            embed_dim: self.embed_dim,
            num_heads: self.num_attention_heads,
            head_dim,
            k_proj: LinearConfig::new(self.embed_dim, self.embed_dim).init(device),
            v_proj: LinearConfig::new(self.embed_dim, self.embed_dim).init(device),
            q_proj: LinearConfig::new(self.embed_dim, self.embed_dim).init(device),
            out_proj: LinearConfig::new(self.embed_dim, self.embed_dim).init(device),
            attn_dropout: DropoutConfig::new(self.attn_dropout).init(),
            scaling: (head_dim as f64).powf(-0.5),
        }
    }
}

#[derive(Module, Debug)]
pub struct PaddleOCRVisionEncoderLayer<B: Backend> {
    pub layer_norm1: LayerNorm<B>,
    pub self_attn: PaddleOcrVisionAttention<B>,
    pub layer_norm2: LayerNorm<B>,
    pub mlp: PaddleOcrVisionMLP<B>,
}

impl<B: Backend> PaddleOCRVisionEncoderLayer<B> {
    /// Encoder layer forward: LN → SelfAttn → residual → LN → MLP → residual.
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 2>,
        segment_lengths: &[usize],
        position_embeddings: &VisionPositionEmbeddings<B>,
    ) -> Tensor<B, 2> {
        let residual = hidden_states.clone();

        let hidden_states = self.layer_norm1.forward(hidden_states);
        let hidden_states =
            self.self_attn
                .forward(hidden_states, segment_lengths, position_embeddings);
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm2.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states);

        residual + hidden_states
    }
}

#[derive(Config, Debug)]
pub struct PaddleOCRVisionEncoderLayerConfig {
    #[config(default = 1152)]
    pub embed_dim: usize,
    #[config(default = 4304)]
    pub intermediate_size: usize,
    #[config(default = 1e-06)]
    pub layer_norm_eps: f64,
    #[config(default = 16)]
    pub num_attention_heads: usize,
    #[config(default = 0.0)]
    pub attn_dropout: f64,
}

impl PaddleOCRVisionEncoderLayerConfig {
    /// Build a `PaddleOCRVisionEncoderLayer`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PaddleOCRVisionEncoderLayer<B> {
        PaddleOCRVisionEncoderLayer {
            layer_norm1: LayerNormConfig::new(self.embed_dim)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
            self_attn: PaddleOcrVisionAttentionConfig::new()
                .with_embed_dim(self.embed_dim)
                .with_num_attention_heads(self.num_attention_heads)
                .with_attn_dropout(self.attn_dropout)
                .init(device),
            layer_norm2: LayerNormConfig::new(self.embed_dim)
                .with_epsilon(self.layer_norm_eps)
                .init(device),
            mlp: PaddleOcrVisionMLPConfig::new()
                .with_hidden_size(self.embed_dim)
                .with_intermediate_size(self.intermediate_size)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct PaddleOCRVisionEncoder<B: Backend> {
    pub layers: Vec<PaddleOCRVisionEncoderLayer<B>>,
    pub rotary_pos_emb: PaddleOCRVisionRotaryEmbedding<B>,
}

impl<B: Backend> PaddleOCRVisionEncoder<B> {
    /// Run all encoder layers over the input embeddings with 2D rotary position embeddings.
    pub fn forward(
        &self,
        inputs_embeds: Tensor<B, 2>,
        image_grid_hw: &[(usize, usize)],
    ) -> Tensor<B, 2> {
        let device = inputs_embeds.device();
        let segment_lengths = image_grid_hw.iter().map(|(h, w)| h * w).collect::<Vec<_>>();
        let position_embeddings = self.build_position_embeddings(image_grid_hw, &device);

        let mut hidden_states = inputs_embeds;
        for layer in self.layers.iter() {
            hidden_states = layer.forward(hidden_states, &segment_lengths, &position_embeddings);
        }

        hidden_states
    }

    fn build_position_embeddings(
        &self,
        image_grid_hw: &[(usize, usize)],
        device: &B::Device,
    ) -> VisionPositionEmbeddings<B> {
        let mut max_grid_size = 0usize;
        for &(h, w) in image_grid_hw {
            max_grid_size = max_grid_size.max(h.max(w));
        }

        // Build height/width index tensors on GPU via arange + broadcast,
        // then concatenate across image segments.
        let mut height_id_parts = Vec::with_capacity(image_grid_hw.len());
        let mut width_id_parts = Vec::with_capacity(image_grid_hw.len());

        for &(h, w) in image_grid_hw {
            // h_ids: [0,0,..,0, 1,1,..,1, ..., h-1,h-1,..,h-1] (each repeated w times)
            let h_ids = Tensor::<B, 1, Int>::arange(0..h as i64, device)
                .unsqueeze_dim::<2>(1)
                .repeat_dim(1, w)
                .reshape([h * w]);
            // w_ids: [0,1,..,w-1, 0,1,..,w-1, ...] (repeated h times)
            let w_ids = Tensor::<B, 1, Int>::arange(0..w as i64, device)
                .unsqueeze_dim::<2>(0)
                .repeat_dim(0, h)
                .reshape([h * w]);

            height_id_parts.push(h_ids);
            width_id_parts.push(w_ids);
        }

        let height_ids = Tensor::cat(height_id_parts, 0);
        let width_ids = Tensor::cat(width_id_parts, 0);

        let rotary_embeddings_max_grid = self.rotary_pos_emb.forward(max_grid_size);
        let height_rotary = rotary_embeddings_max_grid.clone().select(0, height_ids);
        let width_rotary = rotary_embeddings_max_grid.select(0, width_ids);

        let rotary_embeddings = Tensor::cat(vec![height_rotary, width_rotary], 1).repeat(&[1, 2]);

        VisionPositionEmbeddings {
            cos: rotary_embeddings.clone().cos(),
            sin: rotary_embeddings.sin(),
        }
    }
}

#[derive(Config, Debug)]
pub struct PaddleOCRVisionEncoderConfig {
    #[config(default = 27)]
    pub num_hidden_layers: usize,
    #[config(default = 1152)]
    pub embed_dim: usize,
    #[config(default = 4304)]
    pub intermediate_size: usize,
    #[config(default = 1e-06)]
    pub layer_norm_eps: f64,
    #[config(default = 16)]
    pub num_attention_heads: usize,
    #[config(default = 0.0)]
    pub attn_dropout: f64,
}

impl PaddleOCRVisionEncoderConfig {
    /// Build a `PaddleOCRVisionEncoder` with all layers and rotary embeddings.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PaddleOCRVisionEncoder<B> {
        let head_dim = self.embed_dim / self.num_attention_heads;

        PaddleOCRVisionEncoder {
            layers: (0..self.num_hidden_layers)
                .map(|_| {
                    PaddleOCRVisionEncoderLayerConfig::new()
                        .with_embed_dim(self.embed_dim)
                        .with_intermediate_size(self.intermediate_size)
                        .with_layer_norm_eps(self.layer_norm_eps)
                        .with_num_attention_heads(self.num_attention_heads)
                        .with_attn_dropout(self.attn_dropout)
                        .init(device)
                })
                .collect(),
            rotary_pos_emb: PaddleOCRVisionRotaryEmbedding::new(head_dim / 2, device),
        }
    }
}
