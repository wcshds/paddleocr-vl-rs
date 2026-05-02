#[cfg(feature = "mlx")]
mod mlx_half_anchors {
    use burn::prelude::Backend;
    use half::{bf16, f16};
    use pp_doclayout::decoder::generate_anchors;

    fn assert_anchors_are_finite<B: Backend>() {
        let device = B::Device::default();
        let (anchors, valid_mask) =
            generate_anchors::<B>(&[(100, 100), (50, 50), (25, 25)], 0.05, &device);

        let anchors = anchors
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("anchor logits");
        let valid_mask = valid_mask
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("valid mask");

        assert!(
            anchors.iter().all(|value| value.is_finite()),
            "half precision anchor logits must stay finite"
        );
        assert!(
            valid_mask
                .iter()
                .all(|value| *value == 0.0 || *value == 1.0),
            "anchor valid mask should be binary"
        );
    }

    #[test]
    fn mlx_half_anchor_logits_stay_finite() {
        assert_anchors_are_finite::<burn_mlx::Mlx<f16>>();
        assert_anchors_are_finite::<burn_mlx::Mlx<bf16>>();
    }
}
