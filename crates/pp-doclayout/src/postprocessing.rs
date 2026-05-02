// ========================================================================
// PPDocLayoutV2 Post-Processing
// ========================================================================
//
// Fully aligned with HuggingFace PPDocLayoutV2ImageProcessor
// .post_process_object_detection.
//
// Model outputs:
//   - logits:       `[B, Q, C]`   per-query per-class classification logits
//   - pred_boxes:   `[B, Q, 4]`   normalized bbox (cx, cy, w, h)
//   - order_logits: `[B, Q, Q]`   reading-order logits
//
// Post-processing steps (matching the Python reference):
//
// 1. **Reading-order sequence** (`_get_order_seqs`):
//    order_scores = sigmoid(order_logits)                          [B, Q, Q]
//    votes = triu(scores, diag=1).sum(dim=1)
//          + (1 - scores^T).tril(diag=-1).sum(dim=1)              [B, Q]
//    order_pointers = argsort(votes)
//    order_seq: scatter(pointers → ranks)                          [B, Q]
//
// 2. **Coordinate conversion**:
//    (cx, cy, w, h) → (x1, y1, x2, y2) → × target_size
//
// 3. **Global Top-K selection**:
//    scores = sigmoid(logits).flatten(1)    [B, Q×C]
//    select top-K (K = Q), yielding (query_idx, class_id)
//    gather boxes and order_seq by query_idx
//
// 4. **Threshold filtering + order-based sorting**:
//    keep detections where score ≥ threshold
//    sort by order_seq in ascending order
// ========================================================================

use burn::{
    prelude::Backend,
    tensor::{Tensor, activation},
};
use std::collections::HashMap;

fn tensor_to_vec_f32<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Vec<f32> {
    tensor
        .clone()
        .to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap()
}

/// A single detection result for one image.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Confidence scores of each detected element (sorted by reading order).
    pub scores: Vec<f32>,
    /// Class indices of each detected element (sorted by reading order).
    pub labels: Vec<usize>,
    /// Bounding boxes [x1, y1, x2, y2] in pixel coordinates (sorted by reading order).
    pub boxes: Vec<[f32; 4]>,
}

/// PP-DocLayoutV3 detection result.
#[derive(Debug, Clone)]
pub struct DetectionResultV3 {
    /// Confidence scores of each detected element (sorted by reading order).
    pub scores: Vec<f32>,
    /// Class indices of each detected element (sorted by reading order).
    pub labels: Vec<usize>,
    /// Bounding boxes [x1, y1, x2, y2] in pixel coordinates.
    pub boxes: Vec<[f32; 4]>,
    /// Polygon points in pixel coordinates. When a mask has no positive region,
    /// the rectangular bounding box is returned as a conservative fallback.
    pub polygon_points: Vec<Vec<[f32; 2]>>,
}

/// Convert vote scores into per-query reading-order ranks.
///
/// Both Paddle decoders sort by ascending votes and then scatter ranks back to
/// the original query positions: `order_seq[pointers[rank]] = rank`.
fn ranks_from_votes(votes: &[f64]) -> Vec<usize> {
    let mut pointers: Vec<usize> = (0..votes.len()).collect();
    pointers.sort_by(|&a, &b| {
        votes[a]
            .partial_cmp(&votes[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut order_seq = vec![0usize; votes.len()];
    for (rank, &ptr) in pointers.iter().enumerate() {
        order_seq[ptr] = rank;
    }
    order_seq
}

/// Compute the V2 reading-order sequence.
///
/// This follows PaddleOCR / PP-DocLayoutV2's win-accumulation decode:
/// `triu(scores, diag=1).sum(dim=1) + (1 - scores.T).tril(diag=-1).sum(dim=1)`.
///
/// For each query produces a rank — the lower the value, the earlier it
/// appears in reading order.
///
/// - `order_logits`: `[B, Q, Q]`
///
/// Returns: `Vec<Vec<usize>>`, one length-Q order_seq per batch element.
fn get_order_seqs_v2<B: Backend>(order_logits: &Tensor<B, 3>) -> Vec<Vec<usize>> {
    let [batch_size, seq_len, _] = order_logits.dims();
    let order_scores = tensor_to_vec_f32(&activation::sigmoid(order_logits.clone()));

    let mut all_seqs = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let off = b * seq_len * seq_len;

        // Python formula (3D tensor, sum over dim=1 = column-wise aggregation):
        //   triu(scores, diag=1).sum(dim=1)[q] = Σ_{k < q} scores[k][q]
        //   (1 - scores^T).tril(diag=-1).sum(dim=1)[q] = Σ_{k > q} (1 - scores[q][k])
        //
        // votes[q] = "how many other items should come before q" → smaller = earlier
        let mut votes = vec![0.0f64; seq_len];
        for q in 0..seq_len {
            for k in 0..q {
                // Upper-triangular column aggregation: scores[k][q] for k < q
                votes[q] += order_scores[off + k * seq_len + q] as f64;
            }
            for k in (q + 1)..seq_len {
                // Lower-triangular transposed complementary column aggregation:
                // (1 - scores[q][k]) for k > q
                votes[q] += 1.0 - order_scores[off + q * seq_len + k] as f64;
            }
        }

        all_seqs.push(ranks_from_votes(&votes));
    }

    all_seqs
}

/// Post-process model outputs (fully aligned with Python
/// `post_process_object_detection`).
///
/// - `logits`: `[B, Q, C]` classification logits
/// - `pred_boxes`: `[B, Q, 4]` normalized bbox (cx, cy, w, h)
/// - `order_logits`: `[B, Q, Q]` reading-order logits
/// - `target_sizes`: per-image (height, width) in pixels
/// - `threshold`: confidence threshold (Python default 0.5)
pub fn post_process_object_detection<B: Backend>(
    logits: &Tensor<B, 3>,
    pred_boxes: &Tensor<B, 3>,
    order_logits: &Tensor<B, 3>,
    target_sizes: &[(usize, usize)],
    threshold: f32,
) -> Vec<DetectionResult> {
    let [batch_size, num_queries, num_classes] = logits.dims();

    // 1. Compute reading-order sequences
    let order_seqs = get_order_seqs_v2::<B>(order_logits);

    // 2. Coordinate conversion (cx, cy, w, h) → (x1, y1, x2, y2) and scale to original image
    let boxes_data = tensor_to_vec_f32(pred_boxes);

    // 3. Global sigmoid scores [B, Q, C]
    let scores_all = tensor_to_vec_f32(&activation::sigmoid(logits.clone()));

    let mut results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let (target_h, target_w) = target_sizes[b];
        let tw = target_w as f32;
        let th = target_h as f32;

        // Global top-K: flatten Q×C scores and take the top num_queries entries
        let base_s = b * num_queries * num_classes;
        let mut flat_scores: Vec<(f32, usize)> = (0..num_queries * num_classes)
            .map(|idx| (scores_all[base_s + idx], idx))
            .collect();
        flat_scores
            .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        flat_scores.truncate(num_queries);

        // Decompose into (score, query_idx, class_id), gather boxes and order_seq
        let base_b = b * num_queries * 4;
        let order_seq = &order_seqs[b];

        struct Entry {
            score: f32,
            label: usize,
            bbox: [f32; 4],
            order: usize,
        }

        let mut entries: Vec<Entry> = Vec::new();

        for &(score, flat_idx) in &flat_scores {
            if score < threshold {
                break;
            }
            let query_idx = flat_idx / num_classes;
            let class_id = flat_idx % num_classes;

            let cx = boxes_data[base_b + query_idx * 4];
            let cy = boxes_data[base_b + query_idx * 4 + 1];
            let w = boxes_data[base_b + query_idx * 4 + 2];
            let h = boxes_data[base_b + query_idx * 4 + 3];

            let x1 = (cx - 0.5 * w) * tw;
            let y1 = (cy - 0.5 * h) * th;
            let x2 = (cx + 0.5 * w) * tw;
            let y2 = (cy + 0.5 * h) * th;

            entries.push(Entry {
                score,
                label: class_id,
                bbox: [x1, y1, x2, y2],
                order: order_seq[query_idx],
            });
        }

        // Sort by order_seq ascending (reading order)
        entries.sort_by_key(|e| e.order);

        results.push(DetectionResult {
            scores: entries.iter().map(|e| e.score).collect(),
            labels: entries.iter().map(|e| e.label).collect(),
            boxes: entries.iter().map(|e| e.bbox).collect(),
        });
    }

    results
}

fn rectangle_polygon(bbox: [f32; 4]) -> Vec<[f32; 2]> {
    vec![
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]],
    ]
}

fn mask_polygon(
    masks_data: &[f32],
    mask_base: usize,
    mask_h: usize,
    mask_w: usize,
    target_h: usize,
    target_w: usize,
    threshold_logit: f32,
    fallback_bbox: [f32; 4],
) -> Vec<[f32; 2]> {
    // Mirror HuggingFace's V3 postprocessor:
    // bbox -> crop mask in stride-4 mask space -> nearest resize to bbox pixels
    // -> external contour -> approxPolyDP-like simplification.
    let x_min = fallback_bbox[0] as i32;
    let y_min = fallback_bbox[1] as i32;
    let x_max = fallback_bbox[2] as i32;
    let y_max = fallback_bbox[3] as i32;
    let box_w = x_max - x_min;
    let box_h = y_max - y_min;
    if box_w <= 0 || box_h <= 0 {
        return rectangle_polygon(fallback_bbox);
    }

    let scale_x = mask_w as f32 / target_w as f32;
    let scale_y = mask_h as f32 / target_h as f32;
    let x_start = ((x_min as f32 * scale_x).round() as i32).clamp(0, mask_w as i32) as usize;
    let x_end = ((x_max as f32 * scale_x).round() as i32).clamp(0, mask_w as i32) as usize;
    let y_start = ((y_min as f32 * scale_y).round() as i32).clamp(0, mask_h as i32) as usize;
    let y_end = ((y_max as f32 * scale_y).round() as i32).clamp(0, mask_h as i32) as usize;
    if x_start >= x_end || y_start >= y_end {
        return rectangle_polygon(fallback_bbox);
    }

    let box_w_usize = box_w as usize;
    let box_h_usize = box_h as usize;
    let crop_w = x_end - x_start;
    let crop_h = y_end - y_start;
    let mut resized = vec![0u8; box_w_usize * box_h_usize];

    for y in 0..box_h_usize {
        let src_y = y_start + y * crop_h / box_h_usize;
        for x in 0..box_w_usize {
            let src_x = x_start + x * crop_w / box_w_usize;
            let value = masks_data[mask_base + src_y * mask_w + src_x];
            resized[y * box_w_usize + x] = u8::from(value > threshold_logit);
        }
    }

    let Some(mut contour) = largest_external_contour(&resized, box_w_usize, box_h_usize) else {
        return rectangle_polygon(fallback_bbox);
    };

    // OpenCV contours for this model are clockwise in image coordinates. The
    // boundary-edge tracer below returns the opposite orientation for outer
    // loops, so normalize before applying the same custom-vertex heuristic.
    if signed_area_i32(&contour) > 0.0 {
        contour.reverse();
    }

    let contour = contour
        .into_iter()
        .map(|(x, y)| (x.clamp(0, box_w - 1) as f32, y.clamp(0, box_h - 1) as f32))
        .collect::<Vec<_>>();
    let epsilon = 0.004 * arc_length(&contour);
    let approx = approx_poly_dp_closed(&contour, epsilon);
    let polygon = extract_custom_vertices(&approx);

    if polygon.len() < 4 {
        return rectangle_polygon(fallback_bbox);
    }

    let polygon = polygon
        .into_iter()
        .map(|(x, y)| [x + x_min as f32, y + y_min as f32])
        .collect();
    rotate_polygon_to_leftmost_top(polygon)
}

fn rotate_polygon_to_leftmost_top(mut polygon: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    if polygon.len() <= 1 {
        return polygon;
    }

    // OpenCV contours for these masks start at the left-most boundary point;
    // ties pick the top-most point on that same x coordinate.
    let start_idx = polygon
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a[0].partial_cmp(&b[0])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    polygon.rotate_left(start_idx);
    polygon
}

fn largest_external_contour(mask: &[u8], width: usize, height: usize) -> Option<Vec<(i32, i32)>> {
    let mut edges: HashMap<(i32, i32), Vec<(i32, i32)>> = HashMap::new();
    for y in 0..height {
        for x in 0..width {
            if mask[y * width + x] == 0 {
                continue;
            }

            let x0 = x as i32;
            let y0 = y as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            if y == 0 || mask[(y - 1) * width + x] == 0 {
                add_edge(&mut edges, (x0, y0), (x1, y0));
            }
            if x + 1 == width || mask[y * width + x + 1] == 0 {
                add_edge(&mut edges, (x1, y0), (x1, y1));
            }
            if y + 1 == height || mask[(y + 1) * width + x] == 0 {
                add_edge(&mut edges, (x1, y1), (x0, y1));
            }
            if x == 0 || mask[y * width + x - 1] == 0 {
                add_edge(&mut edges, (x0, y1), (x0, y0));
            }
        }
    }

    let mut best = None;
    let mut best_area = 0.0f64;

    loop {
        let Some(start) = edges
            .iter()
            .find_map(|(point, next)| (!next.is_empty()).then_some(*point))
        else {
            break;
        };
        let mut contour = vec![start];
        let mut current = pop_edge(&mut edges, start)?;
        let mut closed = false;

        for _ in 0..(width * height * 4 + 4) {
            if current == start {
                closed = true;
                break;
            }
            contour.push(current);
            let Some(next) = pop_edge(&mut edges, current) else {
                break;
            };
            current = next;
        }

        if closed && contour.len() >= 3 {
            let contour = remove_collinear_i32(contour);
            let area = signed_area_i32(&contour).abs();
            if area > best_area {
                best_area = area;
                best = Some(contour);
            }
        }
    }

    best
}

fn add_edge(edges: &mut HashMap<(i32, i32), Vec<(i32, i32)>>, from: (i32, i32), to: (i32, i32)) {
    edges.entry(from).or_default().push(to);
}

fn pop_edge(
    edges: &mut HashMap<(i32, i32), Vec<(i32, i32)>>,
    from: (i32, i32),
) -> Option<(i32, i32)> {
    edges.get_mut(&from)?.pop()
}

fn remove_collinear_i32(points: Vec<(i32, i32)>) -> Vec<(i32, i32)> {
    if points.len() <= 2 {
        return points;
    }

    let mut out = Vec::with_capacity(points.len());
    for idx in 0..points.len() {
        let prev = points[(idx + points.len() - 1) % points.len()];
        let curr = points[idx];
        let next = points[(idx + 1) % points.len()];
        let dx1 = curr.0 - prev.0;
        let dy1 = curr.1 - prev.1;
        let dx2 = next.0 - curr.0;
        let dy2 = next.1 - curr.1;
        if dx1 * dy2 - dy1 * dx2 != 0 {
            out.push(curr);
        }
    }
    out
}

fn signed_area_i32(points: &[(i32, i32)]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0f64;
    for idx in 0..points.len() {
        let (x0, y0) = points[idx];
        let (x1, y1) = points[(idx + 1) % points.len()];
        area += x0 as f64 * y1 as f64 - y0 as f64 * x1 as f64;
    }
    area * 0.5
}

fn arc_length(points: &[(f32, f32)]) -> f32 {
    if points.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for idx in 0..points.len() {
        let a = points[idx];
        let b = points[(idx + 1) % points.len()];
        total += ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    }
    total
}

fn approx_poly_dp_closed(points: &[(f32, f32)], epsilon: f32) -> Vec<(f32, f32)> {
    if points.len() <= 2 {
        return points.to_vec();
    }

    let mut closed = points.to_vec();
    closed.push(points[0]);
    let mut simplified = rdp_open(&closed, epsilon);
    if simplified.len() > 1 && simplified.first() == simplified.last() {
        simplified.pop();
    }
    remove_collinear_f32(simplified)
}

fn rdp_open(points: &[(f32, f32)], epsilon: f32) -> Vec<(f32, f32)> {
    if points.len() <= 2 {
        return points.to_vec();
    }

    let first = points[0];
    let last = *points.last().unwrap();
    let mut max_dist = 0.0;
    let mut split_idx = 0usize;

    for (idx, &point) in points.iter().enumerate().take(points.len() - 1).skip(1) {
        let dist = point_segment_distance(point, first, last);
        if dist > max_dist {
            max_dist = dist;
            split_idx = idx;
        }
    }

    if max_dist > epsilon {
        let mut left = rdp_open(&points[..=split_idx], epsilon);
        let right = rdp_open(&points[split_idx..], epsilon);
        left.pop();
        left.extend(right);
        left
    } else {
        vec![first, last]
    }
}

fn point_segment_distance(point: (f32, f32), start: (f32, f32), end: (f32, f32)) -> f32 {
    let vx = end.0 - start.0;
    let vy = end.1 - start.1;
    let wx = point.0 - start.0;
    let wy = point.1 - start.1;
    let len_sq = vx * vx + vy * vy;
    if len_sq <= f32::EPSILON {
        return ((point.0 - start.0).powi(2) + (point.1 - start.1).powi(2)).sqrt();
    }
    let t = ((wx * vx + wy * vy) / len_sq).clamp(0.0, 1.0);
    let proj = (start.0 + t * vx, start.1 + t * vy);
    ((point.0 - proj.0).powi(2) + (point.1 - proj.1).powi(2)).sqrt()
}

fn remove_collinear_f32(points: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
    if points.len() <= 2 {
        return points;
    }

    let mut out = Vec::with_capacity(points.len());
    for idx in 0..points.len() {
        let prev = points[(idx + points.len() - 1) % points.len()];
        let curr = points[idx];
        let next = points[(idx + 1) % points.len()];
        let cross = (curr.0 - prev.0) * (next.1 - curr.1) - (curr.1 - prev.1) * (next.0 - curr.0);
        if cross.abs() > 1e-4 {
            out.push(curr);
        }
    }
    out
}

fn extract_custom_vertices(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if points.len() < 3 {
        return points.to_vec();
    }

    let mut output = Vec::with_capacity(points.len());
    for idx in 0..points.len() {
        let previous = points[(idx + points.len() - 1) % points.len()];
        let current = points[idx];
        let next = points[(idx + 1) % points.len()];

        let v1 = (previous.0 - current.0, previous.1 - current.1);
        let v2 = (next.0 - current.0, next.1 - current.1);
        let cross = v1.1 * v2.0 - v1.0 * v2.1;
        if cross >= 0.0 {
            continue;
        }

        let norm1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
        let norm2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();
        if norm1 <= f32::EPSILON || norm2 <= f32::EPSILON {
            output.push(current);
            continue;
        }

        let cos = ((v1.0 * v2.0 + v1.1 * v2.1) / (norm1 * norm2)).clamp(-1.0, 1.0);
        let angle = cos.acos().to_degrees();
        if (angle - 45.0).abs() < 1.0 {
            let mut dir = (v1.0 / norm1 + v2.0 / norm2, v1.1 / norm1 + v2.1 / norm2);
            let dir_norm = (dir.0 * dir.0 + dir.1 * dir.1).sqrt();
            if dir_norm > f32::EPSILON {
                dir.0 /= dir_norm;
                dir.1 /= dir_norm;
                let step_size = (norm1 + norm2) * 0.5;
                output.push((current.0 + dir.0 * step_size, current.1 + dir.1 * step_size));
            } else {
                output.push(current);
            }
        } else {
            output.push(current);
        }
    }

    output
}

/// Post-process PP-DocLayoutV3 outputs.
///
/// V3 uses the same score/top-k/order logic as V2 and can additionally expose
/// query masks. The mask polygon path mirrors HuggingFace's crop/resize/
/// contour/simplify flow and falls back to rectangular boxes only when the
/// mask has no usable positive region.
pub fn post_process_object_detection_v3<B: Backend>(
    logits: &Tensor<B, 3>,
    pred_boxes: &Tensor<B, 3>,
    order_logits: &Tensor<B, 3>,
    out_masks: Option<&Tensor<B, 4>>,
    target_sizes: &[(usize, usize)],
    threshold: f32,
) -> Vec<DetectionResultV3> {
    let [batch_size, num_queries, num_classes] = logits.dims();
    // HuggingFace's generated PP-DocLayoutV3 image processor uses the same
    // triangular win-accumulation decoder as V2.
    let order_seqs = get_order_seqs_v2::<B>(order_logits);
    let boxes_data = tensor_to_vec_f32(pred_boxes);
    let scores_all = tensor_to_vec_f32(&activation::sigmoid(logits.clone()));

    let masks = out_masks.map(|m| {
        let [_b, _q, h, w] = m.dims();
        (tensor_to_vec_f32(m), h, w)
    });
    let threshold_logit = (threshold / (1.0 - threshold)).ln();

    let mut results = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let (target_h, target_w) = target_sizes[b];
        let tw = target_w as f32;
        let th = target_h as f32;

        let base_s = b * num_queries * num_classes;
        let mut flat_scores: Vec<(f32, usize)> = (0..num_queries * num_classes)
            .map(|idx| (scores_all[base_s + idx], idx))
            .collect();
        flat_scores
            .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        flat_scores.truncate(num_queries);

        let base_b = b * num_queries * 4;
        let order_seq = &order_seqs[b];

        struct Entry {
            score: f32,
            label: usize,
            bbox: [f32; 4],
            polygon: Vec<[f32; 2]>,
            order: usize,
        }

        let mut entries = Vec::new();
        for &(score, flat_idx) in &flat_scores {
            if score < threshold {
                break;
            }

            let query_idx = flat_idx / num_classes;
            let class_id = flat_idx % num_classes;

            let cx = boxes_data[base_b + query_idx * 4];
            let cy = boxes_data[base_b + query_idx * 4 + 1];
            let w = boxes_data[base_b + query_idx * 4 + 2];
            let h = boxes_data[base_b + query_idx * 4 + 3];

            let bbox = [
                (cx - 0.5 * w) * tw,
                (cy - 0.5 * h) * th,
                (cx + 0.5 * w) * tw,
                (cy + 0.5 * h) * th,
            ];

            let polygon = if let Some((masks_data, mask_h, mask_w)) = masks.as_ref() {
                let mask_base = (b * num_queries + query_idx) * mask_h * mask_w;
                mask_polygon(
                    masks_data,
                    mask_base,
                    *mask_h,
                    *mask_w,
                    target_h,
                    target_w,
                    threshold_logit,
                    bbox,
                )
            } else {
                rectangle_polygon(bbox)
            };

            entries.push(Entry {
                score,
                label: class_id,
                bbox,
                polygon,
                order: order_seq[query_idx],
            });
        }

        entries.sort_by_key(|e| e.order);

        results.push(DetectionResultV3 {
            scores: entries.iter().map(|e| e.score).collect(),
            labels: entries.iter().map(|e| e.label).collect(),
            boxes: entries.iter().map(|e| e.bbox).collect(),
            polygon_points: entries.into_iter().map(|e| e.polygon).collect(),
        });
    }

    results
}
