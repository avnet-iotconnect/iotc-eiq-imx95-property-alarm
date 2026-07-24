"""YOLOv8 pre/post-processing helpers.

detect.py is the 'what' (grab frame, infer, report); this module is the 'how'.
A YOLOv8 model takes a square RGB image and returns one row per candidate box.
Getting a frame in and detections out means four bits of arithmetic, kept here:

    letterbox        - fit the frame into the square model input without distortion
    quantize_input   - turn pixels into the int8/uint8 tensor the model expects
    decode_detections- read the raw output rows into boxes + classes + scores
    (nms, box mapping)- drop duplicate/rescaled boxes back onto the original frame
"""

from __future__ import annotations

import cv2
import numpy as np

# COCO 80 classes, in the order YOLOv8 was trained on - the output column index is the class id.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize `image` into a `size` x `size` square, keeping aspect ratio and padding with gray.

    Returns the padded image plus the scale and (pad_x, pad_y) offsets, which
    `map_boxes_to_frame` later uses to place detections back on the full frame.
    """
    source_height, source_width = image.shape[:2]
    scale = min(size / source_width, size / source_height)
    resized_width, resized_height = round(source_width * scale), round(source_height * scale)
    resized = cv2.resize(image, (resized_width, resized_height))

    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)  # 114 = YOLO's standard gray pad
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, scale, pad_x, pad_y


def quantize_input(image: np.ndarray, input_detail: dict) -> np.ndarray:
    """Turn a uint8 RGB image into the batched tensor the model wants.

    Pixels are scaled to 0..1; for a quantized model they are then mapped through
    the input tensor's (scale, zero_point) into its int8/uint8 range. Reading these
    from the model - instead of hardcoding - keeps us honest whatever the export did.
    """
    normalized = image.astype(np.float32) / 255.0
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        tensor = normalized
    else:
        scale, zero_point = input_detail["quantization"]
        quantized = np.round(normalized / scale + zero_point)
        info = np.iinfo(dtype)
        tensor = np.clip(quantized, info.min, info.max).astype(dtype)
    return tensor[np.newaxis, ...]  # add the batch dimension -> (1, H, W, 3)


def dequantize_output(raw: np.ndarray, output_detail: dict) -> np.ndarray:
    """Undo quantization so the numbers are real box coords and 0..1 scores again."""
    if output_detail["dtype"] == np.float32:
        return raw.astype(np.float32)
    scale, zero_point = output_detail["quantization"]
    return (raw.astype(np.float32) - zero_point) * scale


REG_MAX = 16  # YOLOv8 predicts each box edge as a distribution over 16 bins (DFL)
STRIDES = (8, 16, 32)  # the three detection scales; feature grids are input_size / stride


def decode_detections(
    output: np.ndarray, input_size: int, confidence_threshold: float, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the head's RAW output into (boxes_xyxy, scores, class_ids) in model-input pixels.

    The model emits (1, 4*REG_MAX + num_classes, num_boxes): the first 64 channels are the box-edge
    distributions, the rest are class logits. This is the decode the NPU graph left out, done in float:
      1. class score  = sigmoid(class logit); keep only boxes whose best class clears the threshold.
      2. box distance  = DFL: softmax each edge's 16 bins, take the expected bin -> left/top/right/bottom.
      3. box corners   = anchor point -/+ those distances, times the anchor's stride -> pixel xyxy.
    Then non-max suppression drops overlapping duplicates.
    """
    predictions = np.squeeze(output).T  # (num_boxes, 4*REG_MAX + num_classes)
    num_classes = predictions.shape[1] - 4 * REG_MAX
    box_distribution = predictions[:, : 4 * REG_MAX]
    class_logits = predictions[:, 4 * REG_MAX :]

    class_ids = np.argmax(class_logits, axis=1)
    scores = _sigmoid(class_logits[np.arange(class_logits.shape[0]), class_ids])
    keep = scores >= confidence_threshold

    anchor_points, anchor_strides = _make_anchors(input_size)
    distances = _distribution_to_distance(box_distribution[keep])
    boxes_xyxy = _distance_to_boxes(distances, anchor_points[keep], anchor_strides[keep])
    scores, class_ids = scores[keep], class_ids[keep]

    survivors = _non_max_suppression(boxes_xyxy, scores, iou_threshold)
    return boxes_xyxy[survivors], scores[survivors], class_ids[survivors]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


_anchor_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _make_anchors(input_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Cell-center anchor points and their strides, one per box, matching YOLOv8's ordering.

    For each scale, every grid cell contributes one anchor at its center (+0.5), listed row by row;
    the three scales are concatenated in stride order - exactly how the model flattened its boxes.
    """
    if input_size in _anchor_cache:
        return _anchor_cache[input_size]
    points, strides = [], []
    for stride in STRIDES:
        cells = input_size // stride
        centers = np.arange(cells, dtype=np.float32) + 0.5
        grid_y, grid_x = np.meshgrid(centers, centers, indexing="ij")
        points.append(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1))
        strides.append(np.full(cells * cells, stride, dtype=np.float32))
    anchors = (np.concatenate(points), np.concatenate(strides))
    _anchor_cache[input_size] = anchors
    return anchors


def _distribution_to_distance(box_distribution: np.ndarray) -> np.ndarray:
    """DFL: each box edge is a softmax over REG_MAX bins; its distance is the expected bin index."""
    bins = box_distribution.reshape(-1, 4, REG_MAX)
    probabilities = _softmax(bins, axis=2)
    return (probabilities * np.arange(REG_MAX, dtype=np.float32)).sum(axis=2)  # (num_boxes, 4) = l,t,r,b


def _distance_to_boxes(distances: np.ndarray, anchor_points: np.ndarray, anchor_strides: np.ndarray) -> np.ndarray:
    """Turn left/top/right/bottom distances (in grid units) into pixel xyxy corners."""
    top_left = anchor_points - distances[:, :2]
    bottom_right = anchor_points + distances[:, 2:]
    boxes_grid = np.concatenate([top_left, bottom_right], axis=1)
    return boxes_grid * anchor_strides[:, np.newaxis]


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def map_boxes_to_frame(
    boxes_xyxy: np.ndarray, scale: float, pad_x: int, pad_y: int, frame_width: int, frame_height: int
) -> np.ndarray:
    """Undo the letterbox: shift out the padding and rescale boxes to the original frame."""
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_height)
    return boxes


def _non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS: keep the highest-scoring box, drop others overlapping it too much, repeat."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    kept: list[int] = []
    while order.size > 0:
        best = order[0]
        kept.append(best)
        overlap_x1 = np.maximum(x1[best], x1[order[1:]])
        overlap_y1 = np.maximum(y1[best], y1[order[1:]])
        overlap_x2 = np.minimum(x2[best], x2[order[1:]])
        overlap_y2 = np.minimum(y2[best], y2[order[1:]])
        overlap_w = np.clip(overlap_x2 - overlap_x1, 0, None)
        overlap_h = np.clip(overlap_y2 - overlap_y1, 0, None)
        intersection = overlap_w * overlap_h
        iou = intersection / (areas[best] + areas[order[1:]] - intersection)
        order = order[1:][iou <= iou_threshold]
    return kept
