"""Visual grounding tool based on Grounding DINO.

Provides open-vocabulary object detection on video frames: given a text
prompt describing objects of interest, returns bounding boxes with
confidence scores and annotated frames with detections drawn.

Supported models (via config grounding.model_id):
  - IDEA-Research/grounding-dino-tiny  (CPU-friendly, fast, good for dev)
  - IDEA-Research/grounding-dino-base  (best open-source accuracy, needs GPU)
"""

import base64
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Cache: (model_id, resolved_device) -> (model, processor)
_model_cache: Dict[Tuple[str, str], tuple] = {}

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


@dataclass
class Detection:
    label: str
    score: float
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class FrameDetectionResult:
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    annotated_b64: str = ""


def _resolve_device(device: str = "auto") -> str:
    """Resolve device string: 'auto' picks CUDA if available."""
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_model(model_id: str = DEFAULT_MODEL_ID, device: str = "auto"):
    """Lazy-load Grounding DINO model and processor.

    Models are cached by model_id so switching between tiny/base doesn't
    require reloading if the same model is requested again.

    The model is kept in fp32 and inference uses torch.autocast for
    automatic mixed precision — this avoids dtype mismatch errors that
    occur when manually casting to fp16 (Grounding DINO's text encoder
    and cross-attention create fp32 intermediates internally).
    """
    resolved_device = _resolve_device(device)
    cache_key = (model_id, resolved_device)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    logger.info("Loading Grounding DINO: %s on %s ...", model_id, resolved_device)

    processor = AutoProcessor.from_pretrained(model_id)

    def _has_meta_tensor(model) -> bool:
        return any(p.device.type == "meta" for p in model.parameters())

    def _load_eager_cpu():
        # Force eager CPU materialization first. Some transformers/torch
        # combinations default to meta tensors for large checkpoints; calling
        # `.to(cuda)` on such a partially materialized model raises:
        # "Cannot copy out of meta tensor; no data!".
        return AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            low_cpu_mem_usage=False,
        )

    model = _load_eager_cpu()
    if _has_meta_tensor(model):
        logger.warning(
            "Grounding DINO loaded with meta tensors via eager CPU path; "
            "retrying with device_map for %s",
            resolved_device,
        )
        kwargs = {"low_cpu_mem_usage": True}
        if resolved_device != "cpu":
            kwargs["device_map"] = {"": resolved_device}
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            **kwargs,
        )
    elif resolved_device != "cpu":
        try:
            model = model.to(resolved_device)
        except NotImplementedError as e:
            if "meta tensor" not in str(e):
                raise
            logger.warning(
                "Grounding DINO .to(%s) hit meta tensor error; "
                "retrying with device_map",
                resolved_device,
            )
            kwargs = {"low_cpu_mem_usage": True}
            kwargs["device_map"] = {"": resolved_device}
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id,
                **kwargs,
            )
    model.eval()
    logger.info("Model loaded (fp32) on %s", resolved_device)

    _model_cache[cache_key] = (model, processor)
    return model, processor


def detect_objects(
    image_bgr: np.ndarray,
    prompt: str,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "auto",
) -> List[Detection]:
    """Run Grounding DINO on a single BGR image."""
    import torch
    from PIL import Image

    model, processor = _load_model(model_id, device)
    model_device = next(model.parameters()).device
    use_amp = str(model_device).startswith("cuda")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    h, w = image_bgr.shape[:2]

    labels_list = [s.strip() for s in prompt.split(".") if s.strip()]
    text_labels = [labels_list]

    inputs = processor(images=pil_image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items() if hasattr(v, "to")}

    with torch.no_grad():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )

    detections = []
    if results:
        result = results[0]
        labels_key = "text_labels" if "text_labels" in result else "labels"
        for box_t, score_t, label in zip(result["boxes"], result["scores"], result[labels_key]):
            box = box_t.cpu().tolist()
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            detections.append(Detection(
                label=label,
                score=round(score_t.item(), 3),
                box=(x1, y1, x2, y2),
            ))

    return detections


def annotate_frame(
    image_bgr: np.ndarray,
    detections: List[Detection],
    jpeg_quality: int = 85,
) -> str:
    """Draw bounding boxes and labels on a frame, return base64 JPEG."""
    canvas = image_bgr.copy()
    label_color_map = {}
    color_idx = 0

    for det in detections:
        if det.label not in label_color_map:
            label_color_map[det.label] = _COLORS[color_idx % len(_COLORS)]
            color_idx += 1
        color = label_color_map[det.label]
        x1, y1, x2, y2 = det.box

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        text = f"{det.label} {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return base64.b64encode(buf).decode("utf-8")


def save_annotated_frame(
    image_bgr: np.ndarray,
    detections: List[Detection],
    output_path: str,
    jpeg_quality: int = 95,
) -> str:
    """Draw bounding boxes on a frame and save to disk. Returns the output path."""
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]
    label_color_map = {}
    color_idx = 0
    line_w = max(2, int(min(h, w) / 300))
    font_scale = max(0.5, min(h, w) / 1200)

    for det in detections:
        if det.label not in label_color_map:
            label_color_map[det.label] = _COLORS[color_idx % len(_COLORS)]
            color_idx += 1
        color = label_color_map[det.label]
        x1, y1, x2, y2 = det.box

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_w)

        text = f"{det.label} {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(canvas, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(canvas, text, (x1 + 3, max(y1 - 4, th + 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return output_path


class VisualGroundingTool:
    """High-level tool: extract frames from video, detect objects, return results."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
        device: str = "auto",
        jpeg_quality: int = 85,
    ):
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.jpeg_quality = jpeg_quality

    def detect_in_video(
        self,
        video_path: str,
        prompt: str,
        timestamps: Optional[List[float]] = None,
        max_frames: int = 4,
    ) -> List[FrameDetectionResult]:
        """Detect objects in video frames at given timestamps.

        If timestamps is None, samples max_frames evenly across the video.
        """
        frames_bgr, actual_ts = self._extract_frames(video_path, timestamps, max_frames)

        results = []
        for bgr, ts in zip(frames_bgr, actual_ts):
            detections = detect_objects(
                bgr, prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                model_id=self.model_id,
                device=self.device,
            )
            annotated_b64 = annotate_frame(bgr, detections, self.jpeg_quality) if detections else ""
            results.append(FrameDetectionResult(
                timestamp=ts,
                detections=detections,
                annotated_b64=annotated_b64,
            ))
        return results

    @staticmethod
    def _extract_frames(
        video_path: str,
        timestamps: Optional[List[float]],
        max_frames: int,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Extract BGR frames at specified timestamps (or uniformly sampled)."""
        try:
            from decord import VideoReader, cpu as decord_cpu
            vr = VideoReader(video_path, ctx=decord_cpu(0))
            total = len(vr)
            fps = vr.get_avg_fps()

            if timestamps is None:
                n = min(max_frames, total)
                indices = np.linspace(0, total - 1, n, dtype=int).tolist()
                actual_ts = [round(i / fps, 2) for i in indices]
            else:
                timestamps = sorted(timestamps)[:max_frames]
                indices = [min(int(t * fps), total - 1) for t in timestamps]
                actual_ts = [round(t, 2) for t in timestamps]

            batch = vr.get_batch(indices).asnumpy()
            frames_bgr = [frame[:, :, ::-1].copy() for frame in batch]
            return frames_bgr, actual_ts
        except Exception as e:
            logger.warning("decord failed, falling back to cv2: %s", e)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if timestamps is None:
            n = min(max_frames, total)
            indices = np.linspace(0, total - 1, n, dtype=int).tolist()
            actual_ts = [round(i / fps, 2) if fps > 0 else 0 for i in indices]
        else:
            timestamps = sorted(timestamps)[:max_frames]
            indices = [min(int(t * fps), total - 1) for t in timestamps]
            actual_ts = [round(t, 2) for t in timestamps]

        frames_bgr = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames_bgr.append(frame)
            else:
                frames_bgr.append(np.zeros((480, 640, 3), dtype=np.uint8))
        cap.release()
        return frames_bgr, actual_ts[:len(frames_bgr)]
