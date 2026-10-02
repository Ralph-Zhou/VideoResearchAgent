"""CLIP embedding for images (keyframes) and text (queries).

A tiny wrapper around ``transformers.CLIPModel`` that:

- loads the model + processor lazily the first time an encode method is called,
- exposes pure-numpy APIs (so callers don't need torch in their surface area),
- runs both vision and text on the configured device with the configured dtype,
- always returns L2-normalised float32 vectors when ``normalize=True``, so
  downstream FAISS ``IndexFlatIP`` searches compute cosine similarity.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..common.config import CLIPConfig

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "float16": "float16",
    "float32": "float32",
    "bfloat16": "bfloat16",
}


class CLIPEmbedder:
    """Stateful CLIP encoder. Not thread-safe; construct one per worker."""

    def __init__(self, cfg: CLIPConfig):
        self.cfg = cfg
        self._model = None  # lazy
        self._processor = None
        self._device = cfg.device
        self._torch = None

    # ------------------------------------------------------------------ util

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Missing torch/transformers; install via the package's `pyproject` dependencies."
            ) from e

        self._torch = torch

        dtype_str = _DTYPE_MAP[self.cfg.dtype]
        torch_dtype = getattr(torch, dtype_str)

        logger.info(
            "Loading CLIP model %s on %s (dtype=%s)",
            self.cfg.model_name,
            self.cfg.device,
            dtype_str,
        )
        # ``torch_dtype`` only applies to vision/text encoders, which is what we want.
        self._processor = CLIPProcessor.from_pretrained(self.cfg.model_name)
        self._model = CLIPModel.from_pretrained(
            self.cfg.model_name, torch_dtype=torch_dtype
        )
        self._model.eval()
        if self._device != "cpu":
            self._model = self._model.to(self._device)

    @staticmethod
    def _to_numpy_normalized(tensor, normalize: bool) -> np.ndarray:
        """Convert a float tensor to float32 numpy, optionally L2-normalised."""
        arr = (
            tensor.detach()
            .to("cpu", dtype=tensor.dtype if tensor.dtype.is_floating_point else None)
            .float()
            .numpy()
        )
        if normalize:
            norms = np.linalg.norm(arr, axis=-1, keepdims=True)
            arr = arr / np.clip(norms, 1e-12, None)
        return arr.astype(np.float32, copy=False)

    # --------------------------------------------------------------- encode

    def encode_images(self, images: list[PILImage]) -> np.ndarray:
        """Encode a list of PIL images to a ``(N, D)`` float32 numpy array."""
        if not images:
            return np.zeros((0, self.cfg.embedding_dim), dtype=np.float32)

        self._ensure_loaded()
        assert self._model is not None and self._processor is not None
        torch = self._torch
        assert torch is not None

        outs: list[np.ndarray] = []
        bs = self.cfg.batch_size
        with torch.inference_mode():
            for i in range(0, len(images), bs):
                chunk = images[i : i + bs]
                inputs = self._processor(images=chunk, return_tensors="pt")
                pixel_values = inputs["pixel_values"]

                if self._device != "cpu":
                    pixel_values = pixel_values.to(self._device)

                # Convert data type to match model weights
                pixel_values = pixel_values.to(self._model.dtype)

                # Extract raw features
                feats = self._model.get_image_features(pixel_values=pixel_values)

                # --- CORE FIX START ---
                # If the output is a model output object instead of a tensor,
                # extract the specific feature attribute.
                if not hasattr(feats, "detach"):
                    # Attempt to get 'image_embeds' (standard CLIP output attribute).
                    # Fallback to 'pooler_output' (generic vision model attribute) if not found.
                    feats = getattr(
                        feats, "image_embeds", getattr(feats, "pooler_output", feats)
                    )
                # --- CORE FIX END ---

                outs.append(
                    self._to_numpy_normalized(feats, normalize=self.cfg.normalize)
                )

        return (
            np.concatenate(outs, axis=0)
            if outs
            else np.zeros((0, self.cfg.embedding_dim), dtype=np.float32)
        )

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings to a ``(N, D)`` float32 numpy array."""
        if not texts:
            return np.zeros((0, self.cfg.embedding_dim), dtype=np.float32)

        self._ensure_loaded()
        assert self._model is not None and self._processor is not None
        torch = self._torch
        assert torch is not None

        outs: list[np.ndarray] = []
        bs = self.cfg.batch_size
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                chunk = texts[i : i + bs]
                inputs = self._processor(
                    text=chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                if self._device != "cpu":
                    input_ids = input_ids.to(self._device)
                    attention_mask = attention_mask.to(self._device)

                # Retrieve the feature object/tensor
                feats = self._model.get_text_features(
                    input_ids=input_ids, attention_mask=attention_mask
                )

                # --- CORE FIX START: Ensure compatibility with different transformers versions ---
                if not hasattr(feats, "detach"):
                    # Attempt to get 'text_embeds' (standard attribute for CLIP text features).
                    # Fallback to 'pooler_output' (pooled output for generic models).
                    feats = getattr(
                        feats, "text_embeds", getattr(feats, "pooler_output", feats)
                    )
                # --- CORE FIX END ---

                outs.append(
                    self._to_numpy_normalized(feats, normalize=self.cfg.normalize)
                )

        return (
            np.concatenate(outs, axis=0)
            if outs
            else np.zeros((0, self.cfg.embedding_dim), dtype=np.float32)
        )
