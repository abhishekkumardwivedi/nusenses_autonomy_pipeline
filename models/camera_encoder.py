"""Pretrained ResNet-50 + fixed 1x1 reduction. No random-weight fallback."""
import os
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    from torch import nn
    import torchvision
    from torchvision.models import ResNet50_Weights, resnet50
except (ImportError, RuntimeError) as exc:
    raise RuntimeError("Stage 3 needs compatible torch/torchvision. Use the RunPod PyTorch environment; "
                       "do not mix Python 3.11 and 3.12 binary packages.") from exc

CAMERA_ORDER = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")


def preprocess(cameras):
    """BGR uint8 -> RGB -> 448x256 -> ImageNet-normalized [6,3,256,448]."""
    images = []
    for name in CAMERA_ORDER:
        image = cameras.get(name)
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Camera image missing or invalid BGR uint8: {name}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(cv2.resize(rgb, (448, 256), interpolation=cv2.INTER_AREA))
    array = np.stack(images).astype(np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(0, 3, 1, 2)))


class CameraEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: Stage 3 requires a CUDA GPU. Use --stage 2 for geometric playback.")
        cache = Path(os.environ.setdefault("TORCH_HOME", "/workspace/.cache/torch"))
        self.device = torch.device("cuda:0")
        self.inference_count = 0
        try:
            cache.mkdir(parents=True, exist_ok=True)
            network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except Exception as exc:
            raise RuntimeError(f"Cannot load pretrained ImageNet ResNet-50 weights from {cache}. "
                               "Check cache permissions / download.pytorch.org access. No random fallback.") from exc
        self.backbone = nn.Sequential(*list(network.children())[:-2])
        # ImageNet ResNet has 2048 output channels, not a pretrained 256-channel
        # neck. Deterministically average each group of 8 via a frozen 1x1 conv.
        # This projection is NOT learned; all learned weights are in the backbone.
        self.projection = nn.Conv2d(2048, 256, 1, groups=256, bias=False)
        with torch.no_grad():
            self.projection.weight.fill_(1 / 8)
        self.requires_grad_(False)
        self.eval()
        try:
            self.to(self.device)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("GPU OOM while loading ResNet-50; free GPU memory and restart Stage 3.") from exc

    def forward(self, images):
        return self.projection(self.backbone(images))

    @torch.inference_mode()
    def encode(self, cameras):
        start = time.perf_counter()
        batch = preprocess(cameras)
        if tuple(batch.shape) != (6, 3, 256, 448):
            raise ValueError(f"Preprocessing tensor shape mismatch: {tuple(batch.shape)}")
        preprocess_ms = (time.perf_counter() - start) * 1000
        try:
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            batch = batch.to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = self(batch)
            if tuple(features.shape) != (6, 256, 8, 14):
                raise ValueError(f"Encoder tensor shape mismatch: {tuple(features.shape)}")
            if not torch.isfinite(features).all().item():
                raise RuntimeError("Non-finite encoder features; check model/input and CUDA precision.")
            # Cache only the small tensor on CPU; release per-sample GPU activations.
            cached = features.unsqueeze(0).float().cpu()
            torch.cuda.synchronize(self.device)
            encoder_ms = (time.perf_counter() - start) * 1000
            del batch, features
            self.inference_count += 1
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("GPU OOM during six-camera inference; playback paused. Free GPU memory before retrying.") from exc
        metrics = {"enabled": True, "model": "ResNet-50", "weights": "ImageNet IMAGENET1K_V2",
                   "projection": "Fixed 1x1 group mean (8 backbone channels per output); not trained",
                   "device": str(self.device), "gpu": torch.cuda.get_device_name(self.device),
                   "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                   "dtype": "float16 autocast", "input_dtype": "float32", "cache_dtype": str(cached.dtype),
                   "cache_device": "cpu", "input_shape": [1, 6, 3, 256, 448],
                   "backbone_batch_shape": [6, 3, 256, 448], "backbone_output_shape": [6, 2048, 8, 14],
                   "output_shape": list(cached.shape), "camera_order": list(CAMERA_ORDER),
                   "feature_channels": 256, "feature_grid": [8, 14], "inference_count": self.inference_count,
                   "preprocess_ms": round(preprocess_ms, 2), "camera_encoder_ms": round(encoder_ms, 2),
                   "gpu_allocated_mb": round(torch.cuda.memory_allocated(self.device) / 2**20, 2),
                   "gpu_reserved_mb": round(torch.cuda.memory_reserved(self.device) / 2**20, 2),
                   "gpu_peak_mb": round(torch.cuda.max_memory_allocated(self.device) / 2**20, 2)}
        return cached, metrics
