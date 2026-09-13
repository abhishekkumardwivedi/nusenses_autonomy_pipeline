# Stage 3: camera encoder and feature inspection

Stage 3 adds the first neural-network inference: six camera images pass through
one ImageNet-pretrained ResNet-50. LiDAR/radar remain raw sensor geometry. No
camera-to-BEV learning, depth, detection, segmentation, fusion or training.

The loader, sample synchronization, ego transforms, timeline, port 8080,
browser transport and single WebRTC peer per viewer are reused. The root README
contains the architecture diagram. Stage 2 remains available via `--stage 2`.

## Model and tensor contract

- torchvision `ResNet50_Weights.IMAGENET1K_V2`, ending at `layer4` before global
  pooling/classification. Pretrained weights must load successfully; errors
  stop Stage 3, with no random-weight fallback.
- Six BGR uint8 OpenCV images are converted to RGB, resized to **448x256**,
  divided by 255, normalized with mean `[.485,.456,.406]` and standard deviation
  `[.229,.224,.225]`, then arranged CHW. Resize stretches to the requested aspect
  ratio and intentionally does not use the classifier's center-crop recipe.
- Logical input `[1,6,3,256,448]`; shared backbone batch `[6,3,256,448]`;
  backbone output `[6,2048,8,14]`; projected output **`[1,6,256,8,14]`**.
- Camera order: front-left, front, front-right, back-left, back, back-right.
  The explicit channel names are also returned in `/state`.
- **Projection is fixed, not learned.** ImageNet ResNet-50 does not supply a
  trained 256-channel projection. A frozen grouped 1x1 convolution averages each
  contiguous group of eight backbone channels. This deterministic reduction
  avoids adding random weights and preserves the pretrained origin of features.
  It is not a trained fusion neck; Stage 4 must evaluate or replace this choice.
- `eval()`, frozen parameters, `torch.inference_mode()`, CUDA FP16 autocast;
  inputs remain float32. Cached tensor is CPU float32, with a shared NumPy view
  for rendering. `player.data['feature_tensor']` exposes the logical tensor for
  later code, without browser or WebRTC dependencies in the encoder module.

## One inference per changed sample

`Player.load()` loads all linked sensors, encodes the six-camera batch once,
then atomically publishes sensor data, feature tensor, metrics and visualization.
The current sample stays cached. A different sample, stepping backward, or a
different scene triggers one inference. Re-requesting the current sample does
not. Pause, WebRTC refreshes, feature camera/mode/channel changes, and focus
changes do not run the model. The cumulative `inference_count` makes this
observable. Only the current sample is cached, not an entire dataset.

## Feature and focus controls

Select **Feature camera** (default CAM_FRONT), then **Mean activation** or
**Channel** (integer 0-255). Mean is `mean(abs(feature), axis=channels)`; Channel
is the selected activation map. Each map is independently min/max normalized,
with a constant map displayed at zero. Turbo colors are a visualization scale,
not classes. The displayed min/max values preserve context for normalization;
colors cannot be compared quantitatively between independently normalized maps.

Double-click a visible panel or use its small Expand button. Double-click
again, press Esc, use Minimize, or select Overview to return. The Focus dropdown
also exposes **combined BEV, LiDAR-only, radar-only, all six cameras and feature
inspection**. All controls operate on the same shared timeline and WebRTC stream.

Focused cameras use original sensor images, preserving their aspect ratio.
Focused BEV is freshly rasterized from original transformed points at the full
panel resolution, never scaled from the overview bitmap. Ego axes remain
x forward/up-screen, y left, z up; equal XY scale, +/-50m range, 10m grid.
Focused features show the actual **8x14** grid with nearest-neighbor enlargement;
enlarging it creates no new spatial information. Scene/sample changes update the
current focus. Browser hit testing accounts for video letterboxing.

## Runtime information and timing

The webpage explains the processing sequence and displays actual `/state`
measurements: model/weights/projection, device/GPU, versions, dtypes, input and
output shapes, sample index/token/timestamp, per-camera dt, sensor counts,
source rate, actual playback rate and WebRTC receive FPS.

Per-sample timings (milliseconds): `sensor_load_ms`, `preprocess_ms`,
`camera_encoder_ms`, `render_ms`, `total_ms`. Encoder timing uses CUDA
synchronization and includes transfer to GPU, backbone/projection, finite-value
validation and copying the small result to CPU. `total_ms` covers the full
sample processing; `view_render_ms` separately measures redraws of cached data.
The first inference includes initialization overhead and is slower.

GPU metrics use PyTorch allocated/reserved/max-allocated bytes divided by 2^20
(MiB). Values are sampled after encoding; the peak is process-lifetime. They
exclude some CUDA driver/context and non-PyTorch allocations, so they are not
equivalent to total memory reported by `nvidia-smi`.

**dt** is sensor capture-time offset relative to the selected nuScenes sample
time. Batching images for inference does not remove temporal skew. The existing
point transforms use each sensor's own capture pose and the sample LiDAR ego
reference. Camera features are perspective features, not ego-frame BEV.

## Exact RunPod setup and launch

Inspect the installed runtime first:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The tested pod already provides Python 3.12.3, PyTorch 2.8.0+cu128, torchvision
0.23.0+cu128, CUDA runtime 12.8 and an NVIDIA RTX 2000 Ada Generation (16GB).
No CUDA/PyTorch reinstall or downgrade was performed. A system-site-packages
venv reuses that stack while installing the small application dependencies:

```bash
cd /workspace/autonomy
python -m venv --system-site-packages .venv-stage3
source .venv-stage3/bin/activate
pip install -r app/requirements.txt
export NUSCENES_DATAROOT=/workspace/data/nuscenes
export NUSCENES_VERSION=v1.0-mini
export TORCH_HOME=/workspace/.cache/torch
python app/server.py --stage 3
```

Only one server can bind `0.0.0.0:8080`. Stop the running server before launching
another. First startup downloads the official 97.8MB ResNet checkpoint to the
persistent TORCH_HOME. Later pod recreation reuses the cache, but its Python
runtime must still provide compatible torch/torchvision packages.

Open https://ikshk0dzrpwflf-8080.proxy.runpod.net/ and press Play or Connect video.
The latter shows a paused sample for inspection. `/health` returns status ok.
Keep Stage 2's Python 3.11 venv separate; it cannot load Python 3.12 binaries.

## Verification

With the Stage 3 environment and variables above:

```bash
python app/verify_stage3.py
python app/verify.py
```

The CUDA script checks correct preprocessing and camera order, eval/frozen/no
gradient behavior, output shape/finite values, differences between cameras and
samples, invalid channel rejection, no inference on view changes or Pause,
exactly one inference per new sample on Next/Play, focused scene changes, and
60 actually decoded WebRTC frames at 1600x900 plus peer cleanup. The Stage 2
script independently checks radar transforms against the nuScenes devkit and
the preserved sensor/playback behavior.

On 2026-09-13 both scripts passed. One warm sample measured 223.2ms sensor load,
38.77ms preprocessing, 43.41ms encoder, 41.0ms render, 346.9ms total. PyTorch
allocated 89.92MiB, reserved 218MiB, peak allocated 174.04MiB. These are observed
sample measurements, not fixed latency guarantees; cold inference is slower.

## Limits and errors

Keyframes are about 2Hz. WebRTC refresh near 25Hz repeats the cached frame; it
does not imply new sensor captures/inference. Slow storage, GPU load or higher
playback rate can cause lag/catch-up. There is no interpolation. Status JSON may
arrive ahead of buffered video; the in-video sample header identifies the actual
displayed sample. Focus is shared between viewers in this single-session app.

CUDA unavailable, missing/incompatible torchvision, weight cache/download
failures, missing cameras, invalid feature channels, shape mismatches, non-finite
outputs and GPU OOM produce errors. Startup failures stop Stage 3. Sample-time
failures pause playback, retain the last complete frame and expose the error.
Use `--stage 2` explicitly for geometric-only playback; there is no silent CPU or
random model fallback. No dataset, weights, caches or image sequences are tracked
in Git. The model is ImageNet-pretrained, not trained for autonomous driving.

Reference: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
