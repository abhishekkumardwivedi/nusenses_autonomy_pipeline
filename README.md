# nuScenes autonomy playback

Minimal Python + plain HTML WebRTC experiments for RunPod. Current scope stops
at recorded synchronized sensor visualization. No neural networks or inference.

- **Stage 1:** in-memory synthetic live video; [instructions](webrtc_test/README.md).
- **Stage 2:** nuScenes devkit sample playback with six cameras, LiDAR/radar
  geometric BEV, and Play/Pause/step/scene controls; [instructions](app/README.md).

## Run Stage 2 on RunPod

```bash
cd /workspace/autonomy
python3.11 -m venv .venv-player
source .venv-player/bin/activate
pip install -r app/requirements.txt
export NUSCENES_DATAROOT=/workspace/data/nuscenes
export NUSCENES_VERSION=v1.0-mini
python app/server.py
```

The default dataset root (when unset) is `/workspace/autonomy/datasets/nuscenes`.
Dataset files are not included. See the Stage 2 instructions for layout,
`v1.0-trainval`, timing limitations, coordinate conventions and verification.
Only one server can use port 8080; stop an existing server before launching.

Open your RunPod HTTP proxy endpoint for port 8080, select a scene, and press
Play. `/health` returns `{"status":"ok"}`. Media uses WebRTC; the HTTP proxy
carries the page and SDP signaling. Optional TURN settings are documented in
the Stage 1 instructions for networks where direct ICE connectivity fails.
