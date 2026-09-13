# Stage 2: nuScenes -> WebRTC

This extends the Stage 1 app with synchronized recorded sample playback.
Six camera images and a geometric LiDAR/radar BEV are composed into one
1600x900 in-memory frame. No ML, image sequences, or intermediate image files.

## Install and launch (RunPod, Python 3.11)

```bash
cd /workspace/autonomy
python3.11 -m venv .venv-player
source .venv-player/bin/activate
pip install -r app/requirements.txt
export NUSCENES_DATAROOT=/workspace/data/nuscenes
export NUSCENES_VERSION=v1.0-mini
python app/server.py --stage 2
```

The dataset already on this pod is `/workspace/data/nuscenes`. If no environment
variable is set, the loader defaults to `/workspace/autonomy/datasets/nuscenes`.
The new environment uses NumPy 1.26 and headless OpenCV because nuScenes devkit
1.2 requires NumPy <2 and depends on headless OpenCV. It supplies the same `cv2`
API. Keep Stage 1's environment separate. No GPU or model is required.

Only one app can bind `0.0.0.0:8080` at a time. Stop the running Stage 1/2 server
with Ctrl+C before launching another. Stage 1 remains runnable with its original
environment and `python server.py` in `webrtc_test/`.

Open on your PC:

https://ikshk0dzrpwflf-8080.proxy.runpod.net/

Select a scene in the dropdown and press **Play**. Play also connects WebRTC if
needed. **Connect video** shows the current sample while paused. Pause holds the
sample; Next/Previous step one sample; Restart returns to sample 1 and pauses.
Scene selection also pauses. Play at the end restarts the scene. Rate choices
are 0.5x, 1x, 2x. All browser tabs share one timeline in this minimal test.
The last video peer disconnect pauses playback and releases its track.

Choose an initial scene using `?scene=scene-0061` (name or token), or set
`NUSCENES_SCENE=scene-0061` before launching. Changing the URL scene affects the
shared timeline. The selected scene must exist in the loaded dataset version.

## Expected dataset layout

```text
NUSCENES_DATAROOT/
  v1.0-mini/
    scene.json
    sample.json
    sample_data.json
    calibrated_sensor.json
    ego_pose.json
    ...all other nuScenes metadata tables...
  samples/
    CAM_FRONT/  CAM_FRONT_LEFT/  CAM_FRONT_RIGHT/
    CAM_BACK/   CAM_BACK_LEFT/   CAM_BACK_RIGHT/
    LIDAR_TOP/
    RADAR_FRONT/ RADAR_FRONT_LEFT/ RADAR_FRONT_RIGHT/
    RADAR_BACK_LEFT/ RADAR_BACK_RIGHT/
  sweeps/    # present in the official dataset; not used for this keyframe player
  maps/
```

For full nuScenes, point `NUSCENES_DATAROOT` at a complete extracted dataset and
set `NUSCENES_VERSION=v1.0-trainval`. Metadata and the referenced sensor files
must both exist. Missing required files produce an explicit error and pause
playback; they are not silently substituted with stale sensor views.

## Synchronization and coordinates

The scene follows `first_sample_token` and each sample's `next` link. All sensor
files come from `sample['data'][channel]` -> `sample_data['filename']`.
The six cameras keep their native perspectives (back views are not mirrored).

Every LiDAR/radar point follows this transform:

```text
reference_ego_from_sensor = inverse(global_from_reference_ego)
                         @ global_from_capture_ego
                         @ capture_ego_from_sensor_calibration
```

The reference is the `LIDAR_TOP` ego pose of the displayed sample. Each radar
uses its own sensor calibration and its own capture-time ego pose. Geometry is
isolated in `nuscenes_player/geometry.py` and independent of rendering.
BEV uses meters: x forward/up-screen, y left/left-screen, z up. The ego is green
at the center; LiDAR is cyan; radar is orange. Grid spacing is 10m; display range
is +/-50m. Point counts include points outside the displayed range. Radar uses
the devkit's default quality/dynamic-state filters; no detections or velocity
vectors are inferred. All finite point heights are flattened into top-down XY.

To verify synchronization, pause, open **Sample and sensor synchronization
details**, and note the sample token, timestamp and per-channel record tokens.
Press Next: all six cameras, LiDAR, radar, sample index and timestamp must advance
as one set. Each sensor's `offset_ms` exposes its capture-time difference from
the sample timestamp; small nonzero offsets are normal. Camera labels and the
video header are embedded in the same encoded frame as the sensor BEV.
JSON status polling can arrive slightly ahead of buffered video; use the
in-video sample header as the identity of the frame actually being viewed.

## Timing and limits

nuScenes keyframes are approximately 2 Hz, not 25 FPS camera video. The player
schedules samples using their actual microsecond timestamp differences and
the selected rate. WebRTC refreshes the latest composite at a target of 25 FPS.
Held frames between samples are expected; there is no sensor interpolation.
Source Hz, actual sample advances/second, and received WebRTC FPS are distinct.
If disk reads/rendering fall behind dataset timing, playback can lag and catch
up; it does not fabricate samples or silently skip them. Ego-motion compensation
aligns sensor captures geometrically but does not compensate moving objects.

## Files and API

- `nuscenes_player/nuscenes_source.py`: devkit/sample-linked sensor loading.
- `nuscenes_player/geometry.py`: reusable calibration and ego-pose transforms.
- `nuscenes_player/renderer.py`: cameras and top-down sensor visualization.
- `nuscenes_player/player.py`: timeline, controls and video track.
- `app/server.py`, `app/index.html`: Stage 2 entry point and plain browser UI.
- `webrtc_test/server.py`: shared Stage 1 signaling/cleanup, configurable track.
- `webrtc_test/client.js`: extracted shared browser WebRTC logic.
- `app/verify.py`: real-data synchronization, geometry and playback checks.

`GET /`, `/client.js`, `/health`, `POST /offer`, `/stop/{peer_id}` are shared with
Stage 1. Stage 2 adds `GET /scenes`, `GET /state`, `POST /control`.
Example control body: `{"action":"scene","value":"scene-0061"}`;
other actions are `play`, `pause`, `next`, `previous`, `restart`, `rate`.

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/state
python app/verify.py
```

Console output includes selected scene, sample token/index/timestamp, loaded
sensor tokens and offsets, LiDAR/radar counts, load+render milliseconds, peer
connection/ICE/cleanup, and WebRTC output FPS every five seconds.

The same optional TURN environment variables from Stage 1 remain supported.
The HTTP proxy serves signaling; media uses WebRTC ICE connectivity.

References:
- https://www.nuscenes.org/public/tutorials/nuscenes_tutorial.html
- https://github.com/nutonomy/nuscenes-devkit

## Verification on RunPod (2026-09-13)

`python app/verify.py` passed on the installed mini dataset: all six cameras
and LiDAR/radar change between samples; sample-data linkage is correct; radar
transforms match the devkit's independent single-sweep transform; controls,
scene changes, and first/last sample boundaries work. `pip check` passed.

The Windows browser received 1600x900 WebRTC video at approximately 20-25 FPS.
Scene-0061 played through 39 samples at about 1.9 samples/second and held the
last sample. Selecting scene-0103 and stepping to sample 2 updated cameras,
BEV, counts, sample index and timestamp together. Shared-client Stage 1 playback
was also rechecked at 24 FPS. `/health` returned status ok through the HTTP proxy.
