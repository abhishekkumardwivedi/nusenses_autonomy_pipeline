# Stage 1: synthetic WebRTC video

Python 3.11, plain HTML/JS, 1280×720 at a target of 25 FPS. Frames are drawn
in memory with NumPy/OpenCV, wrapped in `av.VideoFrame`, and sent by aiortc.
No image files, image refreshes, datasets, or inference are involved.

## Run on RunPod

```bash
cd /workspace/autonomy/webrtc_test
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

The server binds to `0.0.0.0:8080`. Keep the terminal open. Ctrl+C shuts it
down and closes all peers. Run `curl http://127.0.0.1:8080/health` in a second
terminal; expect `{"status": "ok"}`.

Expose **8080 as an HTTP port** in the RunPod pod settings. Open the HTTP
proxy endpoint for port 8080 on your Windows PC:

https://9ykl8eclcuuqaa-8080.proxy.runpod.net

Click **Start Stream**. The browser needs no camera/microphone permission.
Expected status: Connecting → Connected. You should see a moving green circle,
an increasing frame counter, UTC timestamp, and generated FPS. Below the video,
received FPS, decoded frames, and received bytes come from WebRTC `getStats()`.

## Network prerequisite

The HTTP proxy carries the page and SDP signaling, **not WebRTC media**.
RunPod documents that pods do not expose inbound UDP. STUN may allow an
outbound-established direct path depending on NAT/firewall behavior, but it
cannot guarantee one. If the page/health works and ICE fails, use an existing
external TURN relay reachable by both machines (TCP/TLS is useful on restricted
networks). Before launching the server, set:

```bash
export TURN_URL='turns:YOUR_TURN_HOST:443?transport=tcp'
export TURN_USERNAME='YOUR_TURN_USERNAME'
export TURN_PASSWORD='YOUR_TURN_PASSWORD'
python server.py
```

The same ICE configuration is supplied to aiortc and the browser. Use temporary
test credentials: browser ICE credentials are necessarily visible to page
visitors. A TURN service is not bundled or provisioned by this test.

Reference: https://docs.runpod.io/pods/overview

## Routes and cleanup

- `GET /`: HTML page and ICE configuration.
- `GET /health`: `{"status":"ok"}`.
- `POST /offer`: SDP offer → SDP answer and peer ID.
- `POST /stop/{peer_id}`: release a peer when Stop or page unload occurs.

Stop closes the browser peer and requests immediate server cleanup. A lost
network or crashed browser is cleaned up when ICE detects failure; this takes
longer than pressing Stop. Unconnected peers expire after 60 seconds, including
aborted signaling requests. Server shutdown stops every track and peer.

## Expected console output

```text
webrtc INFO Starting synthetic WebRTC server on 0.0.0.0:8080; TURN not configured
======== Running on http://0.0.0.0:8080 ========
webrtc INFO peer <id> new connection
webrtc INFO peer <id> ICE: checking
webrtc INFO peer <id> connection: connecting
webrtc INFO peer <id> ICE: completed
webrtc INFO peer <id> connection: connected
webrtc INFO peer <id> stream start: 1280x720 @ 25 FPS
webrtc INFO peer <id> disconnected; track stopped; active peers=0
```

Timestamps and extra aiohttp/aioice logs are normal; ordering can vary.

## Verify actual streaming

1. Watch the circle, UTC time, and frame counter change continuously.
2. Confirm decoded frames and received bytes increase, with received FPS near 25
   after startup (actual performance depends on CPU/network/browser).
3. In browser DevTools → Network, observe one `/offer` request per Start and no
   repeated image downloads. The video element uses `srcObject`, not an image URL.
4. Use Chrome `chrome://webrtc-internals` or Edge `edge://webrtc-internals` to
   inspect the selected ICE candidate pair, inbound RTP, frames decoded, and FPS.
5. Press Stop or close the tab; look for server cleanup and `active peers=0`
   when this was the only viewer. Start again to confirm reconnection.

HTTP health alone does not prove that video traverses the network.

## Verified on this pod (2026-09-13)

- Python 3.11 imports and startup succeeded; socket bound to `0.0.0.0:8080`.
- Public HTTP proxy `/health` returned HTTP 200 and `{"status":"ok"}`.
- Windows browser connected through this page's proxy URL without TURN. Live
  video displayed at 1280×720, approximately 22–24 received FPS; the visible
  frame counter and decoded-frame/byte counters advanced continuously.
- Two pod-side WebRTC receiver runs each decoded 125 frames with strictly
  increasing timestamps and changes between all consecutive frames, at
  23.6–24.2 FPS including connection startup.
- Browser Stop released the track and returned the server to zero active peers.
- Browser reconnect resumed video at 24 FPS. Closing the browser tab also
  returned the server to zero active peers after ICE consent expired.

These results establish Stage 1 on the tested network. Other networks may
require the optional TURN settings above.
