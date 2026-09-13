"""Stage 1: in-memory synthetic video over WebRTC. Python 3.11."""
import asyncio
import json
import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc import VideoStreamTrack
from av import VideoFrame

LOG = logging.getLogger("webrtc")
WIDTH, HEIGHT, FPS = 1280, 720, 25
# Optional external TURN relay; HTTP proxies do not carry WebRTC media.
ICE_SERVERS = [{"urls": "stun:stun.l.google.com:19302"}]
if os.getenv("TURN_URL"):
    ICE_SERVERS.append({"urls": os.environ["TURN_URL"],
                        "username": os.environ["TURN_USERNAME"],
                        "credential": os.environ["TURN_PASSWORD"]})
CONFIG = RTCConfiguration(iceServers=[RTCIceServer(**s) for s in ICE_SERVERS])
peers = {}


class SyntheticVideoTrack(VideoStreamTrack):
    """Replace frame drawing here with a future frame source."""

    def __init__(self, peer_id):
        super().__init__()
        self.peer_id = peer_id
        self.number = 0
        self.started = None
        self.next_frame = 0

    async def recv(self):
        if self.started is None:
            self.started = time.monotonic()
            LOG.info("peer %s stream start: %sx%s @ %s FPS", self.peer_id, WIDTH, HEIGHT, FPS)
        elapsed = time.monotonic() - self.started
        await asyncio.sleep(max(0, self.next_frame - elapsed))
        elapsed = time.monotonic() - self.started
        self.next_frame = max(self.next_frame + 1 / FPS, elapsed + 1 / FPS)
        # Real elapsed timestamps avoid bursts / slow playback after a stall.
        self.number += 1
        frame = np.full((HEIGHT, WIDTH, 3), (32, 24, 18), dtype=np.uint8)
        x = int((0.5 + 0.5 * math.sin(elapsed * 1.4)) * (WIDTH - 160)) + 80
        y = int(HEIGHT * 0.65 + math.sin(elapsed * 2) * 100)
        cv2.circle(frame, (x, y), 55, (60, 220, 100), -1)
        fps = (self.number - 1) / elapsed if elapsed > 0 else 0
        lines = ["LIVE SYNTHETIC WEBRTC", f"Frame: {self.number}",
                 datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                 f"{WIDTH} x {HEIGHT} | Generated FPS: {fps:.1f} | Target: {FPS}"]
        for row, label in enumerate(lines):
            cv2.putText(frame, label, (35, 55 + row * 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
        video = VideoFrame.from_ndarray(frame, format="bgr24")
        video.pts = round(elapsed * 90000)
        video.time_base = Fraction(1, 90000)
        return video


async def close_peer(peer_id):
    entry = peers.pop(peer_id, None)
    if entry:
        pc, track, timer = entry
        timer.cancel()
        track.stop()
        await pc.close()
        LOG.info("peer %s disconnected; track stopped; active peers=%s", peer_id, len(peers))


async def index(request):
    html = request.app["index_path"].read_text()
    config = json.dumps(ICE_SERVERS).replace("<", "\\u003c")
    return web.Response(text=html.replace("__ICE_SERVERS__", config),
                        content_type="text/html", headers={"Cache-Control": "no-store"})


async def client_script(request):
    return web.FileResponse(Path(__file__).with_name("client.js"),
                            headers={"Cache-Control": "no-store"})


async def health(request):
    return web.json_response({"status": "ok"})


async def offer(request):
    try:
        params = await request.json()
        if params.get("type") != "offer" or not isinstance(params.get("sdp"), str):
            raise ValueError("Expected SDP offer")
    except (ValueError, AttributeError):
        raise web.HTTPBadRequest(text="Expected JSON with type=offer and sdp")
    peer_id = uuid.uuid4().hex
    pc = RTCPeerConnection(CONFIG)
    track = request.app["track_factory"](peer_id)
    timer = asyncio.get_running_loop().call_later(
        60, lambda: asyncio.create_task(close_peer(peer_id)))
    peers[peer_id] = (pc, track, timer)
    LOG.info("peer %s new connection", peer_id)

    @pc.on("iceconnectionstatechange")
    async def ice_state():
        LOG.info("peer %s ICE: %s", peer_id, pc.iceConnectionState)

    @pc.on("connectionstatechange")
    async def connection_state():
        LOG.info("peer %s connection: %s", peer_id, pc.connectionState)
        if pc.connectionState == "connected":
            timer.cancel()
        elif pc.connectionState in ("failed", "closed"):
            await close_peer(peer_id)

    try:
        await pc.setRemoteDescription(RTCSessionDescription(**params))
        pc.addTrack(track)
        await pc.setLocalDescription(await pc.createAnswer())
        return web.json_response({"sdp": pc.localDescription.sdp,
                                  "type": pc.localDescription.type, "peer_id": peer_id})
    except Exception:
        LOG.exception("peer %s negotiation error", peer_id)
        await close_peer(peer_id)
        raise web.HTTPBadRequest(text="WebRTC negotiation failed; see server log")


async def stop(request):
    await close_peer(request.match_info["peer_id"])
    return web.Response(status=204)


async def shutdown(app):
    await asyncio.gather(*(close_peer(key) for key in list(peers)))


def create_app(track_factory=SyntheticVideoTrack, index_path=None):
    """Reuse the tested signaling and cleanup with another video source."""
    app = web.Application()
    app["track_factory"] = track_factory
    app["index_path"] = Path(index_path) if index_path else Path(__file__).with_name("index.html")
    app.add_routes([web.get("/", index), web.get("/health", health),
                    web.get("/client.js", client_script),
                    web.post("/offer", offer), web.post("/stop/{peer_id}", stop)])
    app.on_shutdown.append(shutdown)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    LOG.info("Starting synthetic WebRTC server on 0.0.0.0:8080; TURN %s",
             "configured" if os.getenv("TURN_URL") else "not configured")
    web.run_app(create_app(), host="0.0.0.0", port=8080)
