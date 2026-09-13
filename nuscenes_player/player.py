"""One shared timeline. Sensor I/O/rendering run off the asyncio event loop."""
import asyncio
import logging
import time
from fractions import Fraction
from aiortc import VideoStreamTrack
from av import VideoFrame
from .renderer import render, FOCUS_MODES, PANELS
from .nuscenes_source import CAMERAS

LOG = logging.getLogger("player")


class Player:
    def __init__(self, source, scene, encoder=None):
        self.source, self.initial_scene = source, scene
        self.lock = asyncio.Lock()
        self.playing, self.rate, self.error = False, 1.0, None
        self.frame, self.index, self.viewers = None, 0, 0
        self.scene, self.samples, self.details = None, [], {}
        self.due = 0
        self.fps_start, self.advances = time.monotonic(), 0
        self.task = None
        self.encoder, self.data = encoder, None
        self.view = {"focus": "overview", "feature_camera": "CAM_FRONT", "feature_mode": "mean", "feature_channel": 0}

    def _render(self, scene, samples, index):
        start = time.monotonic()
        data = self.source.load(samples[index])
        loaded = time.monotonic()
        encoder_info = {"enabled": False, "preprocess_ms": 0, "camera_encoder_ms": 0, "inference_count": 0}
        if self.encoder is not None:
            tensor, encoder_info = self.encoder.encode(data["cameras"])
            data["feature_tensor"] = tensor
            data["features"] = tensor.numpy()
        render_start = time.monotonic()
        frame = render(data, scene, samples[index], index, len(samples), self.view, encoder_info)
        rendered = time.monotonic()
        ms = (time.monotonic() - start) * 1000
        details = {"sensors": data["sensors"], "ego_pose": data["ego_pose"],
                   "lidar_points": data["lidar"].shape[1], "radar_points": data["radar"].shape[1],
                   "load_ms": round((loaded - start) * 1000, 1),
                   "sensor_load_ms": round((loaded - start) * 1000, 1),
                   "preprocess_ms": encoder_info["preprocess_ms"], "camera_encoder_ms": encoder_info["camera_encoder_ms"],
                   "render_ms": round((rendered - render_start) * 1000, 1), "load_render_ms": round(ms, 1),
                   "total_ms": round(ms, 1), "view_render_ms": 0, "encoder": encoder_info}
        LOG.info("scene=%s sample=%s/%s timestamp=%s load=%.1fms render=%.1fms total=%.1fms",
                 scene["name"], index + 1, len(samples), samples[index]["timestamp"],
                 details["load_ms"], details["render_ms"], ms)
        if self.encoder is not None:
            LOG.info("Camera encoder #%s preprocess=%.2fms inference=%.2fms shape=%s",
                     encoder_info["inference_count"], details["preprocess_ms"], details["camera_encoder_ms"], encoder_info["output_shape"])
        return frame, details, data

    async def load(self, scene, samples, index):
        if self.data is not None and self.samples[self.index]["token"] == samples[index]["token"]:
            return
        # Publish frame and metadata together, only after every sensor succeeds.
        frame, details, data = await asyncio.to_thread(self._render, scene, samples, index)
        self.frame, self.details, self.data = frame, details, data
        self.scene, self.samples, self.index = scene, samples, index
        self.error = None

    def delay(self):
        if self.index + 1 >= len(self.samples):
            return 0
        return (self.samples[self.index + 1]["timestamp"] - self.samples[self.index]["timestamp"]) / 1e6 / self.rate

    def state(self):
        sample = self.samples[self.index]
        elapsed = max(time.monotonic() - self.fps_start, 0.001)
        pair_index = min(self.index, len(self.samples) - 2)
        source_interval = ((self.samples[pair_index + 1]["timestamp"] - self.samples[pair_index]["timestamp"]) / 1e6
                           if len(self.samples) > 1 else 0)
        return {"scene": self.scene["name"], "scene_token": self.scene["token"],
                "sample_token": sample["token"], "sample_index": self.index + 1,
                "total_samples": len(self.samples), "timestamp": sample["timestamp"],
                "playing": self.playing, "rate": self.rate, "error": self.error,
                "playback_fps": round(self.advances / elapsed, 2) if self.playing else 0,
                "source_fps": round(1 / source_interval, 2) if source_interval > 0 else 0,
                "stage": 3 if self.encoder is not None else 2, "view": dict(self.view),
                "panels": PANELS if self.encoder is not None else {}, "video_size": [1600, 900],
                **self.details}

    async def change_view(self, action, value):
        view = dict(self.view)
        if action == "focus":
            if value not in FOCUS_MODES or (value == "FEATURE" and self.encoder is None):
                raise ValueError("Invalid/unavailable focus panel")
            view["focus"] = value
        elif action == "feature_camera":
            if value not in CAMERAS:
                raise ValueError("Invalid feature camera")
            view[action] = value
        elif action == "feature_mode":
            if value not in ("mean", "channel"):
                raise ValueError("Feature mode must be mean or channel")
            view[action] = value
        elif action == "feature_channel":
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 256:
                raise ValueError("Feature channel must be an integer from 0 to 255")
            view[action] = value
        if view == self.view:
            return
        start = time.monotonic()
        frame = await asyncio.to_thread(render, self.data, self.scene, self.samples[self.index],
                                        self.index, len(self.samples), view, self.details["encoder"])
        self.frame, self.view = frame, view
        self.details["view_render_ms"] = round((time.monotonic() - start) * 1000, 2)

    async def command(self, action, value=None):
        async with self.lock:
            if action in ("focus", "feature_camera", "feature_mode", "feature_channel"):
                await self.change_view(action, value)
            elif action == "play":
                if self.index == len(self.samples) - 1:
                    await self.load(self.scene, self.samples, 0)
                if not self.playing:
                    self.playing = True
                    self.fps_start, self.advances = time.monotonic(), 0
                    self.due = time.monotonic() + self.delay()
            elif action == "pause":
                self.playing = False
            elif action in ("next", "previous", "restart", "scene"):
                self.playing = False
                scene, samples = self.scene, self.samples
                if action == "scene":
                    scene, samples = self.source.scene_samples(value)
                    index = 0
                    LOG.info("Selected scene %s (%s)", scene["name"], scene["token"])
                elif action == "restart":
                    index = 0
                else:
                    index = max(0, min(len(samples) - 1, self.index + (1 if action == "next" else -1)))
                await self.load(scene, samples, index)
            elif action == "rate":
                rate = float(value)
                if rate not in (0.5, 1.0, 2.0):
                    raise ValueError("Rate must be 0.5, 1 or 2")
                self.rate = rate
                self.due = time.monotonic() + self.delay()
                self.fps_start, self.advances = time.monotonic(), 0
            else:
                raise ValueError(f"Unknown action: {action}")
            return self.state()

    async def run(self):
        while True:
            await asyncio.sleep(0.02)
            async with self.lock:
                if not self.playing or time.monotonic() < self.due:
                    continue
                try:
                    if self.index + 1 >= len(self.samples):
                        self.playing = False
                        continue
                    await self.load(self.scene, self.samples, self.index + 1)
                    self.advances += 1
                    self.due += self.delay()
                    if self.index == len(self.samples) - 1:
                        self.playing = False
                except Exception as exc:
                    LOG.exception("Sample playback failed")
                    self.playing, self.error = False, str(exc)

    async def start(self, app):
        scene, samples = self.source.scene_samples(self.initial_scene)
        LOG.info("Selected scene %s (%s)", scene["name"], scene["token"])
        await self.load(scene, samples, 0)
        self.task = asyncio.create_task(self.run())

    async def shutdown(self, app):
        self.playing = False
        # Let an in-flight sample read complete before shutting down its worker.
        async with self.lock:
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


class PlayerTrack(VideoStreamTrack):
    def __init__(self, peer_id, player):
        super().__init__()
        self.peer_id, self.player = peer_id, player
        self.player.viewers += 1
        self.started, self.next_frame = None, 0
        self.count, self.log_time = 0, time.monotonic()

    async def recv(self):
        if self.started is None:
            self.started = time.monotonic()
            self.log_time = self.started
            LOG.info("peer %s nuScenes stream start: 1600x900 @ 25Hz refresh", self.peer_id)
        now = time.monotonic() - self.started
        await asyncio.sleep(max(0, self.next_frame - now))
        now = time.monotonic() - self.started
        self.next_frame = now + 1 / 25
        video = VideoFrame.from_ndarray(self.player.frame, format="bgr24")
        video.pts, video.time_base = round(now * 90000), Fraction(1, 90000)
        self.count += 1
        if time.monotonic() - self.log_time >= 5:
            LOG.info("peer %s WebRTC output FPS=%.1f", self.peer_id, self.count / (time.monotonic() - self.log_time))
            self.count, self.log_time = 0, time.monotonic()
        return video

    def stop(self):
        if self.readyState != "ended":
            self.player.viewers -= 1
            if self.player.viewers == 0:
                self.player.playing = False
        super().stop()
