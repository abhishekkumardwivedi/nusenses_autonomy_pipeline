"""One shared timeline. Sensor I/O/rendering run off the asyncio event loop."""
import asyncio
import logging
import time
from fractions import Fraction
from aiortc import VideoStreamTrack
from av import VideoFrame
from .renderer import render

LOG = logging.getLogger("player")


class Player:
    def __init__(self, source, scene):
        self.source, self.initial_scene = source, scene
        self.lock = asyncio.Lock()
        self.playing, self.rate, self.error = False, 1.0, None
        self.frame, self.index, self.viewers = None, 0, 0
        self.scene, self.samples, self.details = None, [], {}
        self.due = 0
        self.fps_start, self.advances = time.monotonic(), 0
        self.task = None

    def _render(self, scene, samples, index):
        start = time.monotonic()
        data = self.source.load(samples[index])
        loaded = time.monotonic()
        frame = render(data, scene, samples[index], index, len(samples))
        rendered = time.monotonic()
        ms = (time.monotonic() - start) * 1000
        details = {"sensors": data["sensors"], "ego_pose": data["ego_pose"],
                   "lidar_points": data["lidar"].shape[1], "radar_points": data["radar"].shape[1],
                   "load_ms": round((loaded - start) * 1000, 1),
                   "render_ms": round((rendered - loaded) * 1000, 1), "load_render_ms": round(ms, 1)}
        LOG.info("scene=%s sample=%s/%s timestamp=%s load=%.1fms render=%.1fms total=%.1fms",
                 scene["name"], index + 1, len(samples), samples[index]["timestamp"],
                 details["load_ms"], details["render_ms"], ms)
        return frame, details

    async def load(self, scene, samples, index):
        # Publish frame and metadata together, only after every sensor succeeds.
        frame, details = await asyncio.to_thread(self._render, scene, samples, index)
        self.frame, self.details = frame, details
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
                **self.details}

    async def command(self, action, value=None):
        async with self.lock:
            if action == "play":
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
