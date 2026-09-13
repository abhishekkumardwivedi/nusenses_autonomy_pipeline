"""Stage 2/3 entry point. Shared sensor timeline and original WebRTC transport."""
import argparse
import logging
import os
import sys
from pathlib import Path
from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from webrtc_test.server import create_app
from nuscenes_player.nuscenes_source import NuScenesSource
from nuscenes_player.player import Player, PlayerTrack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    logging.info("Loading nuScenes metadata and the initial sample...")
    source = NuScenesSource(os.getenv("NUSCENES_DATAROOT", "/workspace/autonomy/datasets/nuscenes"),
                            os.getenv("NUSCENES_VERSION", "v1.0-mini"))
    encoder = None
    if args.stage == 3:
        from models.camera_encoder import CameraEncoder
        encoder = CameraEncoder()
    player = Player(source, os.getenv("NUSCENES_SCENE", source.scenes[0]["name"]), encoder)
    app = create_app(lambda peer_id: PlayerTrack(peer_id, player), Path(__file__).with_name("index.html"))

    async def scenes(request):
        return web.json_response(source.scenes)

    async def state(request):
        return web.json_response(player.state(), headers={"Cache-Control": "no-store"})

    async def control(request):
        try:
            body = await request.json()
            result = await player.command(body["action"], body.get("value"))
            return web.json_response(result)
        except (ValueError, KeyError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            logging.exception("Playback control failed")
            player.playing, player.error = False, str(exc)
            return web.json_response({"error": str(exc)}, status=500)

    async def player_script(request):
        return web.FileResponse(Path(__file__).with_name("player.js"), headers={"Cache-Control": "no-store"})

    app.add_routes([web.get("/scenes", scenes), web.get("/state", state), web.post("/control", control),
                    web.get("/player.js", player_script)])
    app.on_startup.append(player.start)
    app.on_shutdown.append(player.shutdown)
    logging.info("nuScenes player: %s (%s), binding 0.0.0.0:8080", source.root, source.nusc.version)
    web.run_app(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
