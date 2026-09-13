"""Small real-data geometry/CUDA/WebRTC check; no generated image files."""
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.camera_encoder import CameraEncoder, CAMERA_ORDER
from models.spatial_bev import SpatialBEV, lift_camera, resized_intrinsic
from nuscenes_player.nuscenes_source import NuScenesSource
from nuscenes_player.player import Player
from nuscenes_player.geometry import transform_points
from app.verify_stage3 import verify_webrtc


async def main():
    source = NuScenesSource(os.environ['NUSCENES_DATAROOT'])
    encoder, projector = CameraEncoder(), SpatialBEV()
    player = Player(source, source.scenes[0]['name'], encoder, projector)
    await player.start(None)
    try:
        first = player.data['bev_tensor'].clone()
        assert list(player.data['feature_tensor'].shape) == [1,6,256,8,14]
        assert list(first.shape) == [1,256,200,200] and torch.isfinite(first).all()
        counts = player.data['bev_counts']
        assert counts.sum() > 0 and torch.equal(counts, player.data['bev_camera_counts'].sum(0))
        assert torch.count_nonzero(first[0,:,counts == 0]) == 0
        # Compare the exact projected sparse LiDAR pixels/depths with the
        # nuScenes devkit's independent sensor->global->camera path.
        sample = player.samples[player.index]
        for name in CAMERA_ORDER:
            uv, depths, _ = source.nusc.explorer.map_pointcloud_to_image(
                sample['data']['LIDAR_TOP'], sample['data'][name])
            assert len(depths) > 0
            geo = player.data['camera_geometry'][name]
            sd = source.nusc.get('sample_data', sample['data'][name])
            cam = transform_points(player.data['lidar'], np.linalg.inv(geo['to_ego']))
            valid = cam[2] > 1
            cam = cam[:,valid]
            pixels = geo['intrinsic'] @ cam
            pixels = pixels[:2] / pixels[2:3]
            mask = (pixels[0] > 1) & (pixels[0] < sd['width']-1) & (pixels[1] > 1) & (pixels[1] < sd['height']-1)
            # Devkit mutates a float32 point cloud through large global
            # translations; our composed float64 transform avoids that loss.
            np.testing.assert_allclose(pixels[:,mask], uv[:2], atol=.05)
            np.testing.assert_allclose(cam[2,mask], depths, atol=.002)
            k = resized_intrinsic(geo['intrinsic'], sd['width'], sd['height'])
            scaled = k @ cam[:,mask]
            np.testing.assert_allclose(scaled[:2]/scaled[2:3], uv[:2]*np.array([[448/sd['width']],[256/sd['height']]]), atol=.02)
            cells, ego, visible = lift_camera(player.data, name)
            assert visible > 0 and len(cells) > 0
            axis, sign = (0,-1) if name == 'CAM_BACK' else ((0,1) if name == 'CAM_FRONT' else (1,1 if 'LEFT' in name else -1))
            fraction = float(np.mean(ego[axis] * sign > 0))
            assert fraction > .6, (name, 'mirrored/wrong orientation', fraction)
            print(f'{name}: visible depth={visible}, cells={len(cells)}, expected half-plane={fraction:.3f}', flush=True)
        calls = (encoder.inference_count, projector.calls)
        for focus in [*CAMERA_ORDER, 'BEV', 'SPATIAL', 'FEATURE', 'overview']:
            await player.command('focus', focus)
            assert player.frame.shape == (900,1600,3)
        await player.command('pause')
        await asyncio.sleep(.6)
        assert (encoder.inference_count, projector.calls) == calls
        await player.command('next')
        assert not torch.equal(first, player.data['bev_tensor'])
        assert (encoder.inference_count, projector.calls) == (calls[0]+1,calls[1]+1)
        await player.command('focus','SPATIAL')
        await verify_webrtc(player)
        assert projector.calls == encoder.inference_count
        print('PASS shapes, finite output, depth vs devkit, resize, all camera directions, cached views, next sample and WebRTC', flush=True)
        print(json.dumps(player.details['spatial'], indent=2), flush=True)
    finally:
        await player.shutdown(None)


if __name__ == '__main__':
    asyncio.run(main())
