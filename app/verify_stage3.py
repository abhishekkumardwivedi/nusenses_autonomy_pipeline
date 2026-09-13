"""Real CUDA + mini dataset + loopback WebRTC checks; no image files written."""
import asyncio
import json
import os
import sys
from pathlib import Path
import time

import numpy as np
import torch
from aiohttp import ClientSession
from aiohttp.test_utils import TestServer
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.camera_encoder import CameraEncoder, preprocess, CAMERA_ORDER
from nuscenes_player.nuscenes_source import NuScenesSource, CAMERAS
from nuscenes_player.player import Player, PlayerTrack
from nuscenes_player.renderer import feature_activation, draw_bev
from webrtc_test.server import create_app, peers


async def verify_webrtc(player):
    server = TestServer(create_app(lambda key: PlayerTrack(key, player)))
    await server.start_server()
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    ready = asyncio.get_running_loop().create_future()
    @pc.on('track')
    def on_track(track):
        ready.set_result(track)
    try:
        async with ClientSession() as session:
            pc.addTransceiver('video', direction='recvonly')
            await pc.setLocalDescription(await pc.createOffer())
            async with session.post(server.make_url('/offer'), json={
                'sdp':pc.localDescription.sdp, 'type':pc.localDescription.type}) as response:
                assert response.status == 200
                answer = await response.json()
            await pc.setRemoteDescription(RTCSessionDescription(sdp=answer['sdp'], type=answer['type']))
            track = await asyncio.wait_for(ready, 20)
            first = None
            pts = []
            count = player.encoder.inference_count
            for i in range(60):
                if i == 20:
                    await player.command('next')
                frame = await asyncio.wait_for(track.recv(), 20)
                assert (frame.width, frame.height) == (1600,900)
                pts.append(frame.pts)
                if i == 10:
                    first = frame.to_ndarray(format='bgr24')
            assert not np.array_equal(first, frame.to_ndarray(format='bgr24'))
            assert all(b > a for a,b in zip(pts,pts[1:]))
            assert player.encoder.inference_count == count + 1
            async with session.post(server.make_url('/stop/' + answer['peer_id'])) as response:
                assert response.status == 204
            assert not peers
    finally:
        await pc.close()
        await server.close()
    print('PASS WebRTC SDP, 60 decoded 1600x900 frames, monotonic PTS, changed sample, cleanup', flush=True)


async def main():
    source = NuScenesSource(os.getenv('NUSCENES_DATAROOT','/workspace/autonomy/datasets/nuscenes'),
                            os.getenv('NUSCENES_VERSION','v1.0-mini'))
    calls = []
    original_load = source.load
    def record_load(sample):
        data = original_load(sample)
        calls.append(sample['token'])
        return data
    source.load = record_load
    encoder = CameraEncoder()
    assert not encoder.training and all(not module.training for module in encoder.modules())
    assert all(not p.requires_grad for p in encoder.parameters())
    assert torch.cuda.is_available()
    # Known BGR pixel proves RGB swap and ImageNet normalization independently.
    blue = {name:np.full((4,4,3), [255,0,0], dtype=np.uint8) for name in CAMERA_ORDER}
    prepared = preprocess(blue)
    assert list(prepared.shape) == [6,3,256,448]
    np.testing.assert_allclose(prepared[0,:,0,0].numpy(),
                               [(0-.485)/.229,(0-.456)/.224,(1-.406)/.225], rtol=1e-6)
    player = Player(source, source.scenes[0]['name'], encoder)
    await player.start(None)
    try:
        assert tuple(CAMERAS) == CAMERA_ORDER
        first = player.data['feature_tensor'].clone()
        assert list(first.shape) == [1,6,256,8,14]
        assert not first.requires_grad and first.grad_fn is None and torch.isfinite(first).all()
        assert all(not torch.equal(first[0,0],first[0,i]) for i in range(1,6))
        assert all(p.grad is None for p in encoder.parameters())
        count = encoder.inference_count
        for action,value in [('feature_mode','channel'),('feature_channel',17),('feature_camera','CAM_BACK'),
                              ('focus','CAM_FRONT'),('focus','LIDAR'),('focus','RADAR'),('focus','FEATURE'),('focus','overview')]:
            await player.command(action,value)
            assert player.frame.shape == (900,1600,3)
        assert encoder.inference_count == count
        for invalid in [-1,256,1.5,True,'1']:
            try:
                await player.command('feature_channel',invalid)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Accepted invalid channel {invalid}')
        await player.command('pause')
        await asyncio.sleep(.7)
        assert encoder.inference_count == count
        await player.command('previous') # Already sample 1: no change, no inference.
        assert encoder.inference_count == count
        await player.command('next')
        assert encoder.inference_count == count + 1
        assert not torch.equal(first,player.data['feature_tensor'])
        assert player.data['lidar'].shape[1] > 0 and player.data['radar'].shape[1] > 0
        for mode in ['mean','channel']:
            heatmap,low,high = feature_activation(player.data['features'],'CAM_FRONT',mode,17)
            assert heatmap.shape == (8,14,3) and np.isfinite([low,high]).all()
        # BEV focus must rasterize actual points at high resolution, with the
        # two sensor subsets different, not stretch a cached combined bitmap.
        lidar = np.zeros((900,1600,3),np.uint8)
        radar = lidar.copy()
        draw_bev(lidar,player.data,[0,48,1600,852],'LIDAR')
        draw_bev(radar,player.data,[0,48,1600,852],'RADAR')
        assert np.count_nonzero(lidar) > 10000 and not np.array_equal(lidar,radar)
        await player.command('focus','FEATURE')
        focused = player.frame.copy()
        await player.command('scene',source.scenes[1]['name'])
        assert player.view['focus'] == 'FEATURE' and not np.array_equal(focused,player.frame)
        count = encoder.inference_count
        index = player.index
        await player.command('rate',2)
        await player.command('play')
        deadline = time.monotonic() + 15
        while player.index < index + 3 and time.monotonic() < deadline:
            await asyncio.sleep(.05)
        await player.command('pause')
        assert player.index >= index + 3
        assert encoder.inference_count - count == player.index - index
        assert encoder.inference_count == len(calls)
        assert all(a != b for a,b in zip(calls,calls[1:]))
        await player.command('restart')
        assert player.index == 0 and not player.playing
        await player.command('feature_mode','mean')
        await player.command('focus','overview')
        await verify_webrtc(player)
        print('PASS CUDA, preprocessing, eval/no gradients, finite/different features, cached views, Pause, exact per-sample inference, focused scene changes', flush=True)
        print(json.dumps(player.details,indent=2),flush=True)
    finally:
        await player.shutdown(None)


asyncio.run(main())
