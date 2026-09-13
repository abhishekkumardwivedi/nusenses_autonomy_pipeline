"""Run on the real dataset: python app/verify.py (no output images)."""
import asyncio
import os
import sys
import time
from pathlib import Path
import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nuscenes_player.geometry import pose_matrix, sensor_to_ego, transform_points
from nuscenes_player.nuscenes_source import NuScenesSource, CAMERAS
from nuscenes_player.player import Player
from nuscenes_player.renderer import render, bev_pixels


async def main():
    # Known geometry independent of the implementation: 90-degree sensor yaw,
    # sensor translated 1m forward, capture ego 10m ahead of reference ego.
    identity = {"rotation": [1, 0, 0, 0], "translation": [0, 0, 0]}
    calibration = {"rotation": [2**-0.5, 0, 0, 2**-0.5], "translation": [1, 0, 0]}
    capture = {"rotation": [1, 0, 0, 0], "translation": [10, 0, 0]}
    actual = transform_points(np.array([[2], [0], [0]]), sensor_to_ego(calibration, capture, identity))
    np.testing.assert_allclose(actual[:, 0], [11, 2, 0], atol=1e-10)
    np.testing.assert_allclose(bev_pixels(np.array([[1, 0], [0, 1], [0, 0]]), (100, 100), 3), [[100, 97], [97, 100]])
    source = NuScenesSource(os.getenv('NUSCENES_DATAROOT', '/workspace/autonomy/datasets/nuscenes'),
                            os.getenv('NUSCENES_VERSION', 'v1.0-mini'))
    scene, samples = source.scene_samples(source.scenes[0]['token'])
    data = source.load(samples[0])
    following = source.load(samples[1])
    assert len(data['cameras']) == 6 and data['lidar'].shape[1] > 0 and data['radar'].shape[1] > 0
    for channel in CAMERAS:
        assert not np.array_equal(data['cameras'][channel], following['cameras'][channel]), channel
    assert not np.array_equal(data['lidar'], following['lidar'])
    assert not np.array_equal(data['radar'], following['radar'])
    for channel, record in data['sensors'].items():
        assert record['token'] == samples[0]['data'][channel]
        assert source.nusc.get('sample_data', record['token'])['sample_token'] == samples[0]['token']
    # Compare every radar against the devkit's independent multisweep transform,
    # with one sweep. Devkit returns reference LiDAR coordinates; convert to ego.
    lidar_record = source.nusc.get('sample_data', samples[0]['data']['LIDAR_TOP'])
    reference_calibration = source.nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
    offset = 0
    for channel, info in data['sensors'].items():
        if not channel.startswith('RADAR_'):
            continue
        cloud, _ = RadarPointCloud.from_file_multisweep(source.nusc, samples[0], channel, 'LIDAR_TOP', nsweeps=1, min_distance=0)
        expected = transform_points(cloud.points[:3], pose_matrix(reference_calibration))
        np.testing.assert_allclose(data['radar'][:, offset:offset + info['points']], expected, atol=1e-6)
        offset += info['points']
    frame = render(data, scene, samples[0], 0, len(samples))
    assert frame.shape == (900, 1600, 3)
    print('PASS six changing cameras, LiDAR/radar updates, sample linkage, BEV orientation, radar transforms vs devkit', flush=True)
    player = Player(source, scene['name'])
    await player.start(None)
    try:
        await player.command('next')
        assert player.index == 1
        await player.command('previous')
        assert player.index == 0
        await player.command('scene', source.scenes[1]['token'])
        assert player.scene['token'] == source.scenes[1]['token']
        await player.command('play')
        await asyncio.sleep(2.2)
        await player.command('pause')
        index = player.index
        assert index > 0
        await asyncio.sleep(0.7)
        assert player.index == index
        await player.command('restart')
        assert player.index == 0 and not player.playing
        await player.command('previous')
        assert player.index == 0
        await player.load(player.scene, player.samples, len(player.samples) - 2)
        await player.command('play')
        start = time.monotonic()
        while player.playing and time.monotonic() - start < 10:
            await asyncio.sleep(0.1)
        assert not player.playing and player.index == len(player.samples) - 1
        await player.command('play')
        assert player.index == 0
        await player.command('pause')
        print('PASS scene selection, Play/Pause/Next/Previous/Restart, first/last boundaries and end-of-scene', flush=True)
    finally:
        await player.shutdown(None)


asyncio.run(main())
