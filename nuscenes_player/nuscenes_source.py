"""Read only sensor records linked by sample['data']; never scan image folders."""
from pathlib import Path
import logging
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from .geometry import sensor_to_ego, transform_points

LOG = logging.getLogger("nuscenes")
CAMERAS = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
           "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")


class NuScenesSource:
    def __init__(self, dataroot, version="v1.0-mini"):
        self.root = Path(dataroot)
        if not (self.root / version / "scene.json").is_file():
            raise FileNotFoundError(f"Missing {self.root / version / 'scene.json'}; set NUSCENES_DATAROOT / NUSCENES_VERSION")
        self.nusc = NuScenes(version=version, dataroot=str(self.root), verbose=False)
        self.scenes = [{"name": s["name"], "token": s["token"],
                        "samples": s["nbr_samples"]} for s in self.nusc.scene]

    def scene_samples(self, name_or_token):
        scene = next((s for s in self.nusc.scene if name_or_token in (s["name"], s["token"])), None)
        if scene is None:
            raise ValueError(f"Unknown scene: {name_or_token}")
        samples = []
        token = scene["first_sample_token"]
        while token:
            sample = self.nusc.get("sample", token)
            samples.append(sample)
            token = sample["next"]
        return scene, samples

    def load(self, sample):
        records = {channel: self.nusc.get("sample_data", token)
                   for channel, token in sample["data"].items()
                   if channel in CAMERAS or channel == "LIDAR_TOP" or channel.startswith("RADAR_")}
        for channel in (*CAMERAS, "LIDAR_TOP"):
            if channel not in records:
                raise ValueError(f"Sample {sample['token']} has no {channel}")
        reference = self.nusc.get("ego_pose", records["LIDAR_TOP"]["ego_pose_token"])
        cameras, radars, lidar = {}, [], None
        info = {}
        for channel, record in records.items():
            path = self.root / record["filename"]
            info[channel] = {"token": record["token"], "timestamp": record["timestamp"],
                             "offset_ms": (record["timestamp"] - sample["timestamp"]) / 1000}
            if channel in CAMERAS:
                image = cv2.imread(str(path))
                if image is None:
                    raise FileNotFoundError(f"Cannot read camera: {path}")
                cameras[channel] = image
            else:
                cloud = (LidarPointCloud if channel == "LIDAR_TOP" else RadarPointCloud).from_file(str(path))
                calibration = self.nusc.get("calibrated_sensor", record["calibrated_sensor_token"])
                capture_pose = self.nusc.get("ego_pose", record["ego_pose_token"])
                xyz = transform_points(cloud.points[:3], sensor_to_ego(calibration, capture_pose, reference))
                info[channel]["points"] = xyz.shape[1]
                if channel == "LIDAR_TOP":
                    lidar = xyz
                else:
                    radars.append(xyz)
        radar = np.concatenate(radars, axis=1) if radars else np.empty((3, 0))
        LOG.info("sample %s timestamp=%s sensors=%s LiDAR=%s radar=%s",
                 sample["token"], sample["timestamp"],
                 ", ".join(f"{c}:{v['token']} dt={v['offset_ms']:+.1f}ms" for c, v in info.items()),
                 lidar.shape[1], radar.shape[1])
        return {"cameras": cameras, "lidar": lidar, "radar": radar,
                "sensors": info, "ego_pose": reference}
