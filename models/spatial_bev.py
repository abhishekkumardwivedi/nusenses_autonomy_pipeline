"""Current-sample camera features lifted with sparse LiDAR depth; no history."""
import time

import numpy as np
import torch

from .camera_encoder import CAMERA_ORDER
from nuscenes_player.geometry import transform_points


def resized_intrinsic(intrinsic, width, height):
    """Stage 3 stretches the whole image to 448x256, with no crop/padding."""
    return np.diag([448 / width, 256 / height, 1.0]) @ intrinsic


def lift_camera(data, name):
    """One nearest positive camera-Z depth per 32x32 image region.

    The descriptor covers a region, not a metric point. Use its region centre
    as the representative ray (not the ResNet receptive-field centre). This
    coarse ray approximation introduces lateral error; no depth is invented
    for regions without LiDAR evidence. Sensor motion is compensated by the
    existing per-capture transforms; independently moving objects are not.
    """
    geo = data["camera_geometry"][name]
    height, width = data["cameras"][name].shape[:2]
    intrinsic = resized_intrinsic(geo["intrinsic"], width, height)
    camera = transform_points(data["lidar"], np.linalg.inv(geo["to_ego"]))
    valid = np.isfinite(camera).all(axis=0) & (camera[2] > 1.0)
    camera = camera[:, valid]
    uv = intrinsic @ camera
    uv = uv[:2] / uv[2:3]
    visible = (uv[0] >= 0) & (uv[0] < 448) & (uv[1] >= 0) & (uv[1] < 256)
    uv, depths = uv[:, visible], camera[2, visible]
    cells = (uv[1] // 32).astype(np.int64) * 14 + (uv[0] // 32).astype(np.int64)
    sparse = np.full(112, np.inf)
    np.minimum.at(sparse, cells, depths)  # nearest surface in each coarse region
    selected = np.flatnonzero(np.isfinite(sparse))
    pixels = np.vstack(((selected % 14 + .5) * 32,
                        (selected // 14 + .5) * 32, np.ones(len(selected))))
    lifted = (np.linalg.inv(intrinsic) @ pixels) * sparse[selected]
    ego = transform_points(lifted, geo["to_ego"])
    return selected, ego, int(visible.sum())


class SpatialBEV:
    """Average camera descriptors in 0.5m ego bins; cache only current output."""
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.calls = 0

    @torch.inference_mode()
    def project(self, features, data):
        if tuple(features.shape) != (1, 6, 256, 8, 14):
            raise ValueError(f"Expected [1,6,256,8,14], got {tuple(features.shape)}")
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        features = features.to(self.device)
        sums = torch.zeros((256, 200 * 200), device=self.device)
        counts = torch.zeros(200 * 200, device=self.device)
        camera_counts, geometry = [], {}
        for index, name in enumerate(CAMERA_ORDER):  # six cameras, no per-point loops
            cells, ego, visible = lift_camera(data, name)
            valid = np.isfinite(ego).all(axis=0) & (ego[0] >= -50) & (ego[0] < 50) & (ego[1] >= -50) & (ego[1] < 50)
            cells, ego = cells[valid], ego[:, valid]
            # Tensor rows increase with X, columns with Y; renderer flips BOTH
            # axes so forward is up and left is left, matching the sensor BEV.
            xy = np.floor((ego[:2] + 50) / .5).astype(np.int64)
            bins = torch.as_tensor(xy[0] * 200 + xy[1], device=self.device)
            selected = torch.as_tensor(cells, device=self.device)
            descriptors = features[0, index].reshape(256, 112)[:, selected]
            sums.index_add_(1, bins, descriptors)
            per_camera = torch.bincount(bins, minlength=40000).float()
            counts += per_camera
            camera_counts.append(per_camera.reshape(200, 200).cpu())
            geometry[name] = {"visible_depth_points": visible, "lifted_cells_in_range": len(cells),
                              "median_xy": np.median(ego[:2], axis=1).tolist() if len(cells) else None}
        bev = (sums / counts.clamp_min(1)).reshape(1, 256, 200, 200)
        if not torch.isfinite(bev).all().item():
            raise ValueError("Non-finite spatial BEV")
        result = {"bev_tensor": bev.cpu(), "bev_counts": counts.reshape(200, 200).cpu(),
                  "bev_camera_counts": torch.stack(camera_counts)}
        torch.cuda.synchronize(self.device)
        self.calls += 1
        result["spatial"] = {"shape": list(bev.shape), "camera_to_bev_ms": round((time.perf_counter() - start) * 1000, 2),
                             "gpu_allocated_mb": round(torch.cuda.memory_allocated(self.device) / 2**20, 2),
                             "occupied_cells": int((counts > 0).sum().item()), "calls": self.calls,
                             "cameras": geometry}
        return result
