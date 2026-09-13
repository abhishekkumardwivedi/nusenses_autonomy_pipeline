"""Sensor -> capture ego -> global -> reference ego. No rendering here."""
import numpy as np
from pyquaternion import Quaternion


def pose_matrix(record):
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(record["rotation"]).rotation_matrix
    matrix[:3, 3] = record["translation"]
    return matrix


def sensor_to_ego(calibration, capture_pose, reference_pose):
    """Reference is the LIDAR_TOP ego pose for the current sample.

    nuScenes quaternions are [w,x,y,z]. Each sensor uses its OWN capture
    ego pose, including radar measurements a few milliseconds apart.
    """
    return np.linalg.inv(pose_matrix(reference_pose)) @ pose_matrix(capture_pose) @ pose_matrix(calibration)


def transform_points(xyz, matrix):
    """xyz: (3,N). Ego axes: x forward, y left, z up. Meters."""
    return matrix[:3, :3] @ xyz + matrix[:3, 3:4]
