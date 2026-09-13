"""Compose camera images and geometric BEV entirely in RAM (1600x900)."""
import cv2
import numpy as np
from .nuscenes_source import CAMERAS

WIDTH, HEIGHT = 1600, 900


def text(frame, label, x, y, scale=0.55, color=(230, 235, 240)):
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def bev_pixels(points, center, scale):
    # Vehicle forward (+x) is UP; left (+y) is LEFT. Equal x/y meter scale.
    return np.column_stack((center[0] - points[1] * scale, center[1] - points[0] * scale))


def render(data, scene, sample, index, total):
    frame = np.full((HEIGHT, WIDTH, 3), (25, 21, 18), np.uint8)
    text(frame, f"{scene['name']} | sample {index + 1}/{total} | timestamp {sample['timestamp']} us", 15, 23, 0.65)
    for slot, channel in enumerate(CAMERAS):
        x, y = (slot % 3) * 533, 34 + (slot // 3) * 266
        image = data["cameras"][channel]
        scale = min(531 / image.shape[1], 238 / image.shape[0])
        resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
        dx = x + (531 - resized.shape[1]) // 2
        dy = y + 26 + (238 - resized.shape[0]) // 2
        frame[dy:dy + resized.shape[0], dx:dx + resized.shape[1]] = resized
        dt = data["sensors"][channel]["offset_ms"]
        text(frame, f"{channel} | dt {dt:+.1f} ms", x + 10, y + 18)

    # Square 300px BEV centered beneath cameras, +/-50m in each axis.
    x0, y0, side = 650, 585, 300
    center, ppm = (x0 + side // 2, y0 + side // 2), 3.0
    frame[y0:y0 + side, x0:x0 + side] = (14, 14, 14)
    for offset in range(0, side + 1, 30):
        cv2.line(frame, (x0 + offset, y0), (x0 + offset, y0 + side), (45, 45, 45))
        cv2.line(frame, (x0, y0 + offset), (x0 + side, y0 + offset), (45, 45, 45))
    for name, color, radius in (("lidar", (210, 180, 65), 0), ("radar", (60, 155, 255), 2)):
        xyz = data[name]
        mask = np.isfinite(xyz).all(axis=0) & (np.abs(xyz[0]) < 49.5) & (np.abs(xyz[1]) < 49.5)
        pixels = np.rint(bev_pixels(xyz[:, mask], center, ppm)).astype(int)
        if radius:
            for px, py in pixels:
                cv2.circle(frame, (int(px), int(py)), radius, color, -1)
        elif len(pixels):
            frame[pixels[:, 1], pixels[:, 0]] = color
    cv2.rectangle(frame, (center[0] - 3, center[1] - 7), (center[0] + 3, center[1] + 7), (100, 240, 100), -1)
    cv2.arrowedLine(frame, center, (center[0], center[1] - 27), (100, 240, 100), 2)
    text(frame, "SENSOR BEV | +x forward / +y left", 645, 579, 0.47)
    text(frame, "LiDAR (cyan)", 990, 630, color=(210, 180, 65))
    text(frame, f"{data['lidar'].shape[1]:,} points", 990, 655)
    text(frame, "Radar (orange, devkit filters)", 990, 695, color=(60, 155, 255))
    text(frame, f"{data['radar'].shape[1]:,} points", 990, 720)
    text(frame, "Grid: 10m | Range: +/-50m", 230, 675)
    text(frame, "Reference: LIDAR_TOP ego pose", 230, 705)
    text(frame, "Recorded keyframes; no inference", 230, 735)
    text(frame, f"Ego global xyz: {np.round(data['ego_pose']['translation'], 2).tolist()}", 990, 770, 0.45)
    return frame
