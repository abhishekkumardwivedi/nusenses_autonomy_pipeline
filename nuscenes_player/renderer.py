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


def render_stage2(data, scene, sample, index, total):
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


# Rectangles are also returned to the browser for letterbox-aware hit testing.
PANELS = {name: [(i % 3) * 533, 48 + (i // 3) * 244, 531, 240]
          for i, name in enumerate(CAMERAS)}
PANELS.update({"BEV": [0, 542, 530, 350], "FEATURE": [540, 542, 580, 350]})
SPATIAL_PANELS = {**PANELS, "BEV": [0, 542, 530, 350],
                  "SPATIAL": [533, 542, 534, 350], "FEATURE": [1070, 542, 530, 350]}
FOCUS_MODES = ("overview", *CAMERAS, "BEV", "LIDAR", "RADAR", "FEATURE", "SPATIAL")


def fit_image(frame, image, rect, nearest=False):
    x, y, w, h = rect
    scale = min(w / image.shape[1], h / image.shape[0])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
    x += (w - size[0]) // 2
    y += (h - size[1]) // 2
    frame[y:y + size[1], x:x + size[0]] = resized


def camera_panel(frame, data, name, rect):
    x, y, w, h = rect
    text(frame, f"RAW CAMERA | {name} | dt {data['sensors'][name]['offset_ms']:+.1f} ms", x + 10, y + 20)
    fit_image(frame, data["cameras"][name], [x, y + 30, w, h - 32])


def draw_bev(frame, data, rect, mode="BEV"):
    """Rasterize original ego-frame points at the requested panel resolution."""
    x, y, w, h = rect
    text(frame, f"{mode} | Geometric projection - no ML", x + 10, y + 20)
    side = min(w - 24, h - 60)
    x0, y0 = x + (w - side) // 2, y + 34
    center = (x0 + side // 2, y0 + side // 2)
    ppm = side / 100
    for meters in range(-50, 51, 10):
        offset = round(meters * ppm)
        cv2.line(frame, (center[0] + offset, y0), (center[0] + offset, y0 + side), (45, 48, 50))
        cv2.line(frame, (x0, center[1] + offset), (x0 + side, center[1] + offset), (45, 48, 50))
    for name, color in (("lidar", (210, 180, 65)), ("radar", (60, 155, 255))):
        if (mode == "LIDAR" and name != "lidar") or (mode == "RADAR" and name != "radar"):
            continue
        points = data[name]
        mask = np.isfinite(points).all(axis=0) & (np.abs(points[0]) < 49.5) & (np.abs(points[1]) < 49.5)
        pixels = np.rint(bev_pixels(points[:, mask], center, ppm)).astype(int)
        if name == "lidar" and len(pixels):
            frame[pixels[:, 1], pixels[:, 0]] = color
        else:
            for px, py in pixels:
                cv2.circle(frame, (int(px), int(py)), 3 if side > 500 else 2, color, -1)
    half_w, half_l = max(3, round(ppm)), max(6, round(2.3 * ppm))
    cv2.rectangle(frame, (center[0] - half_w, center[1] - half_l),
                  (center[0] + half_w, center[1] + half_l), (100, 240, 100), 1)
    cv2.arrowedLine(frame, center, (center[0], center[1] - max(20, round(5 * ppm))), (100, 240, 100), 2)
    text(frame, "10m grid | +/-50m | x forward/up, y left | cyan LiDAR, orange radar", x + 8, y + h - 7, 0.42)


def feature_activation(features, camera, mode, channel):
    if camera not in CAMERAS or mode not in ("mean", "channel"):
        raise ValueError("Invalid feature camera or mode")
    if not isinstance(channel, int) or isinstance(channel, bool) or not 0 <= channel < 256:
        raise ValueError("Feature channel must be an integer from 0 to 255")
    # The cached CPU ndarray shares storage with the cached torch tensor.
    f = features[0, CAMERAS.index(camera)]
    activation = np.abs(f).mean(axis=0) if mode == "mean" else f[channel]
    low, high = float(activation.min()), float(activation.max())
    normalized = np.zeros_like(activation) if high == low else (activation - low) / (high - low)
    heatmap = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return heatmap, low, high


def feature_panel(frame, data, view, rect):
    x, y, w, h = rect
    camera, mode, channel = view["feature_camera"], view["feature_mode"], view["feature_channel"]
    text(frame, f"CAMERA ENCODER | {camera}", x + 10, y + 20)
    label = "mean(abs(channels))" if mode == "mean" else f"channel {channel}"
    heatmap, low, high = feature_activation(data["features"], camera, mode, channel)
    text(frame, f"ResNet-50 + fixed reduction | {label} | 8x14", x + 10, y + 43, 0.48)
    fit_image(frame, heatmap, [x + 8, y + 53, w - 16, h - 80], nearest=True)
    text(frame, f"Per-map min/max: {low:.3f} / {high:.3f} | not semantic classes", x + 10, y + h - 8, 0.43)


def spatial_panel(frame, data, rect):
    x, y, w, h = rect
    text(frame, "CAMERA SPATIAL BEV", x + 10, y + 20)
    text(frame, "LiDAR-assisted depth for geometry validation", x + 10, y + 40, .45)
    evidence = data["bev_counts"].numpy() > 0
    activation = data["bev_tensor"].numpy()[0]
    activation = np.abs(activation).mean(axis=0)
    normalized = np.zeros_like(activation)
    if evidence.any():
        values = activation[evidence]
        low, high = values.min(), values.max()
        normalized[evidence] = (values - low) / (high - low) if high > low else 1
    heatmap = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap[~evidence] = (36, 36, 36)  # unknown, never a free-space prediction
    heatmap = np.ascontiguousarray(heatmap[::-1, ::-1])
    side = min(w - 24, h - 88)
    left, top = x + (w - side) // 2, y + 51
    fit_image(frame, heatmap, [left, top, side, side], nearest=True)
    for i in range(0, 201, 20):
        offset = min(side - 1, round(i * side / 200))
        cv2.line(frame, (left + offset, top), (left + offset, top + side - 1), (58, 58, 58))
        cv2.line(frame, (left, top + offset), (left + side - 1, top + offset), (58, 58, 58))
    center = (left + side // 2, top + side // 2)
    cv2.arrowedLine(frame, center, (center[0], center[1] - max(12, side // 20)), (100, 240, 100), 1)
    text(frame, "+X up / +Y left | +/-50m | 0.5m cells", x + 10, y + h - 22, .43)
    text(frame, "Gray = no evidence, not free space | mean abs features", x + 10, y + h - 5, .40)


def render(data, scene, sample, index, total, view=None, metrics=None):
    view = view or {"focus": "overview", "feature_camera": "CAM_FRONT", "feature_mode": "mean", "feature_channel": 0}
    focus = view["focus"]
    if focus == "overview" and "features" not in data:
        return render_stage2(data, scene, sample, index, total)
    frame = np.full((HEIGHT, WIDTH, 3), (25, 21, 18), np.uint8)
    stage = 4 if "bev_tensor" in data else (3 if "features" in data else 2)
    text(frame, f"Stage {stage} | {scene['name']} | sample {index + 1}/{total} | {sample['timestamp']} us | {focus}", 15, 27, 0.65)
    if focus in CAMERAS:
        camera_panel(frame, data, focus, [0, 48, WIDTH, HEIGHT - 48])
    elif focus in ("BEV", "LIDAR", "RADAR"):
        draw_bev(frame, data, [0, 48, WIDTH, HEIGHT - 48], focus)
    elif focus == "FEATURE":
        feature_panel(frame, data, view, [0, 48, WIDTH, HEIGHT - 48])
    elif focus == "SPATIAL":
        spatial_panel(frame, data, [0, 48, WIDTH, HEIGHT - 48])
    else:
        for camera in CAMERAS:
            camera_panel(frame, data, camera, PANELS[camera])
        panels = SPATIAL_PANELS if stage == 4 else PANELS
        draw_bev(frame, data, panels["BEV"])
        feature_panel(frame, data, view, panels["FEATURE"])
        if stage == 4:
            spatial_panel(frame, data, panels["SPATIAL"])
            return frame
        info = metrics or {}
        lines = ["AI inference: Camera Encoder ENABLED", "ImageNet ResNet-50 (shared 6-camera batch)",
                 "Backbone 2048 -> fixed 1x1 mean -> 256", "Input [1,6,3,256,448]", "Features [1,6,256,8,14]",
                 f"Preprocess {info.get('preprocess_ms', 0):.1f} ms | Encoder {info.get('camera_encoder_ms', 0):.1f} ms",
                 f"LiDAR {data['lidar'].shape[1]} | Radar {data['radar'].shape[1]}",
                 "Sensor BEV: raw geometry, no ML", "Features: learned representation, not classes", "Double click a panel to inspect"]
        for i, label in enumerate(lines):
            text(frame, label, 1135, 568 + i * 29, 0.47)
    return frame
