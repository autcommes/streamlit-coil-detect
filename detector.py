"""核心算法：HSV 筛选 + Canny + 形态学 + 面积过滤。

为部署到 Streamlit Cloud 而独立成一个项目。本文件不依赖项目根目录的 run.py。
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _default_font_path() -> str:
    """跨平台寻找中文字体（云端 Linux 通常需通过 packages.txt 安装 fonts-noto-cjk）。"""
    candidates = {
        "Windows": [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyh.ttf",
            "C:/Windows/Fonts/simhei.ttf",
        ],
        "Darwin": [
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Songti.ttc",
        ],
        "Linux": [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ],
    }
    for p in candidates.get(platform.system(), []):
        if os.path.exists(p):
            return p
    return ""


FONT_PATH = os.environ.get("COIL_FONT_PATH") or _default_font_path()
GREEN = (0, 255, 0)
CYAN = (0, 200, 200)

# 自动降采样阈值：最长边超过此值时先 cv2.resize 再处理。
# 目的：1) 让 Streamlit Cloud 1GB 内存扛得住大图；
#       2) 让默认参数（按 ~2K 图调过）在 4K/7K 输入下也能直接用。
MAX_PROCESS_DIM = int(os.environ.get("COIL_MAX_DIM", "2000"))


@dataclass
class Params:
    board_h_min: int = 8
    board_h_max: int = 30
    board_s_min: int = 50
    board_s_max: int = 255
    board_v_min: int = 50
    board_v_max: int = 255
    coil_h_min: int = 10
    coil_h_max: int = 28
    coil_s_min: int = 120
    coil_s_max: int = 255
    coil_v_min: int = 90
    coil_v_max: int = 255
    canny_lo: int = 30
    canny_hi: int = 80
    min_area: int = 30
    max_area: int = 6000
    open_ksize: int = 3
    close_ksize: int = 11
    close_iters: int = 3


DEFAULTS = Params()


def put_cn_text(img_bgr, text, pos, font_size=24, color=(0, 255, 0)):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size) if FONT_PATH else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def detect_boards(mask_raw, h_img):
    row_sum = np.sum(mask_raw // 255, axis=1)
    nz_rows = np.where(row_sum >= 10)[0]
    if len(nz_rows) == 0:
        return [], 0, 0, 0, 0
    mid_gaps = np.where(
        (row_sum < 10)
        & (np.arange(h_img) > 50)
        & (np.arange(h_img) < h_img - 50)
    )[0]
    gap_y = int(mid_gaps[len(mid_gaps) // 2]) if len(mid_gaps) else h_img // 2

    b1_rows = nz_rows[nz_rows < gap_y]
    b2_rows = nz_rows[nz_rows > gap_y]
    if len(b1_rows) == 0 or len(b2_rows) == 0:
        b1y1, b1y2 = int(nz_rows[0]), int(nz_rows[-1])
        b2y1, b2y2 = b1y1, b1y2
    else:
        b1y1, b1y2 = int(b1_rows[0]), int(b1_rows[-1])
        b2y1, b2y2 = int(b2_rows[0]), int(b2_rows[-1])

    col_sum = np.sum(mask_raw // 255, axis=0)
    nz_cols = np.where(col_sum > 5)[0]
    if len(nz_cols) == 0:
        return [], 0, 0, 0, 0
    bx1, bx2 = int(nz_cols[0]), int(nz_cols[-1])

    return [
        (bx1, b1y1, bx2 - bx1, b1y2 - b1y1),
        (bx1, b2y1, bx2 - bx1, b2y2 - b2y1),
    ], bx1, bx2, b1y1, b2y2


def process_image(img_bgr: np.ndarray, p: Params = DEFAULTS) -> Tuple[np.ndarray, dict]:
    """输入 BGR ndarray + 参数，输出 (结果图BGR, 元信息)。

    大图（最长边 > MAX_PROCESS_DIM）会被自动等比降采样到处理尺寸再做检测，
    保证 Streamlit Cloud 不 OOM、且参数表跨分辨率通用。
    返回的可视化结果就是处理尺寸（清晰、易下载），但 info 里的坐标/尺寸
    都已还原到原图坐标系，便于对应到原图位置。
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr, {"contour_count": 0, "center_x": 0, "center_y": 0,
                         "width": 0, "height": 0, "error": "空图像"}

    h_orig, w_orig = img_bgr.shape[:2]
    longest = max(h_orig, w_orig)
    if longest > MAX_PROCESS_DIM:
        scale = MAX_PROCESS_DIM / longest
        proc = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        proc = img_bgr

    h_img, w_img = proc.shape[:2]
    inv_scale = 1.0 / scale  # 处理图坐标 → 原图坐标
    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    board_lo = np.array([p.board_h_min, p.board_s_min, p.board_v_min])
    board_hi = np.array([p.board_h_max, p.board_s_max, p.board_v_max])
    coil_lo = np.array([p.coil_h_min, p.coil_s_min, p.coil_v_min])
    coil_hi = np.array([p.coil_h_max, p.coil_s_max, p.coil_v_max])

    mask_raw = cv2.inRange(hsv, board_lo, board_hi)
    rects, bx1, bx2, b1y1, b2y2 = detect_boards(mask_raw, h_img)

    info = {"contour_count": 0, "center_x": 0.0, "center_y": 0.0,
            "width": w_orig, "height": h_orig}
    if scale != 1.0:
        info["downscaled_to"] = f"{w_img}x{h_img}"
        info["scale"] = round(scale, 4)

    if not rects:
        info["error"] = "未检测到板（请调宽板 HSV 范围）"
        return result, info

    mask_coil = cv2.inRange(hsv, coil_lo, coil_hi)
    mask_coil_filled = np.zeros_like(mask_coil)
    ks_open = max(1, p.open_ksize) | 1
    ks_close = max(1, p.close_ksize) | 1
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (ks_open, ks_open))
    k_fill = cv2.getStructuringElement(cv2.MORPH_RECT, (ks_close, ks_close))
    for rx, ry, rw, rh in rects:
        roi = mask_coil[ry:ry + rh, rx:rx + rw].copy()
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, k_open, iterations=1)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k_fill,
                               iterations=max(1, p.close_iters))
        mask_coil_filled[ry:ry + rh, rx:rx + rw] = roi

    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), p.canny_lo, p.canny_hi)
    edges = cv2.bitwise_and(edges, edges, mask=mask_coil_filled)
    ctrs, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detail = [c for c in ctrs if p.min_area < cv2.contourArea(c) < p.max_area]

    cv2.drawContours(result, detail, -1, GREEN, 2)
    for rx, ry, rw, rh in rects:
        cv2.rectangle(result, (rx, ry), (rx + rw, ry + rh), GREEN, 2)

    cx = (bx1 + bx2) / 2
    cy = (b1y1 + b2y2) / 2
    cv2.drawMarker(result, (int(cx), int(cy)), GREEN, cv2.MARKER_CROSS, 30, 2)

    cx_orig = cx * inv_scale
    cy_orig = cy * inv_scale
    label = f"匹配框中心:({cx_orig:.2f},{cy_orig:.2f})"
    result = put_cn_text(result, label,
                         (int(cx) + 10, int(cy) - 36), font_size=28, color=GREEN)
    result = put_cn_text(result, f"{w_orig} x {h_orig}",
                         (w_img - 210, h_img - 40), font_size=26, color=CYAN)

    info.update({"contour_count": len(detail),
                 "center_x": cx_orig, "center_y": cy_orig})
    return result, info
