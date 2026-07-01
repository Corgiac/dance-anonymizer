"""
Phase 3: 特效 & 深度排序模块 — SAM 2 直出 mask 版本
=====================================================
- SAM 2 输出的 mask 边缘精准平滑, 无需 EMA 时序缓存
- 保留: 白边提取、深度排序、填充渲染
- 新增: Pillow 文字标签渲染 (正上方居中, 支持中文)
- 删除: TemporalMaskCache 及其所有相关逻辑
"""
import os, cv2
import numpy as np
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont


# ================================================================
# Alpha Edge (白边提取)
# ================================================================

def extract_edge_alpha(alpha: np.ndarray, edge_width: int = 3) -> np.ndarray:
    """提取锐利白边: dilate → 差分（调用方保证 alpha 已二值化）"""
    hard = alpha.astype(np.uint8)
    if hard.max() == 0:
        return np.zeros_like(alpha, dtype=np.float32)
    ksize = max(3, edge_width) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(hard, kernel)
    edge = (dilated - hard).astype(np.float32)  # 0 或 1, 锐利无模糊
    return np.clip(edge, 0.0, 1.0)


def blend_body(alpha: np.ndarray, dilate_size: int = 0) -> np.ndarray:
    """膨胀主体遮罩，确保遮盖快速运动的肢体残影（调用方保证 alpha 已二值化）"""
    body = alpha.astype(np.float32)
    if dilate_size > 0:
        ksize = max(3, dilate_size) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        body = cv2.dilate(body, kernel, iterations=1).astype(np.float32)
    return body


def blend_edge(alpha: np.ndarray, edge_width: int = 3) -> np.ndarray:
    """从二值化 mask 提取白边（调用方保证 alpha 已二值化）"""
    return extract_edge_alpha(alpha, edge_width)


# ================================================================
# 深度排序 — 纯实时动态
# ================================================================

def calculate_depth_order(
    track_results: List,
    temporal_window: int = 5,
    smooth_history: Optional[dict] = None,
) -> Tuple[List[int], dict]:
    sorted_tracks = sorted(track_results, key=lambda t: t.foot_y)
    return [t.track_id for t in sorted_tracks], (smooth_history or {})


# ================================================================
# 辅助渲染函数
# ================================================================

def _hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def _render_gradient_fill(h, w, alpha, color_top, color_bot):
    gradient = np.zeros((h, w, 3), dtype=np.float32)
    for row in range(h):
        t = row / max(h - 1, 1)
        b = color_top[0] * (1 - t) + color_bot[0] * t
        g_val = color_top[1] * (1 - t) + color_bot[1] * t
        r_val = color_top[2] * (1 - t) + color_bot[2] * t
        gradient[row, :] = [b, g_val, r_val]
    a = np.clip(alpha, 0, 1)[:, :, np.newaxis]
    return (gradient * a).astype(np.uint8)


def _render_blur_fill(frame, alpha, blur_ksize=31):
    blurred = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), sigmaX=20)
    a = np.clip(alpha, 0, 1)[:, :, np.newaxis]
    return (frame.astype(np.float32) * (1 - a) + blurred.astype(np.float32) * a).astype(np.uint8)


# ================================================================
# 特效渲染核心
# ================================================================

def apply_shadow_outline_effect(
    frame, depth_order, track_id_to_idx, track_results,
    dilate_kernel_size=3, fill_mode="solid",
    fill_color="#000000", border_color="#FFFFFF", opacity=1.0,
):
    h, w = frame.shape[:2]
    result = frame.copy()
    fill_bgr = _hex_to_bgr(fill_color)
    border_bgr = _hex_to_bgr(border_color)
    grad_top, grad_bot = fill_bgr, tuple(max(0, c - 60) for c in fill_bgr)

    for tid in depth_order:
        idx = track_id_to_idx.get(tid)
        if idx is None: continue
        tr = track_results[idx]
        alpha = tr.mask.astype(np.float32)
        if alpha.max() < 0.01: continue
        # 软阈值映射: (x-0.10)/0.10 → clip [0,1]
        # Cutie prob 0.20 → 1.0  (实心)
        # Cutie prob 0.15 → 0.5  (过渡)
        # Cutie prob 0.12 → 0.2  (微弱但不断)
        # Cutie prob 0.10 → 0.0  (真背景, 不引入噪点)
        alpha = np.clip((alpha - 0.10) / 0.10, 0.0, 1.0)

        # 主体遮罩使用软阈值 (连续值, 帧间无跳变)
        body_dilate = max(3, dilate_kernel_size - 2)
        body = blend_body(alpha, dilate_size=body_dilate)
        # 白边使用硬阈值提取 (保持锐利), 在软阈值 alpha 上取 0.5 分界
        alpha_edge = (alpha > 0.5).astype(np.float32)
        edge = blend_edge(alpha_edge, edge_width=dilate_kernel_size)

        if fill_mode == "blur":
            result = _render_blur_fill(result, body * opacity)
        elif fill_mode == "gradient":
            grad_layer = _render_gradient_fill(h, w, body * opacity, grad_top, grad_bot)
            a = np.clip(body * opacity, 0, 1)[:, :, np.newaxis]
            result = (result.astype(np.float32) * (1 - a) + grad_layer.astype(np.float32)).astype(np.uint8)
        else:
            a = np.clip(body * opacity, 0, 1)
            a3 = a[:, :, np.newaxis]
            fill = np.array(fill_bgr, dtype=np.float32).reshape(1, 1, 3)
            result = (result.astype(np.float32) * (1 - a3) + fill * a3).astype(np.uint8)

        if edge.max() > 0.001:
            e3 = edge[:, :, np.newaxis]
            border = np.array(border_bgr, dtype=np.float32).reshape(1, 1, 3)
            result = (result.astype(np.float32) * (1 - e3) + border * e3).astype(np.uint8)

    return result


# ================================================================
# Pillow 文字标签 (正上方居中, 支持中文)
# label_mode: "all" = 所有人(含默认ID) | "custom_only" = 仅自定义昵称
# ================================================================

FONT_PATHS = [
    "/System/Library/Fonts/STHeiti Medium.ttc",   # macOS
    "/System/Library/Fonts/PingFang.ttc",          # macOS
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux
    "C:/Windows/Fonts/msyh.ttc",                   # Windows
]


def _resolve_font():
    for p in FONT_PATHS:
        if os.path.exists(p):
            return p
    return None


def draw_text_labels(frame_bgr, track_results, labels_config, font_path=None, label_mode="all", font_size=24):
    """
    在帧上绘制跟随人物的文字标签 (Pillow, 正上方居中)。
    label_mode:
      "all"         — 所有人: labels_config 有则 @昵称, 无则 ID:{id}
      "custom_only" — 仅 labels_config 中存在的人, 仅 @昵称
    """
    if label_mode == "all" and labels_config is None:
        labels_config = {}
    if label_mode == "custom_only" and not labels_config:
        return frame_bgr

    fp = font_path if font_path and os.path.exists(font_path) else _resolve_font()
    if fp is None:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(fp, font_size)
    except Exception:
        return frame_bgr

    for tr in track_results:
        tid_str = str(tr.track_id)

        # 确定要显示的文字
        if label_mode == "all":
            if labels_config and tid_str in labels_config:
                text = "@" + labels_config[tid_str].get("text", "")
            else:
                text = f"人物{tr.track_id + 1}"
        else:  # custom_only
            if labels_config and tid_str in labels_config:
                text = "@" + labels_config[tid_str].get("text", "")
            else:
                continue

        if not text or text == "@":
            continue

        # ★ 从 bbox 计算正上方居中坐标 (比 mask 更稳定, 不受噪点影响)
        x1, y1, x2, y2 = tr.bbox
        center_x = (x1 + x2) / 2
        top_y = y1

        # Pillow 测量文字尺寸
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 正上方居中: 水平居中, 垂直距头顶 25px
        tx = max(0, min(center_x - tw // 2, w - tw))
        ty = max(0, top_y - th - 25)

        # 黑色半透明背景框
        draw.rectangle([tx - 4, ty - 2, tx + tw + 4, ty + th + 2], fill=(0, 0, 0, 180))
        draw.text((tx, ty), text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ================================================================
# 便捷封装
# ================================================================

def process_frame_effects(
    frame, track_results, smooth_history=None, frame_idx=0,
    dilate_kernel_size=3, temporal_window=8, target_ids=None,
    fill_mode="solid", fill_color="#000000",
    border_color="#FFFFFF", opacity=1.0,
    labels_config=None, font_path=None, label_mode="custom_only",
):
    """
    对单帧应用特效 + 文字标签。
    label_mode: "all" (预览,含默认ID) | "custom_only" (最终视频,仅昵称)
    """
    # 保留完整 track_results 给标签绘制用
    all_track_results = track_results

    # 打码仅作用于 target_ids
    if target_ids is not None:
        target_set = set(target_ids)
        track_results = [tr for tr in track_results if tr.track_id in target_set]

    if track_results:
        ordered_ids, smooth_history = calculate_depth_order(
            track_results,
            temporal_window=max(3, temporal_window // 2),
            smooth_history=smooth_history,
        )
        track_id_to_idx = {tr.track_id: i for i, tr in enumerate(track_results)}
        result = apply_shadow_outline_effect(
            frame=frame, depth_order=ordered_ids,
            track_id_to_idx=track_id_to_idx, track_results=track_results,
            dilate_kernel_size=dilate_kernel_size,
            fill_mode=fill_mode, fill_color=fill_color,
            border_color=border_color, opacity=opacity,
        )
    else:
        result = frame.copy()

    # 文字标签 (在特效之上, 使用完整 track_results)
    result = draw_text_labels(result, all_track_results, labels_config,
                               font_path=font_path, label_mode=label_mode)

    return result, smooth_history
