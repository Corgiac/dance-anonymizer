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
    return [t.track_id for t in sorted_tracks], (smooth_history if smooth_history is not None else {})


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
# 美白 — HSV 调色，人物区域内提明度降饱和度
# ================================================================

def apply_skin_whiten(frame, track_results, amount):
    """amount 0-100, 在人物 mask 范围内美白"""
    if amount <= 0:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = amount / 100.0
    for tr in track_results:
        mask = tr.mask
        body = np.clip((mask - 0.10) / 0.10, 0.0, 1.0)
        body = (body > 0.3).astype(np.float32)
        hsv[:, :, 1] -= body * factor * 50   # 降饱和度
        hsv[:, :, 2] += body * factor * 50   # 提明度
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ================================================================
# 拉腿 — 竖直区域纵向拉伸
# ================================================================

def apply_leg_stretch(frame, zone_top, zone_bot, stretch_pct):
    """zone_top/bot 为 0-1 比例, stretch_pct 为 0-100"""
    if stretch_pct <= 0 or zone_top >= zone_bot:
        return frame
    h, w = frame.shape[:2]
    zt = int(zone_top * h)
    zb = int(zone_bot * h)
    stretch = 1.0 + (stretch_pct / 100.0) * 0.5
    zone_h = zb - zt
    stretch_h = int(zone_h * stretch)
    shift = stretch_h - zone_h
    new_h = h + shift

    map_y = np.zeros((new_h, w), dtype=np.float32)
    map_x = np.tile(np.arange(w, dtype=np.float32), (new_h, 1))

    feather = 4  # 边界羽化像素数
    for y in range(new_h):
        if y < zt - feather:
            map_y[y, :] = y
        elif y < zt + feather:
            # 上边界羽化
            t = (y - (zt - feather)) / (2 * feather)
            t = t * t * (3 - 2 * t)  # smoothstep
            orig_y = zt + (y - zt) / stretch
            map_y[y, :] = y * (1 - t) + orig_y * t
        elif y < zt + stretch_h - feather:
            orig_y = zt + (y - zt) / stretch
            map_y[y, :] = orig_y
        elif y < zt + stretch_h + feather:
            # 下边界羽化
            t = (y - (zt + stretch_h - feather)) / (2 * feather)
            t = t * t * (3 - 2 * t)  # smoothstep
            orig_y = zt + (y - zt) / stretch
            shifted_y = y - shift
            map_y[y, :] = orig_y * (1 - t) + shifted_y * t
        else:
            map_y[y, :] = y - shift

    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# ================================================================
# 面部打码 — 基于现有人物 mask
# ================================================================


def apply_face_blur(frame, track_results, face_blur_ids, sticker_img=None,
                    sticker_scale=0.40, size_history=None):
    """对指定人物面部贴纸叠加 — 保持贴纸原比例，可调节大小。size_history 用于平滑贴纸尺寸。"""
    if not face_blur_ids or sticker_img is None:
        return frame
    result = frame.copy()
    h, w = frame.shape[:2]
    face_blur_set = set(face_blur_ids)
    if size_history is None:
        size_history = {}
    for tr in track_results:
        if tr.track_id not in face_blur_set:
            continue
        x1, y1, x2, y2 = tr.bbox
        bw, bh = x2 - x1, y2 - y1
        tid = tr.track_id
        # 贴纸大小锁定：首帧算好 (w, h)，之后完全不变
        if tid not in size_history:
            first_w = int(bw * sticker_scale)
            sh2, sw2 = sticker_img.shape[:2]
            first_h = int(first_w * sh2 / sw2) if sw2 > 0 else first_w
            size_history[tid] = (first_w, first_h)
        sticker_w, sticker_h = size_history[tid]
        # 面部中心：bbox 上方 12% 处，水平居中
        face_cx = (x1 + x2) // 2
        face_cy = y1 + int(bh * 0.12)
        # 贴纸按原始大小叠加，超出画面边缘的部分自然裁掉
        sticker_resized = cv2.resize(sticker_img, (sticker_w, sticker_h))
        if sticker_resized.shape[2] == 4:
            sticker_rgb_full = sticker_resized[:, :, :3]
            sticker_a_full = (sticker_resized[:, :, 3:4].astype(np.float32) / 255.0)
        else:
            sticker_rgb_full = sticker_resized
            sticker_a_full = np.ones((sticker_h, sticker_w, 1), dtype=np.float32)
        # 计算贴纸在帧上的实际可见区域
        sx1 = max(0, face_cx - sticker_w // 2)
        sy1 = max(0, face_cy - sticker_h // 2)
        sx2 = min(w, sx1 + sticker_w)
        sy2 = min(h, sy1 + sticker_h)
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        # 贴纸被裁部分对应下标
        dx1 = sx1 - (face_cx - sticker_w // 2)
        dy1 = sy1 - (face_cy - sticker_h // 2)
        dx2 = dx1 + (sx2 - sx1)
        dy2 = dy1 + (sy2 - sy1)
        sticker_rgb = sticker_rgb_full[dy1:dy2, dx1:dx2]
        sticker_a = sticker_a_full[dy1:dy2, dx1:dx2]
        roi = result[sy1:sy2, sx1:sx2]
        result[sy1:sy2, sx1:sx2] = (
            roi.astype(np.float32) * (1 - sticker_a) +
            sticker_rgb.astype(np.float32) * sticker_a
        ).astype(np.uint8)
    return result


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

        if fill_mode == "none":
            pass  # 只画白边，不填色
        elif fill_mode == "blur":
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
    face_blur_ids=None, sticker_img=None, sticker_scale=0.40,
    skin_whiten=0, leg_stretch_on=False,
    leg_stretch=0, leg_zone_top=0.5, leg_zone_bot=0.75,
):
    """
    对单帧应用特效 + 文字标签。
    label_mode: "all" (预览,含默认ID) | "custom_only" (最终视频,仅昵称)
    """
    # 保留完整 track_results 给标签绘制用
    all_track_results = track_results

    # 美白：在所有人物区域调整 HSV
    if skin_whiten > 0:
        frame = apply_skin_whiten(frame, all_track_results, skin_whiten)

    # 面部贴纸
    if face_blur_ids:
        frame = apply_face_blur(frame, all_track_results, face_blur_ids,
                                sticker_img=sticker_img, sticker_scale=sticker_scale,
                                size_history=smooth_history)

    # 打码仅作用于 target_ids
    if target_ids is not None:
        target_set = set(target_ids)
        non_target_tracks = [tr for tr in track_results if tr.track_id not in target_set]
        track_results = [tr for tr in track_results if tr.track_id in target_set]

        # 掩码相减：从 target mask 中扣除 non-target 重叠区域
        if non_target_tracks and track_results:
            h, w = track_results[0].mask.shape[:2]
            exclusion = np.zeros((h, w), dtype=np.float32)
            for nt in non_target_tracks:
                exclusion = np.maximum(exclusion, nt.mask.astype(np.float32))
            for tr in track_results:
                tr.mask = (tr.mask.astype(np.float32) * (1.0 - exclusion)).astype(tr.mask.dtype)

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

    # 拉腿：最后一步，竖直区域拉伸（避免与其他效果维度冲突）
    if leg_stretch_on and leg_stretch > 0:
        result = apply_leg_stretch(result, leg_zone_top, leg_zone_bot, leg_stretch)

    return result, smooth_history
