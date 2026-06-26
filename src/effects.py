"""
Phase 3: 特效 & 深度排序模块 — Soft Alpha Pipeline
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from collections import deque


# ================================================================
# Alpha Edge
# ================================================================

def _extract_edge_alpha(alpha: np.ndarray, edge_width: int = 3) -> np.ndarray:
    """
    从硬掩码提取白边: dilate - original, 无 findContours, 无 GaussianBlur。
    假定输入已是二值或可二值化的 mask。
    """
    hard = (alpha > 0.3).astype(np.uint8)
    if hard.max() == 0:
        return np.zeros_like(alpha, dtype=np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width, edge_width))
    dilated = cv2.dilate(hard, kernel)
    edge = (dilated - hard).astype(np.float32)
    return edge


# ================================================================
# EMA 时序缓存
# ================================================================

class TemporalMaskCache:
    def __init__(self, window_size=5, ema_decay=0.3):
        self.ema_decay = ema_decay
        self._cache: Dict[int, np.ndarray] = {}

    def update(self, track_id: int, alpha: np.ndarray, frame_idx: int):
        self._cache[track_id] = alpha.copy()

    def get_blended_body(self, track_id: int) -> Optional[np.ndarray]:
        alpha = self._cache.get(track_id)
        if alpha is None:
            return None
        return np.clip(cv2.GaussianBlur(alpha, (3, 3), sigmaX=0.8), 0.0, 1.0)

    def get_blended_edge(self, track_id: int, edge_width: int = 3) -> Optional[np.ndarray]:
        alpha = self._cache.get(track_id)
        if alpha is None:
            return None
        smoothed = cv2.GaussianBlur(alpha, (5, 5), sigmaX=1.0)
        return _extract_edge_alpha(smoothed, edge_width)

    def cleanup_stale(self, active_ids: set):
        for tid in list(self._cache.keys()):
            if tid not in active_ids:
                del self._cache[tid]

    def reset(self):
        self._cache.clear()


# ================================================================
# 深度排序 — 首帧冻结
# ================================================================

def calculate_depth_order(
    track_results: List,
    temporal_window: int = 5,
    smooth_history: Optional[dict] = None,
) -> Tuple[List[int], dict]:
    BUFFER_FRAMES = 15

    frozen_order = getattr(calculate_depth_order, "_frozen_order", None)
    frozen_foot_y = getattr(calculate_depth_order, "_frozen_foot_y", None)
    foot_y_buffer = getattr(calculate_depth_order, "_foot_y_buffer", None)
    buffer_count = getattr(calculate_depth_order, "_buffer_count", 0)

    current_ids = [t.track_id for t in track_results]
    current_foot_y = {t.track_id: t.foot_y for t in track_results}

    # ================================================================
    # 初始化缓冲: 前 15 帧动态排序 + 收集 foot_y, 满后取中位数永久冻结
    # ================================================================
    if frozen_order is None:
        if foot_y_buffer is None:
            foot_y_buffer = {}

        for t in track_results:
            tid = t.track_id
            if tid not in foot_y_buffer:
                foot_y_buffer[tid] = []
            foot_y_buffer[tid].append(t.foot_y)

        buffer_count += 1
        calculate_depth_order._foot_y_buffer = foot_y_buffer
        calculate_depth_order._buffer_count = buffer_count

        if buffer_count < BUFFER_FRAMES:
            # 缓冲期内: 按当前帧 foot_y 动态排序
            sorted_tracks = sorted(track_results, key=lambda t: t.foot_y)
            return [t.track_id for t in sorted_tracks], (smooth_history or {})

        # 缓冲期满: 取每个 track_id 的中位数 foot_y 永久冻结
        median_foot_y = {}
        for tid, fys in foot_y_buffer.items():
            median_foot_y[tid] = float(np.median(fys))

        sorted_ids = sorted(median_foot_y.keys(), key=lambda tid: median_foot_y[tid])
        frozen_order = sorted_ids
        calculate_depth_order._frozen_order = frozen_order
        calculate_depth_order._frozen_foot_y = median_foot_y
        # 释放缓冲内存
        calculate_depth_order._foot_y_buffer = None
        calculate_depth_order._buffer_count = 0

    # ================================================================
    # 冻结后: 用冻结顺序, 新 ID 按 foot_y 插入正确位置而非末尾
    # ================================================================
    ordered = [tid for tid in frozen_order if tid in current_ids]

    for tid in current_ids:
        if tid not in ordered:
            new_fy = current_foot_y[tid]
            insert_pos = len(ordered)
            for i, existing_tid in enumerate(ordered):
                existing_fy = frozen_foot_y.get(existing_tid) if frozen_foot_y else None
                if existing_fy is not None and new_fy < existing_fy:
                    insert_pos = i
                    break
            ordered.insert(insert_pos, tid)
            # 同步写入 frozen_order, 后续帧该 ID 不再是"新"
            frozen_order.insert(insert_pos, tid)
            if frozen_foot_y is not None:
                frozen_foot_y[tid] = new_fy
            calculate_depth_order._frozen_order = frozen_order
            calculate_depth_order._frozen_foot_y = frozen_foot_y

    return ordered, (smooth_history or {})


# ================================================================
# 辅助
# ================================================================

def _hex_to_bgr(hex_color: str) -> Tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def _render_gradient_fill(h: int, w: int, alpha: np.ndarray,
                           color_top: tuple, color_bot: tuple) -> np.ndarray:
    gradient = np.zeros((h, w, 3), dtype=np.float32)
    for row in range(h):
        t = row / max(h - 1, 1)
        b = color_top[0] * (1 - t) + color_bot[0] * t
        g_val = color_top[1] * (1 - t) + color_bot[1] * t
        r_val = color_top[2] * (1 - t) + color_bot[2] * t
        gradient[row, :] = [b, g_val, r_val]
    a = np.clip(alpha, 0, 1)[:, :, np.newaxis]
    return (gradient * a).astype(np.uint8)


def _render_blur_fill(frame: np.ndarray, alpha: np.ndarray,
                       blur_ksize: int = 31) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), sigmaX=20)
    a = np.clip(alpha, 0, 1)[:, :, np.newaxis]
    result = (frame.astype(np.float32) * (1 - a) + blurred.astype(np.float32) * a)
    return result.astype(np.uint8)


# ================================================================
# 特效渲染核心
# ================================================================

def apply_shadow_outline_effect(
    frame: np.ndarray,
    depth_order: List[int],
    track_id_to_idx: dict,
    track_results: List,
    temporal_cache: TemporalMaskCache,
    frame_idx: int,
    dilate_kernel_size: int = 3,
    fill_mode: str = "solid",
    fill_color: str = "#000000",
    border_color: str = "#FFFFFF",
    opacity: float = 1.0,
    target_ids: Optional[List[int]] = None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    result = frame.copy()
    fill_bgr = _hex_to_bgr(fill_color)
    border_bgr = _hex_to_bgr(border_color)
    grad_top = fill_bgr
    grad_bot = tuple(max(0, c - 60) for c in fill_bgr)

    # ================================================================
    # Single-Pass: 远→近, 每人先画实体紧接着画白边, 作为一个整体被更近的人覆盖
    # ================================================================
    for tid in depth_order:
        idx = track_id_to_idx.get(tid)
        if idx is None: continue
        tr = track_results[idx]
        alpha = tr.mask.astype(np.float32)
        if alpha.max() < 0.01: continue

        temporal_cache.update(tid, alpha, frame_idx)
        blended_body = temporal_cache.get_blended_body(tid)
        if blended_body is None: blended_body = alpha

        # 1. 实体填充
        if fill_mode == "blur":
            result = _render_blur_fill(result, blended_body * opacity)
        elif fill_mode == "gradient":
            grad_layer = _render_gradient_fill(h, w, blended_body * opacity, grad_top, grad_bot)
            a = np.clip(blended_body * opacity, 0, 1)[:, :, np.newaxis]
            result = (result.astype(np.float32) * (1 - a)
                      + grad_layer.astype(np.float32)).astype(np.uint8)
        else:
            a = np.clip(blended_body * opacity, 0, 1)
            a3 = a[:, :, np.newaxis]
            fill = np.array(fill_bgr, dtype=np.float32).reshape(1, 1, 3)
            result = (result.astype(np.float32) * (1 - a3) + fill * a3).astype(np.uint8)

        # 2. 白边 — 紧接在同一循环内绘制, "远人的白边"会被"近人的实体"正确覆盖
        blended_edge = temporal_cache.get_blended_edge(tid, edge_width=dilate_kernel_size)
        if blended_edge is None:
            blended_edge = _extract_edge_alpha(blended_body, dilate_kernel_size)
        if blended_edge.max() > 0.001:
            e3 = blended_edge[:, :, np.newaxis]
            border = np.array(border_bgr, dtype=np.float32).reshape(1, 1, 3)
            result = (result.astype(np.float32) * (1 - e3) + border * e3).astype(np.uint8)

    return result


# ================================================================
# 便捷函数
# ================================================================

def process_frame_effects(
    frame: np.ndarray,
    track_results: List,
    temporal_cache: Optional[TemporalMaskCache] = None,
    smooth_history: Optional[dict] = None,
    frame_idx: int = 0,
    dilate_kernel_size: int = 3,
    temporal_window: int = 8,
    target_ids: Optional[List[int]] = None,
    fill_mode: str = "solid",
    fill_color: str = "#000000",
    border_color: str = "#FFFFFF",
    opacity: float = 1.0,
) -> Tuple[np.ndarray, dict, TemporalMaskCache]:
    if temporal_cache is None:
        temporal_cache = TemporalMaskCache(window_size=temporal_window, ema_decay=0.3)

    if target_ids is not None:
        target_set = set(target_ids)
        track_results = [tr for tr in track_results if tr.track_id in target_set]

    if not track_results:
        return frame.copy(), (smooth_history or {}), temporal_cache

    ordered_ids, smooth_history = calculate_depth_order(
        track_results,
        temporal_window=max(3, temporal_window // 2),
        smooth_history=smooth_history,
    )

    track_id_to_idx = {tr.track_id: i for i, tr in enumerate(track_results)}

    result = apply_shadow_outline_effect(
        frame=frame,
        depth_order=ordered_ids,
        track_id_to_idx=track_id_to_idx,
        track_results=track_results,
        temporal_cache=temporal_cache,
        frame_idx=frame_idx,
        dilate_kernel_size=dilate_kernel_size,
        fill_mode=fill_mode,
        fill_color=fill_color,
        border_color=border_color,
        opacity=opacity,
        target_ids=target_ids,
    )

    active_ids = {tr.track_id for tr in track_results}
    temporal_cache.cleanup_stale(active_ids)

    return result, smooth_history, temporal_cache
