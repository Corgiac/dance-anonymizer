"""
Phase 2: 分割 & 追踪模块 — Soft Alpha Pipeline
=================================================
- YOLOv8-Seg + BoT-SORT 追踪
- 输出 soft alpha mask (float32 [0,1]) 而非 bool mask
- 时序 fallback: 当前帧无 mask 时复用上一帧 EMA alpha

设计原则:
  1. YOLO probability → soft alpha via clip((prob-0.1)/0.4, 0, 1)
  2. BoT-SORT 自动维护 track_id
  3. TrackResult.mask 现在是 float32 alpha, 不再用 bool
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from ultralytics import YOLO


@dataclass
class TrackResult:
    """单个人物的追踪结果 — mask 为 soft alpha float32 [0,1]"""
    track_id: int
    bbox: Tuple[int, int, int, int]  # xyxy
    confidence: float
    mask: np.ndarray                   # (H, W) float32 alpha [0, 1]
    foot_y: float = 0.0

    def __repr__(self):
        return (f"TrackResult(id={self.track_id}, "
                f"bbox={self.bbox}, conf={self.confidence:.3f}, "
                f"foot_y={self.foot_y:.1f})")


@dataclass
class TrackerConfig:
    """追踪器运行时配置"""
    model_path: str = "yolo11s-seg.pt"
    device: str = "cuda"
    conf_threshold: float = 0.3
    iou_threshold: float = 0.55
    imgsz: int = 1280                   # 推理分辨率
    retina_masks: bool = True           # 高精度原生分辨率 mask
    track_buffer: int = 60
    iou_match_threshold: float = 0.35
    expected_count: Optional[int] = None
    half: bool = False
    verbose: bool = True
    ghost_frames: int = 1
    min_alpha_sum: float = 800.0


class DanceTracker:
    """
    舞蹈视频人物追踪器 — Soft Alpha Pipeline
    YOLOv8-Seg + BoT-SORT → float32 alpha mask
    """

    def __init__(self, config: TrackerConfig = TrackerConfig()):
        self.config = config
        if config.verbose:
            print(f"[DanceTracker] 加载模型: {config.model_path}")
        self.model = YOLO(config.model_path)
        self._device = config.device
        # 时序 fallback 缓存: {track_id: alpha_mask}
        self._fallback_alphas: dict = {}

    def track(self, frame: np.ndarray) -> List[TrackResult]:
        h, w = frame.shape[:2]
        results = self._detect_and_track(frame)
        return results

    def reset(self):
        self._fallback_alphas.clear()

    # ================================================================
    # BoT-SORT + Soft Alpha
    # ================================================================

    def _detect_and_track(self, frame: np.ndarray) -> List[TrackResult]:
        results = self.model.track(
            source=frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            device=self._device,
            classes=[0],
            verbose=False,
            half=self.config.half,
            retina_masks=self.config.retina_masks,
            imgsz=self.config.imgsz,
            persist=True,
            tracker="botsort.yaml",
        )

        track_results = []
        result = results[0]
        seen_ids = set()
        h, w = frame.shape[:2]

        if result is not None and result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                tid_raw = result.boxes.id
                if tid_raw is None:
                    continue
                track_id = int(tid_raw[i].item())
                seen_ids.add(track_id)

                bbox_xyxy = result.boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = bbox_xyxy.astype(int).tolist()
                conf = float(result.boxes.conf[i].item())

                if result.masks is not None:
                    m = result.masks.data[i].cpu().numpy()
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.float32), (w, h),
                                       interpolation=cv2.INTER_LINEAR)
                    # 硬阈值二值化 + 闭运算填洞 + 边缘抗锯齿 → float32 [0,1]
                    hard = (m > 0.4).astype(np.uint8)
                    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    closed = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, kernel_close)
                    alpha = cv2.GaussianBlur(closed.astype(np.float32), (3, 3), sigmaX=0)
                else:
                    continue

                # 小区域过滤: alpha 总和 < min_alpha_sum 则跳过
                if alpha.sum() < self.config.min_alpha_sum:
                    continue

                # BBox 微小膨胀
                bw, bh = x2 - x1, y2 - y1
                expand_w, expand_h = int(bw * 0.1), int(bh * 0.05)
                x1 = max(0, x1 - expand_w)
                y1 = max(0, y1 - expand_h)
                x2 = min(w, x2 + expand_w)
                y2 = min(h, y2 + expand_h)

                # foot_y: alpha 加权中心
                if alpha.sum() > 100:
                    row_weights = alpha.sum(axis=1)
                    ys = np.where(row_weights > 0.01)[0]
                    foot_y = float(ys.max()) if len(ys) > 0 else float(y2)
                else:
                    foot_y = float(y2)

                # 更新 fallback 缓存
                self._fallback_alphas[track_id] = alpha.copy()

                track_results.append(TrackResult(
                    track_id=track_id, bbox=(x1, y1, x2, y2),
                    confidence=conf, mask=alpha, foot_y=foot_y,
                ))

        # ★ 时序 fallback
        for tid, cached_alpha in list(self._fallback_alphas.items()):
            if tid not in seen_ids:
                decayed = cached_alpha * 0.85
                if decayed.sum() > 400:
                    track_results.append(TrackResult(
                        track_id=tid, bbox=(0, 0, w, h), confidence=0.3,
                        mask=decayed.astype(np.float32),
                        foot_y=float(np.argmax(decayed.sum(axis=1))) if decayed.any() else float(h),
                    ))
                    self._fallback_alphas[tid] = decayed
                else:
                    del self._fallback_alphas[tid]

        track_results.sort(key=lambda t: t.track_id)
        return track_results


# ================================================================
# 辅助函数
# ================================================================

def cv2_resize_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    """将 mask resize 到 (h, w)。兼容 bool 和 float。"""
    return cv2.resize(
        mask.astype(np.float32), (w, h),
        interpolation=cv2.INTER_LINEAR
    )
