"""
Phase 2: 首帧检测模块 — YOLO → SAM 2 混合架构
=================================================
- 仅负责首帧 YOLO 检测, 提供规范化 bbox 列表给 SAM 2 注册
- SAM 2 接管全片追踪, 本模块不再负责时序跟踪/fallback/mask生成
"""
import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class TrackResult:
    """单个人物结果 — mask 由 SAM 2 提供, 此处仅保留兼容结构"""
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
    """首帧检测器配置 (仅 YOLO 部分, SAM 2 配置见 pipeline)"""
    model_path: str = "yolo11s-seg.pt"
    device: str = "mps"
    conf_threshold: float = 0.3
    iou_threshold: float = 0.55
    imgsz: int = 1280
    verbose: bool = True


def compute_foot_y(mask: np.ndarray) -> float:
    """从 mask 计算脚部 y 坐标 (行权重加权底部)"""
    if mask.sum() < 100:
        return float(mask.shape[0])
    row_weights = mask.sum(axis=1)
    ys = np.where(row_weights > 0.01)[0]
    return float(ys.max()) if len(ys) > 0 else float(mask.shape[0])


def compute_bbox_from_mask(mask: np.ndarray, expand_ratio: float = 0.05) -> Tuple[int, int, int, int]:
    """从二值/soft mask 提取包围框 (带微小膨胀)"""
    h, w = mask.shape
    binary = (mask > 0.15).astype(np.uint8)
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any() or not cols.any():
        return (0, 0, w, h)
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y1, y2 = int(y_indices[0]), int(y_indices[-1])
    x1, x2 = int(x_indices[0]), int(x_indices[-1])
    # 微小膨胀
    bw, bh = x2 - x1, y2 - y1
    expand_w, expand_h = int(bw * expand_ratio), int(bh * expand_ratio)
    x1 = max(0, x1 - expand_w)
    y1 = max(0, y1 - expand_h)
    x2 = min(w, x2 + expand_w)
    y2 = min(h, y2 + expand_h)
    return (x1, y1, x2, y2)


class DanceTracker:
    """
    首帧 YOLO 检测器 — 仅为 SAM 2 提供初始 bbox 注册。
    不再负责逐帧追踪、soft alpha 转换、fallback。
    """

    def __init__(self, config: TrackerConfig = TrackerConfig()):
        self.config = config
        if config.verbose:
            print(f"[DanceTracker] 加载首帧检测模型: {config.model_path}")
        self.model = YOLO(config.model_path)

    def detect_first_frame(self, frame: np.ndarray) -> List[TrackResult]:
        """
        对首帧做 YOLO 检测, 返回带 bbox + foot_y 的 TrackResult 列表。
        mask 由 YOLO 分割头提供 (仅用于计算 foot_y + bbox, SAM 2 会重新生成)。
        """
        results = self.model.predict(
            source=frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            classes=[0],
            verbose=False,
            retina_masks=True,
            imgsz=self.config.imgsz,
        )

        track_results = []
        result = results[0]
        h, w = frame.shape[:2]
        next_id = 0

        if result is not None and result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                bbox_xyxy = result.boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = bbox_xyxy.astype(int).tolist()
                conf = float(result.boxes.conf[i].item())
                track_id = next_id
                next_id += 1

                # 从 YOLO mask 提取初步 alpha (仅用于 foot_y + bbox 计算)
                if result.masks is not None:
                    m = result.masks.data[i].cpu().numpy()
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.float32), (w, h),
                                       interpolation=cv2.INTER_LINEAR)
                    alpha = np.where(m > 0.2, m, 0.0).astype(np.float32)
                    alpha = cv2.GaussianBlur(alpha, (5, 5), sigmaX=1.0)
                    alpha = np.clip(alpha, 0.0, 1.0)
                else:
                    continue

                foot_y = compute_foot_y(alpha)
                # bbox 膨胀
                bw, bh = x2 - x1, y2 - y1
                expand_w, expand_h = int(bw * 0.1), int(bh * 0.05)
                x1 = max(0, x1 - expand_w)
                y1 = max(0, y1 - expand_h)
                x2 = min(w, x2 + expand_w)
                y2 = min(h, y2 + expand_h)

                track_results.append(TrackResult(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    mask=alpha,
                    foot_y=foot_y,
                ))

        track_results.sort(key=lambda t: t.track_id)
        return track_results

    def reset(self):
        pass
