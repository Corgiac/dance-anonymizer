"""
视频追踪引擎抽象层 — 支持 SAM 2 / Cutie 双引擎热切换
=====================================================
BaseVideoTracker → SAM2Tracker | CutieTracker
"""
import os, sys
import cv2
import numpy as np
import torch
from typing import List, Tuple
from abc import ABC, abstractmethod

from .tracker import TrackResult, compute_foot_y, compute_bbox_from_mask


# ================================================================
# 辅助: 首帧检测结果从左到右排序 (与 /analyze 算法一致)
# ================================================================

def sort_detections_left_to_right(detections: List[TrackResult]) -> List[TrackResult]:
    """按 bbox 中心 X 升序排列, 重新分配 ID 0,1,2..."""
    detections.sort(key=lambda t: (t.bbox[0] + t.bbox[2]) / 2.0)
    sorted_dets = []
    for new_id, det in enumerate(detections):
        sorted_dets.append(TrackResult(
            track_id=new_id, bbox=det.bbox,
            confidence=det.confidence, mask=det.mask, foot_y=det.foot_y,
        ))
    return sorted_dets


# ================================================================
# 抽象基类
# ================================================================

class BaseVideoTracker(ABC):
    """视频目标追踪引擎接口"""

    @abstractmethod
    def initialize(self, first_frame: np.ndarray,
                   detections: List[TrackResult],
                   all_tracked_ids: List[int]) -> None:
        """用首帧和 YOLO 检测结果初始化追踪器"""
        ...

    @abstractmethod
    def step(self, frame: np.ndarray, frame_idx: int) -> List[TrackResult]:
        """处理一帧, 返回追踪结果列表"""
        ...

    @abstractmethod
    def reset(self):
        ...


# ================================================================
# SAM 2 引擎
# ================================================================

class SAM2Tracker(BaseVideoTracker):
    """
    SAM 2 视频预测器引擎。
    首帧初始化 → propagate_in_video 逐帧产出 mask。
    """

    def __init__(self, model_path: str, device: str = None,
                 verbose: bool = True):
        self.model_path = model_path
        self.device = device
        self.verbose = verbose
        self._predictor = None
        self._inference_state = None
        self._propagator = None
        self._w, self._h = 0, 0

    def _resolve_device(self, actual_frames: int) -> str:
        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available() and actual_frames <= 120:
            return "mps"
        else:
            return "cpu"

    def initialize(self, first_frame: np.ndarray,
                   detections: List[TrackResult],
                   all_tracked_ids: List[int]) -> None:
        """
        参数:
          first_frame: BGR uint8 首帧
          detections: YOLO 检测结果 (已排序, ID 已对齐)
          all_tracked_ids: 需要追踪的 ID 列表
        """
        from sam2.build_sam import build_sam2_video_predictor

        self._h, self._w = first_frame.shape[:2]

        # 从帧目录初始化 SAM 2 (帧需要预先抽取到 frames_dir)
        # 这里假设帧已经在 _frames_dir 目录中
        # 注意: SAM 2 的 init_state 需要加载全部帧, 所以帧目录必须在调用前准备好
        if self.verbose:
            print(f"[SAM2Tracker] 初始化, 设备: {self.device}")

    def initialize_from_dir(self, frames_dir: str, first_frame: np.ndarray,
                             detections: List[TrackResult],
                             all_tracked_ids: List[int]) -> int:
        """SAM 2 专用: 从帧目录初始化"""
        from sam2.build_sam import build_sam2_video_predictor

        self._h, self._w = first_frame.shape[:2]
        self._frames_dir = frames_dir

        # 统计帧数
        actual_frames = len([f for f in os.listdir(frames_dir)
                             if f.endswith('.jpg')])
        device = self._resolve_device(actual_frames)
        if self.verbose:
            print(f"[SAM2Tracker] 设备: {device}, 帧数: {actual_frames}")

        predictor = build_sam2_video_predictor(
            "sam2_hiera_t.yaml", ckpt_path=self.model_path, device=device)
        inference_state = predictor.init_state(video_path=frames_dir)

        # ★ 引擎内部强制从左到右排序 (与 /analyze 对齐, 防御性编程)
        detections = sort_detections_left_to_right(detections)

        # 注册目标
        track_set = set(all_tracked_ids)
        for det in detections:
            tid = det.track_id
            if tid not in track_set:
                if self.verbose:
                    print(f"[SAM2Tracker]   跳过 ID:{tid}")
                continue
            x1, y1, x2, y2 = det.bbox
            predictor.add_new_points_or_box(
                inference_state=inference_state, frame_idx=0, obj_id=tid,
                box=[float(x1), float(y1), float(x2), float(y2)],
            )
            if self.verbose:
                print(f"[SAM2Tracker]   注册 ID:{tid} bbox=[{x1},{y1},{x2},{y2}]")

        self._predictor = predictor
        self._inference_state = inference_state
        # 启动生成器
        self._propagator = predictor.propagate_in_video(inference_state)
        self._actual_frames = actual_frames
        return actual_frames

    def step(self, frame: np.ndarray, frame_idx: int) -> List[TrackResult]:
        """从帧图片读取并推进 SAM 2 传播 (frame 参数仅用于渲染, 追踪数据来自 propagate)"""
        if self._propagator is None:
            return []

        try:
            out_frame_idx, out_obj_ids, out_mask_logits = next(self._propagator)
        except StopIteration:
            return []

        if out_frame_idx != frame_idx:
            # 帧索引对齐
            while out_frame_idx < frame_idx:
                try:
                    out_frame_idx, out_obj_ids, out_mask_logits = next(self._propagator)
                except StopIteration:
                    return []

        track_results = []
        mask_logits_all = out_mask_logits
        for i, obj_id in enumerate(out_obj_ids):
            obj_id_int = int(obj_id)
            mask_tensor = torch.sigmoid(mask_logits_all[i]).cpu()
            mask = mask_tensor.squeeze().numpy().astype(np.float32)
            if mask.shape[:2] != (self._h, self._w):
                mask = cv2.resize(mask, (self._w, self._h),
                                  interpolation=cv2.INTER_LINEAR)
            if mask.ndim == 3:
                mask = mask.squeeze(axis=0)
            mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
            if mask.max() < 0.01:
                continue

            foot_y = compute_foot_y(mask)
            bbox = compute_bbox_from_mask(mask)
            track_results.append(TrackResult(
                track_id=obj_id_int, bbox=bbox,
                confidence=0.9, mask=mask, foot_y=foot_y,
            ))

        track_results.sort(key=lambda t: t.track_id)
        return track_results

    def reset(self):
        self._predictor = None
        self._inference_state = None
        self._propagator = None

    @property
    def step_name(self) -> str:
        return "处理中"


# ================================================================
# Cutie 引擎 (备选)
# ================================================================

class CutieTracker(BaseVideoTracker):
    """
    Cutie 视频目标分割引擎。
    Cutie: https://github.com/hkchengrex/Cutie
    已 clone 到 vendor/Cutie。
    注意: Cutie 原生仅支持 CUDA, macOS 上会自动尝试 MPS 或 CPU。
    """

    def __init__(self, model_path: str = None, device: str = None,
                 verbose: bool = True, sam2_ckpt: str = None):
        self.model_path = model_path or "weights/cutie-base-mega.pth"
        self._sam2_ckpt = sam2_ckpt  # SAM 2 权重路径 (用于首帧mask精修)
        self.device = device or ("cuda" if torch.cuda.is_available()
                                  else "mps" if torch.backends.mps.is_available()
                                  else "cpu")
        self.verbose = verbose
        self._processor = None
        self._model = None
        self._objects = []
        self._cutie_to_track = {}
        self._w, self._h = 0, 0
        self._first_mask = None

    def _ensure_cutie_path(self):
        vendor = os.path.join(os.path.dirname(__file__), "..", "vendor", "Cutie")
        if os.path.isdir(vendor) and vendor not in sys.path:
            sys.path.insert(0, vendor)

    def _refine_first_frame_sam2(self, first_frame, detections, all_tracked_ids):
        """用 SAM 2 单帧预测器精修首帧 YOLO mask, 改善肢体末端质量"""
        track_set = set(all_tracked_ids)
        if not track_set or not self._sam2_ckpt:
            return {}

        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # SAM 2 加载需要 map_location 补丁 (非 CUDA)
        _original_load = torch.load
        if not torch.cuda.is_available():
            torch.load = lambda *a, **kw: _original_load(*a, **{**kw, 'map_location': 'cpu'})

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2 = build_sam2("sam2_hiera_t.yaml", ckpt_path=self._sam2_ckpt,
                               device=self.device)
            predictor = SAM2ImagePredictor(sam2)
            predictor.set_image(first_frame)

            refined = {}
            for det in detections:
                tid = det.track_id
                if tid not in track_set:
                    continue
                x1, y1, x2, y2 = det.bbox
                masks, scores, _ = predictor.predict(
                    box=np.array([[x1, y1, x2, y2]]), multimask_output=False)
                refined[tid] = masks[0].astype(np.float32)

            del sam2, predictor
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return refined
        finally:
            torch.load = _original_load

    def initialize(self, first_frame: np.ndarray,
                   detections: List[TrackResult],
                   all_tracked_ids: List[int]) -> None:
        self._ensure_cutie_path()

        from cutie.utils.get_default_model import get_default_model
        from cutie.inference.inference_core import InferenceCore

        self._h, self._w = first_frame.shape[:2]
        if self.verbose:
            print(f"[CutieTracker] 初始化, 设备: {self.device}")

        # ★ 引擎内部强制从左到右排序
        detections = sort_detections_left_to_right(detections)

        # SAM 2 精修首帧 mask (改善肢体末端)
        sam2_masks = self._refine_first_frame_sam2(first_frame, detections, all_tracked_ids)
        if self.verbose and sam2_masks:
            print(f"[CutieTracker]   SAM 2 精修了 {len(sam2_masks)} 个首帧 mask")

        # 构建 index mask: 仅画入追踪目标, Cutie obj 从 1 开始连续编号
        combined_mask = np.zeros((self._h, self._w), dtype=np.int32)
        self._cutie_to_track = {}  # cutie_obj → real track_id
        track_set = set(all_tracked_ids)
        cutie_idx = 1
        for det in detections:
            tid = det.track_id
            if tid not in track_set:
                continue
            # 优先使用 SAM 2 精修 mask, 回退到 YOLO mask
            if tid in sam2_masks:
                mask = sam2_masks[tid]
            else:
                mask = det.mask
            binary = (mask > 0.15).astype(np.uint8)
            # ★ 不膨胀 + 不覆盖: 避免相邻人物 mask 互相侵蚀导致手臂被"夺走"
            unassigned = (combined_mask == 0) & (binary > 0)
            combined_mask[unassigned] = cutie_idx
            self._cutie_to_track[cutie_idx] = tid
            self._objects.append(cutie_idx)
            if self.verbose:
                print(f"[CutieTracker]   注册 ID:{tid} → Cutie obj:{cutie_idx}")
            cutie_idx += 1

        if not self._objects:
            raise RuntimeError("Cutie: 没有需要追踪的目标")

        # Cutie 硬编码 .cuda() + autocast + torch.load(CUDA), 需 monkey-patch
        if self.device != "cuda":
            _original_cuda = torch.nn.Module.cuda
            _original_autocast = torch.cuda.amp.autocast
            _original_load = torch.load

            def _patched_cuda(module, device=None):
                return module.to(self.device)
            torch.nn.Module.cuda = _patched_cuda
            torch.cuda.amp.autocast = lambda enabled=True: torch.no_grad()
            torch.load = lambda *a, **kw: _original_load(*a, **{**kw, 'map_location': 'cpu'})

            # 清除 Hydra 全局状态 (多次调用会冲突)
            from hydra.core.global_hydra import GlobalHydra
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()

            try:
                self._model = get_default_model()
                self._model = self._model.to(self.device)
            finally:
                torch.nn.Module.cuda = _original_cuda
                torch.cuda.amp.autocast = _original_autocast
                torch.load = _original_load
        else:
            from hydra.core.global_hydra import GlobalHydra
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            self._model = get_default_model()

        self._processor = InferenceCore(self._model, cfg=self._model.cfg)
        self._processor.max_internal_size = max(self._h, self._w)

        # 首帧: BGR→RGB→tensor→设备
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.to(self.device)
        mask_tensor = torch.from_numpy(combined_mask).to(self.device)

        with torch.no_grad():
            # 去掉 autocast (非 CUDA 不支持)
            self._processor.step(frame_tensor, mask_tensor,
                                  objects=self._objects)

    def step(self, frame: np.ndarray, frame_idx: int) -> List[TrackResult]:
        if self._processor is None:
            return []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.to(self.device)

        with torch.no_grad():
            prob = self._processor.step(frame_tensor)

        if prob is None:
            return []

        # prob: (num_objects, H, W) on device, 强制转为 float32 [0,1]
        prob_np = prob.cpu().numpy().astype(np.float32)

        track_results = []
        for obj_idx in self._objects:
            if obj_idx >= prob_np.shape[0]:
                continue
            # ★ Cutie 的 prob 按 obj_idx 索引, 不是按 enumerate 的 i
            mask = prob_np[obj_idx].astype(np.float32)
            mask = np.clip(mask, 0.0, 1.0)
            if mask.shape[:2] != (self._h, self._w):
                mask = cv2.resize(mask, (self._w, self._h),
                                  interpolation=cv2.INTER_LINEAR)
                mask = np.clip(mask, 0.0, 1.0)
            if mask.max() < 0.01:
                continue

            track_id = self._cutie_to_track.get(obj_idx, 0)
            foot_y = compute_foot_y(mask)
            bbox = compute_bbox_from_mask(mask)
            track_results.append(TrackResult(
                track_id=track_id, bbox=bbox,
                confidence=0.85, mask=mask, foot_y=foot_y,
            ))

        track_results.sort(key=lambda t: t.track_id)
        return track_results

    def reset(self):
        self._processor = None
        self._model = None
        self._objects = []
        self._cutie_to_track = {}

    @property
    def step_name(self) -> str:
        return "处理中"


# ================================================================
# 工厂函数
# ================================================================

def create_tracker(engine_type: str, **kwargs) -> BaseVideoTracker:
    """根据配置创建追踪引擎实例"""
    engines = {
        "sam2": SAM2Tracker,
        "cutie": CutieTracker,
    }
    cls = engines.get(engine_type.lower())
    if cls is None:
        raise ValueError(f"未知引擎: {engine_type}, 可选: {list(engines.keys())}")
    # Cutie 接收 SAM 2 权重的路径 (用于首帧 mask 精修)
    if engine_type.lower() == "cutie" and "model_path" in kwargs:
        kwargs.setdefault("sam2_ckpt", kwargs["model_path"])
    return cls(**kwargs)
