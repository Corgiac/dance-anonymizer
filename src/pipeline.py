"""
Phase 4a: 处理流水线 — YOLO + 可切换追踪引擎
================================================
追踪引擎: SAM 2 (默认) | Cutie (备选)
"""
import os, time, shutil
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .utils import has_audio_stream, merge_audio_with_moviepy, get_video_info
from .tracker import DanceTracker, TrackerConfig, TrackResult
from .effects import process_frame_effects
from .engine import create_tracker


class DanceAnonymizerPipeline:

    def __init__(self, tracker_config: TrackerConfig = TrackerConfig(),
                 effect_config: Optional[dict] = None,
                 engine_config: Optional[dict] = None):
        self.tracker_config = tracker_config
        self.effect_config = effect_config or {}
        self.engine_config = engine_config or {}
        self._tracker: Optional[DanceTracker] = None

    def _init_tracker(self):
        if self._tracker is None:
            self._tracker = DanceTracker(self.tracker_config)

    def process(
        self,
        input_path: str,
        output_path: str,
        target_ids: Optional[List[int]] = None,
        labels_config: Optional[dict] = None,
        precomputed_detections: Optional[List[TrackResult]] = None,
        max_frames: Optional[int] = None,
        show_progress: bool = True,
        cancel_event=None,
        progress_callback=None,
        fill_mode: str = "solid",
        fill_color: str = "#000000",
        border_color: str = "#FFFFFF",
        opacity: float = 1.0,
        label_mode: str = "custom_only",
    ) -> str:
        self._init_tracker()

        info = get_video_info(input_path)
        w, h = info["width"], info["height"]
        fps = info["fps"]
        total_frames = info["total_frames"]
        if show_progress:
            print(f"[Pipeline] 输入: {w}x{h} @ {fps:.2f}fps, {total_frames} 帧")

        # ---- 配置 ----
        dilate_kernel_size = self.effect_config.get("dilate_kernel_size", 3)
        temporal_window = self.effect_config.get("temporal_window", 2)
        engine_type = self.engine_config.get("type", "sam2")
        engine_model = self.engine_config.get("model_path",
                         os.path.join(os.path.dirname(__file__), "..", "sam2_hiera_tiny.pt"))
        if show_progress:
            print(f"[Pipeline] 追踪引擎: {engine_type}")

        # ---- 步骤 1: 抽帧 ----
        task_dir = os.path.join(os.path.dirname(output_path) or "data/output")
        frames_dir = os.path.join(task_dir, "_frames")
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)
        os.makedirs(frames_dir, exist_ok=True)

        if show_progress:
            print("[Pipeline] 步骤 1/4: 抽帧到 JPEG...")
        if progress_callback:
            progress_callback({"step": 1, "step_name": "准备视频", "step_total": 4})

        cap = cv2.VideoCapture(input_path)
        frame_idx = 0
        try:
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(frames_dir, f"{frame_idx:05d}.jpg"),
                            frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_idx += 1
        finally:
            cap.release()

        actual_frames = frame_idx
        if show_progress:
            print(f"[Pipeline]   抽帧: {actual_frames} 张 → {frames_dir}")

        # ---- 步骤 2: 首帧检测 + 引擎初始化 ----
        if show_progress:
            print("[Pipeline] 步骤 2/4: 首帧检测 + 引擎初始化...")
        if progress_callback:
            progress_callback({"step": 2, "step_name": "分析人物", "step_total": 4})

        first_frame_path = os.path.join(frames_dir, "00000.jpg")
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            raise RuntimeError("无法读取首帧")

        # 使用预计算检测结果 (来自 /analyze, 已含 SAM2 精修 mask, 避免重复 YOLO)
        sam2_already_refined = False
        if precomputed_detections is not None:
            detections = precomputed_detections
            sam2_already_refined = True
            if show_progress:
                print(f"[Pipeline]   复用首帧检测: {len(detections)} 人 (含 SAM2 精修)")
        else:
            detections = self._tracker.detect_first_frame(first_frame)
            if show_progress:
                print(f"[Pipeline]   YOLO: {len(detections)} 人")
            # 从左到右排序 + 重分配 ID
            detections.sort(key=lambda t: (t.bbox[0] + t.bbox[2]) / 2.0)
            sorted_dets = []
            for new_id, det in enumerate(detections):
                sorted_dets.append(TrackResult(track_id=new_id, bbox=det.bbox,
                    confidence=det.confidence, mask=det.mask, foot_y=det.foot_y))
            detections = sorted_dets

        # 追踪范围
        all_tracked = set(target_ids) if target_ids else set()
        if labels_config:
            all_tracked.update(int(k) for k in labels_config.keys())
        all_tracked_ids = sorted(all_tracked) if all_tracked else [d.track_id for d in detections]

        # 创建引擎
        engine = create_tracker(engine_type, model_path=engine_model,
                                 verbose=show_progress)

        if engine_type == "sam2":
            # SAM 2: 从帧目录初始化
            engine.initialize_from_dir(frames_dir, first_frame,
                                        detections, all_tracked_ids)
        else:
            # Cutie 等其他引擎: 帧级初始化 (已精修则跳过重复 SAM2)
            if sam2_already_refined and hasattr(engine, '_sam2_ckpt'):
                engine._sam2_ckpt = None  # 禁用 Cutie 自己的 SAM2 精修
            engine.initialize(first_frame, detections, all_tracked_ids)

        # ---- 步骤 3: 追踪 + 渲染 ----
        if show_progress:
            print("[Pipeline] 步骤 3/4: 追踪 + 渲染...")
        if progress_callback:
            progress_callback({
                "step": 3, "step_name": engine.step_name, "step_total": 4,
                "frames_done": 0, "frames_total": actual_frames,
                "elapsed": 0, "eta": 0, "fps": 0,
            })

        tmp_video = output_path + ".tmp_video.mp4"
        writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            # macOS 上 OpenCV 默认构建可能缺少 mp4v 编码器，尝试 avc1 回退
            writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(
                f"无法创建输出视频文件。请检查 OpenCV 编码器支持 (mp4v/avc1)。"
                f"可尝试: pip install opencv-python-headless 或 conda install -c conda-forge opencv")
        smooth_history = {}
        pbar = tqdm(total=actual_frames, desc=engine.step_name, unit="帧",
                     disable=not show_progress)

        processed_count = 0
        start_time = time.time()

        try:
            for f_idx in range(actual_frames):
                if cancel_event is not None and cancel_event.is_set():
                    tqdm.write("  [取消]") if show_progress else None
                    break

                jpg_path = os.path.join(frames_dir, f"{f_idx:05d}.jpg")
                frame = cv2.imread(jpg_path)
                if frame is None:
                    continue

                # 引擎步进 (传副本防止原地修改)
                track_results = engine.step(frame.copy(), f_idx)

                if not track_results:
                    writer.write(frame)
                    processed_count += 1
                    pbar.update(1)
                    continue

                # 特效渲染
                result_frame, smooth_history = process_frame_effects(
                    frame=frame, track_results=track_results,
                    smooth_history=smooth_history, frame_idx=f_idx,
                    dilate_kernel_size=dilate_kernel_size,
                    temporal_window=temporal_window,
                    target_ids=target_ids,
                    fill_mode=fill_mode, fill_color=fill_color,
                    border_color=border_color, opacity=opacity,
                    labels_config=labels_config, label_mode=label_mode,
                )
                writer.write(result_frame)
                processed_count += 1
                pbar.update(1)

                if progress_callback and processed_count % 5 == 0:
                    elapsed = time.time() - start_time
                    fps_proc = processed_count / elapsed if elapsed > 0 else 0
                    eta = (actual_frames - processed_count) / fps_proc if fps_proc > 0 else 0
                    progress_callback({
                        "step": 3, "step_name": engine.step_name, "step_total": 4,
                        "frames_done": processed_count, "frames_total": actual_frames,
                        "elapsed": elapsed, "eta": eta, "fps": fps_proc,
                    })

        finally:
            pbar.close()
            writer.release()

        elapsed = time.time() - start_time
        if show_progress:
            fps_proc = processed_count / elapsed if elapsed > 0 else 0
            print(f"[Pipeline]   渲染完成: {processed_count} 帧 "
                  f"耗时 {elapsed:.1f}s ({fps_proc:.2f} fps)")

        # ---- 步骤 4: 音频 + 清理 ----
        if show_progress:
            print("[Pipeline] 步骤 4/4: 音频合成 + 清理...")
        if progress_callback:
            progress_callback({"step": 4, "step_name": "生成视频", "step_total": 4})

        if has_audio_stream(input_path):
            merge_audio_with_moviepy(tmp_video, input_path, output_path)
            if show_progress:
                print(f"[Pipeline] 成品: {output_path}")
        else:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(tmp_video, output_path)
            if show_progress:
                print(f"[Pipeline] 成品(无音频): {output_path}")

        if os.path.exists(tmp_video):
            try:
                os.remove(tmp_video)
            except OSError:
                pass
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)
            if show_progress:
                print(f"[Pipeline]   已清理: {frames_dir}")

        return output_path

    def reset(self):
        if self._tracker:
            self._tracker.reset()
