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
        follow_id: Optional[int] = None,
        crop_params: Optional[dict] = None,
        face_blur_ids: Optional[List[int]] = None,
        sticker_img: Optional[np.ndarray] = None,
        sticker_scale: float = 0.40,
        skin_whiten: int = 0,
        leg_stretch_on: bool = False,
        leg_stretch: int = 0,
        leg_zone_top: float = 0.50,
        leg_zone_bot: float = 0.75,
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

        # ---- 步骤 1: 获取首帧 + 抽帧(仅SAM2) ----
        if show_progress:
            print("[Pipeline] 步骤 1/4: 读取视频...")
        if progress_callback:
            progress_callback({"step": 1, "step_name": "准备视频", "step_total": 4})

        cap = cv2.VideoCapture(input_path)
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            cap.release()
            raise RuntimeError("无法读取首帧")

        frames_dir = None
        actual_frames = total_frames
        if max_frames is not None and max_frames < total_frames:
            actual_frames = max_frames

        if engine_type == "sam2":
            task_dir = os.path.join(os.path.dirname(output_path) or "data/output")
            frames_dir = os.path.join(task_dir, "_frames")
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir, ignore_errors=True)
            os.makedirs(frames_dir, exist_ok=True)
            cv2.imwrite(os.path.join(frames_dir, "00000.jpg"),
                        first_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_idx = 1
            while True:
                if max_frames is not None and frame_idx >= max_frames: break
                ret, frame = cap.read()
                if not ret: break
                cv2.imwrite(os.path.join(frames_dir, f"{frame_idx:05d}.jpg"),
                            frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_idx += 1
            cap.release()
            actual_frames = frame_idx
            cap = None
            if show_progress:
                print(f"[Pipeline]   抽帧: {actual_frames} 张 → {frames_dir}")
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap.read()
            if show_progress:
                print(f"[Pipeline]   Cutie 流式处理, 共 {actual_frames} 帧")

        # ---- 步骤 2: 首帧检测 + 引擎初始化 ----
        if show_progress:
            print("[Pipeline] 步骤 2/4: 首帧检测 + 引擎初始化...")
        if progress_callback:
            progress_callback({"step": 2, "step_name": "分析人物", "step_total": 4})

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
            detections.sort(key=lambda t: (t.bbox[0] + t.bbox[2]) / 2.0)
            sorted_dets = []
            for new_id, det in enumerate(detections):
                sorted_dets.append(TrackResult(track_id=new_id, bbox=det.bbox,
                    confidence=det.confidence, mask=det.mask, foot_y=det.foot_y))
            detections = sorted_dets

        all_tracked_ids = [d.track_id for d in detections]

        # 跟随模式参数
        follow_enabled = follow_id is not None and follow_id >= 0
        smooth_x, smooth_y = w / 2, h / 2
        follow_y_offset = 0.12
        crop_w, crop_h = int(w * 0.60), int(h * 0.90)
        if crop_w % 2: crop_w -= 1
        if crop_h % 2: crop_h -= 1
        if crop_params:
            cp = crop_params; out_w, out_h = cp["w"], cp["h"]
        else:
            out_w, out_h = (crop_w, crop_h) if follow_enabled else (w, h)

        # 创建引擎
        engine = create_tracker(engine_type, model_path=engine_model,
                                 verbose=show_progress)
        engine._max_input_dim = self.effect_config.get("max_input_dim", 960)

        # 跟随模式：首帧也裁切
        init_frame = first_frame
        if follow_enabled:
            cx, cy = int(smooth_x), int(smooth_y)
            x1i = max(0, min(w - crop_w, cx - crop_w // 2))
            y1i = max(0, min(h - crop_h, cy - crop_h // 2))
            init_frame = first_frame[y1i:y1i + crop_h, x1i:x1i + crop_w]
            adjusted_dets = []
            for det in detections:
                new_bbox = (
                    max(0, det.bbox[0] - x1i),
                    max(0, det.bbox[1] - y1i),
                    min(crop_w, det.bbox[2] - x1i),
                    min(crop_h, det.bbox[3] - y1i),
                )
                mask_cropped = det.mask[y1i:y1i + crop_h, x1i:x1i + crop_w]
                adjusted_dets.append(TrackResult(
                    track_id=det.track_id, bbox=new_bbox,
                    confidence=det.confidence, mask=mask_cropped, foot_y=det.foot_y))
            detections = adjusted_dets

        if engine_type == "sam2":
            engine.initialize_from_dir(frames_dir, init_frame,
                                        detections, all_tracked_ids)
        else:
            if sam2_already_refined and hasattr(engine, '_sam2_ckpt'):
                engine._sam2_ckpt = None
            engine.initialize(init_frame, detections, all_tracked_ids)

        # ---- 步骤 3: 追踪 + 渲染 ----
        if show_progress:
            print("[Pipeline] 步骤 3/4: 追踪 + 渲染...")
        if progress_callback:
            progress_callback({
                "step": 3, "step_name": engine.step_name, "step_total": 4,
                "frames_done": 0, "frames_total": actual_frames,
                "elapsed": 0, "eta": 0, "fps": 0,
            })

        # PNG 序列临时目录（高质量无损保存，最后 ffmpeg 编码）
        encode_dir = os.path.join(os.path.dirname(output_path), "_encode_frames")
        os.makedirs(encode_dir, exist_ok=True)
        smooth_history = {}
        _yolo_persons = {}
        processed_count = 0
        start_time = time.time()
        frame_skip = self.effect_config.get("frame_skip", 1)

        smooth_factor = 0.40
        prev_x1, prev_y1 = 0, 0

        pbar = None
        try:
            pbar = tqdm(total=actual_frames, desc=engine.step_name, unit="帧",
                         disable=not show_progress)

            for f_idx in range(actual_frames):
                if cancel_event is not None and cancel_event.is_set():
                    tqdm.write("  [取消]") if show_progress else None
                    break

                if cap is not None:
                    ret, frame = cap.read()
                    if not ret: break
                else:
                    jpg_path = os.path.join(frames_dir, f"{f_idx:05d}.jpg")
                    frame = cv2.imread(jpg_path)
                    if frame is None: continue

                # 先跟随裁切，再 CUTIE/效果
                work_frame = frame
                if follow_enabled:
                    if f_idx > 0 and track_results:
                        follow_track = next((t for t in track_results if t.track_id == follow_id), None)
                        if follow_track is not None:
                            bx1, by1, bx2, by2 = follow_track.bbox
                            target_x = prev_x1 + (bx1 + bx2) / 2
                            target_y = prev_y1 + (by1 + by2) / 2 - h * follow_y_offset
                            smooth_x += (target_x - smooth_x) * smooth_factor
                            smooth_y += (target_y - smooth_y) * smooth_factor
                    cx, cy = int(smooth_x), int(smooth_y)
                    prev_x1 = max(0, min(w - crop_w, cx - crop_w // 2))
                    prev_y1 = max(0, min(h - crop_h, cy - crop_h // 2))
                    work_frame = frame[prev_y1:prev_y1 + crop_h, prev_x1:prev_x1 + crop_w]

                # 跳帧
                if frame_skip > 0 and f_idx > 0 and f_idx % (frame_skip + 1) != 0:
                    pass
                else:
                    track_results = engine.step(work_frame.copy(), f_idx)

                # 定期 YOLO 检测新人
                if f_idx > 0 and f_idx % 30 == 0 and self._tracker is not None:
                    try:
                        new_raw = self._tracker.detect_first_frame(work_frame.copy())
                        all_ids = set(tr.track_id for tr in (track_results or []))
                        all_ids.update(_yolo_persons.keys())
                        next_id = max(all_ids, default=-1) + 1
                        existing_bboxes = [(tr.bbox, tr.track_id) for tr in (track_results or [])]
                        existing_bboxes += [(p['mask_track'].bbox, tid) for tid, p in _yolo_persons.items()]
                        for nr in new_raw:
                            bx1, by1, bx2, by2 = nr.bbox
                            is_new = True
                            for (ox1, oy1, ox2, oy2), _ in existing_bboxes:
                                inter_w = max(0, min(bx2, ox2) - max(bx1, ox1))
                                inter_h = max(0, min(by2, oy2) - max(by1, oy1))
                                inter = inter_w * inter_h
                                area_n = (bx2-bx1)*(by2-by1)
                                area_o = (ox2-ox1)*(oy2-oy1)
                                iou = inter / (area_n + area_o - inter + 1e-6)
                                if iou > 0.3:
                                    is_new = False
                                    break
                            if is_new:
                                nr.track_id = next_id
                                next_id += 1
                                _yolo_persons[nr.track_id] = {'mask_track': nr, 'last_seen': f_idx}
                                all_tracked_ids.append(nr.track_id)
                                if show_progress:
                                    tqdm.write(f"  [新人] ID:{nr.track_id} 入镜 (帧 {f_idx})")
                        gone = [tid for tid, p in _yolo_persons.items() if f_idx - p['last_seen'] > 60]
                        for tid in gone:
                            del _yolo_persons[tid]
                            if tid in all_tracked_ids:
                                all_tracked_ids.remove(tid)
                    except Exception:
                        pass

                if _yolo_persons:
                    track_results = list(track_results or [])
                    for tid, p in _yolo_persons.items():
                        track_results.append(p['mask_track'])

                if not track_results:
                    cv2.imwrite(os.path.join(encode_dir, f"{processed_count:06d}.png"), work_frame)
                    processed_count += 1
                    pbar.update(1)
                    continue

                # 特效渲染
                result_frame, smooth_history = process_frame_effects(
                    frame=work_frame, track_results=track_results,
                    smooth_history=smooth_history, frame_idx=f_idx,
                    dilate_kernel_size=dilate_kernel_size,
                    temporal_window=temporal_window,
                    target_ids=target_ids,
                    fill_mode=fill_mode, fill_color=fill_color,
                    border_color=border_color, opacity=opacity,
                    labels_config=labels_config, label_mode=label_mode,
                    face_blur_ids=face_blur_ids,
                    sticker_img=sticker_img,
                    sticker_scale=sticker_scale,
                    skin_whiten=skin_whiten,
                    leg_stretch_on=leg_stretch_on,
                    leg_stretch=leg_stretch,
                    leg_zone_top=leg_zone_top,
                    leg_zone_bot=leg_zone_bot,
                )
                if result_frame.shape[0] != out_h:
                    result_frame = cv2.resize(result_frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(os.path.join(encode_dir, f"{processed_count:06d}.png"), result_frame)
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
            if pbar is not None:
                pbar.close()

        elapsed = time.time() - start_time
        if show_progress:
            fps_proc = processed_count / elapsed if elapsed > 0 else 0
            print(f"[Pipeline]   渲染完成: {processed_count} 帧 "
                  f"耗时 {elapsed:.1f}s ({fps_proc:.2f} fps)")

        # ---- 步骤 4: ffmpeg 编码高质量 MP4 + 音频 ----
        if show_progress:
            print("[Pipeline] 步骤 4/4: 编码视频 + 音频合成...")
        if progress_callback:
            progress_callback({"step": 4, "step_name": "生成视频", "step_total": 4})

        # 用 ffmpeg 从 PNG 序列编码高质量 MP4
        tmp_video = output_path + ".tmp_video.mp4"
        self._encode_png_sequence(encode_dir, tmp_video, fps, out_w, out_h, show_progress)
        # 清理 PNG 缓存
        shutil.rmtree(encode_dir, ignore_errors=True)

        has_audio = has_audio_stream(input_path)
        if has_audio:
            merge_audio_with_moviepy(tmp_video, input_path, output_path)
            if show_progress:
                print(f"[Pipeline] 成品(含音频): {output_path}")
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
        if frames_dir and os.path.exists(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)
            if show_progress:
                print(f"[Pipeline]   已清理: {frames_dir}")

        return output_path

    @staticmethod
    def _encode_png_sequence(frames_dir, output_path, fps, width, height, show_progress):
        """用 ffmpeg 将 PNG 序列编码为高质量 MP4。"""
        import subprocess
        from .utils import _get_ffmpeg
        ffmpeg = _get_ffmpeg()
        cmd = [
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "%06d.png"),
            "-c:v", "libx264",
            "-crf", "16",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={width}:{height}:flags=lanczos",
            output_path
        ]
        if show_progress:
            print(f"[Pipeline]   ffmpeg 编码: {width}x{height} CRF 16")
        subprocess.run(cmd, capture_output=True, timeout=600)

    def reset(self):
        if self._tracker:
            self._tracker.reset()
