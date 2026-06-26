"""
Phase 4a: 处理流水线 (Processing Pipeline)
=============================================
串联所有模块: VideoReader → DanceTracker → Effects → VideoWriter → Audio Merge

设计原则:
  1. 逐帧处理，绝不将全部帧加载到内存
  2. 帧索引全程对齐，确保追踪状态连续
  3. 异常帧优雅降级 (检测失败时保留原帧)
  4. 最终成品严格保持原始分辨率、帧率、音频同步
"""

import os
import sys
import time
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from .utils import (
    VideoReader, VideoWriter,
    has_audio_stream, merge_audio_with_moviepy,
    get_video_info,
)
from .tracker import DanceTracker, TrackerConfig
from .effects import process_frame_effects, TemporalMaskCache


class DanceAnonymizerPipeline:
    """
    舞蹈视频智能打码处理流水线。

    使用方式:
        pipeline = DanceAnonymizerPipeline(config)
        pipeline.process("input.mp4", "output.mp4", target_ids=[1, 3])
    """

    def __init__(self, tracker_config: TrackerConfig = TrackerConfig(),
                 effect_config: Optional[dict] = None):
        """
        参数:
            tracker_config: 追踪器配置
            effect_config:  特效参数字典
        """
        self.tracker_config = tracker_config
        self.effect_config = effect_config or {}

        # 懒加载: 首次 process() 时初始化
        self._tracker: Optional[DanceTracker] = None

    def _init_tracker(self):
        """初始化追踪器 (懒加载)。"""
        if self._tracker is None:
            self._tracker = DanceTracker(self.tracker_config)

    def process(
        self,
        input_path: str,
        output_path: str,
        target_ids: Optional[List[int]] = None,
        show_progress: bool = True,
        cancel_event = None,              # threading.Event / asyncio.Event 取消标记
        # ---- v3 特效参数 ----
        fill_mode: str = "solid",
        fill_color: str = "#000000",
        border_color: str = "#FFFFFF",
        opacity: float = 1.0,
    ) -> str:
        """执行完整的流水线处理。"""
        self._init_tracker()

        info = get_video_info(input_path)
        if show_progress:
            print(f"[Pipeline] 输入视频: {info['width']}x{info['height']} "
                  f"@ {info['fps']:.2f}fps, {info['total_frames']} 帧")

        smooth_history = {}
        temporal_cache: Optional[TemporalMaskCache] = None
        dilate_kernel_size = self.effect_config.get("dilate_kernel_size", 3)
        temporal_window = self.effect_config.get("temporal_window", 8)

        # 临时视频路径 (无音频)
        tmp_video = output_path + ".tmp_video.mp4"

        # ---- 流水线主循环 ----
        try:
            with VideoReader(input_path) as reader:
                with VideoWriter(
                    tmp_video, reader.fps, reader.frame_size, "mp4v"
                ) as writer:

                    pbar = tqdm(
                        total=reader.total_frames,
                        desc="处理中",
                        unit="帧",
                        disable=not show_progress,
                    )

                    start_time = time.time()
                    processed_count = 0

                    for frame_idx, frame in reader.frames():
                        # ---- 取消检查 ----
                        if cancel_event is not None and cancel_event.is_set():
                            tqdm.write("  [取消] 用户中断, 停止渲染") if show_progress else None
                            break

                        try:
                            # Step 1: 追踪
                            track_results = self._tracker.track(frame)

                            # Step 2: 特效渲染 (v3)
                            result_frame, smooth_history, temporal_cache = \
                                process_frame_effects(
                                    frame=frame,
                                    track_results=track_results,
                                    temporal_cache=temporal_cache,
                                    smooth_history=smooth_history,
                                    frame_idx=frame_idx,
                                    dilate_kernel_size=dilate_kernel_size,
                                    temporal_window=temporal_window,
                                    target_ids=target_ids,
                                    fill_mode=fill_mode,
                                    fill_color=fill_color,
                                    border_color=border_color,
                                    opacity=opacity,
                                )

                            # Step 3: 写入
                            writer.write(result_frame)
                            processed_count += 1

                        except Exception as e:
                            # 异常降级: 保留原帧
                            if show_progress:
                                tqdm.write(
                                    f"  [警告] 第{frame_idx}帧处理异常: {e}, "
                                    f"已降级为原帧"
                                )
                            writer.write(frame)
                            processed_count += 1

                        pbar.update(1)

                    pbar.close()

                    elapsed = time.time() - start_time
                    if show_progress:
                        fps_processing = processed_count / elapsed if elapsed > 0 else 0
                        print(f"[Pipeline] 处理完成: {processed_count} 帧 "
                              f"耗时 {elapsed:.1f}s ({fps_processing:.2f} fps)")

            # ---- 音频合并 ----
            if show_progress:
                print("[Pipeline] 检查音频流...")

            if has_audio_stream(input_path):
                if show_progress:
                    print("[Pipeline] 合成音视频 (moviepy)...")
                merge_audio_with_moviepy(tmp_video, input_path, output_path)
                if show_progress:
                    print(f"[Pipeline] 最终成品: {output_path}")
            else:
                # 无音频: 直接重命名
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(tmp_video, output_path)
                if show_progress:
                    print(f"[Pipeline] 最终成品 (无音频): {output_path}")

        finally:
            # 清理临时视频
            if os.path.exists(tmp_video):
                try:
                    os.remove(tmp_video)
                except OSError:
                    pass

        return output_path

    def reset(self):
        """重置所有内部状态。"""
        if self._tracker:
            self._tracker.reset()
