"""
Phase 1: 视频 I/O 模块 (Video I/O Module)
============================================
- 视频帧提取 (逐帧读取，避免全量加载)
- 视频帧写入 (逐帧写出，即时释放)
- 原始音频提取
- 音视频同步合成

设计原则:
  1. 逐帧处理，绝不将全部帧加载到内存
  2. 使用 with 上下文管理器确保资源自动释放
  3. 输出视频严格保持原始分辨率、帧率、编码参数
  4. 音频与视频完全同步，无时差
"""

import os
import cv2
import numpy as np
from typing import Iterator, Optional, Tuple


class VideoReader:
    """
    逐帧视频读取器，支持上下文管理器。

    使用方式:
        with VideoReader("input.mp4") as reader:
            for frame in reader.frames():
                process(frame)
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._total_frames: int = 0
        self._width: int = 0
        self._height: int = 0
        self._fourcc: str = ""

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise IOError(f"无法打开视频文件: {self.video_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        self._fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        return self

    def __exit__(self, *args):
        if self._cap is not None:
            self._cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self._width, self._height)

    @property
    def fourcc(self) -> str:
        return self._fourcc

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        逐帧迭代器，返回 (帧索引, BGR图像)。

        内存优化: 每次 yield 后可立即处理，不累积帧数据。
        """
        idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """随机访问指定帧 (非高性能，调试用)。"""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()
        return frame if ret else None


class VideoWriter:
    """
    逐帧视频写出器，支持上下文管理器。

    使用方式:
        with VideoWriter("output.mp4", fps=30, size=(1920, 1080)) as writer:
            for frame in processed_frames:
                writer.write(frame)
    """

    def __init__(self, video_path: str, fps: float, size: Tuple[int, int],
                 fourcc: str = "mp4v"):
        self.video_path = video_path
        self.fps = fps
        self.size = size
        self.fourcc = fourcc

        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        self._writer: Optional[cv2.VideoWriter] = None

    def __enter__(self) -> "VideoWriter":
        fourcc_int = cv2.VideoWriter_fourcc(*self.fourcc)
        self._writer = cv2.VideoWriter(
            self.video_path, fourcc_int, self.fps, self.size
        )
        if not self._writer.isOpened():
            raise IOError(f"无法创建输出视频: {self.video_path}")
        return self

    def __exit__(self, *args):
        if self._writer is not None:
            self._writer.release()

    def write(self, frame: np.ndarray):
        """写入单帧。frame 必须为 BGR uint8 且尺寸匹配构造函数。"""
        self._writer.write(frame)


def _get_ffmpeg():
    """获取 ffmpeg 可执行文件路径 (优先系统安装版本，兼容性更好)。"""
    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def has_audio_stream(video_path: str) -> bool:
    """检测视频是否包含音频流 (使用 ffmpeg -i 输出解析)。"""
    import subprocess
    try:
        result = subprocess.run(
            [_get_ffmpeg(), "-i", video_path],
            capture_output=True, text=True, timeout=30)
        return "Audio:" in result.stderr
    except Exception:
        return False


def merge_audio_with_moviepy(
    video_path: str,
    audio_source_path: str,
    output_path: str
) -> str:
    """
    使用 ffmpeg 将无声视频与原始音频合成为最终成品。
    """
    import subprocess
    ffmpeg = _get_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", video_path,
        "-i", audio_source_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, timeout=300)
    return output_path


def get_video_info(video_path: str) -> dict:
    """获取视频基本信息 (调试/验证用)。"""
    with VideoReader(video_path) as reader:
        return {
            "path": video_path,
            "width": reader.width,
            "height": reader.height,
            "fps": reader.fps,
            "total_frames": reader.total_frames,
            "fourcc": reader.fourcc,
        }
