"""
Phase 4b: CLI 入口 (Command Line Interface)
=============================================
舞蹈视频智能打码/特效渲染系统 - 命令行入口

用法:
    # 处理全部人物
    python main.py --input data/input/dance.mp4 --output data/output/result.mp4

    # 只处理指定ID
    python main.py --input data/input/dance.mp4 --output data/output/result.mp4 --target_ids 1,3

    # 自定义白边宽度
    python main.py --input data/input/dance.mp4 --output data/output/result.mp4 --thickness 7

    # 使用CPU推理
    python main.py --input data/input/dance.mp4 --output data/output/result.mp4 --device cpu

    # 使用自定义配置文件
    python main.py --config my_config.yaml --input data/input/dance.mp4 --output data/output/result.mp4

参数说明:
    --input       输入视频路径 (必需)
    --output      输出视频路径 (必需)
    --config      配置文件路径 (默认: config.yaml)
    --target_ids  指定处理的人物ID，逗号分隔 (默认: 全部)
    --thickness   白边宽度，奇数 (默认: 5)
    --device      推理设备 cuda/cpu (默认: cuda)
    --model       YOLO模型路径 (默认: yolov8n-seg.pt)
    --conf        检测置信度阈值 (默认: 0.5)
    --track_buffer  遮挡缓冲帧数 (默认: 60)
    --expected_count  预期人数 (默认: 自动)
"""

import argparse
import os
import sys

import yaml

# 确保 src 在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tracker import TrackerConfig
from src.pipeline import DanceAnonymizerPipeline


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件。"""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="舞蹈视频智能打码/特效渲染系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --input dance.mp4 --output result.mp4
  python main.py --input dance.mp4 --output result.mp4 --target_ids 1,3 --thickness 7
  python main.py --input dance.mp4 --output result.mp4 --device cpu
        """,
    )

    # 必需参数
    parser.add_argument("--input", "-i", required=True,
                        help="输入视频路径")
    parser.add_argument("--output", "-o", required=True,
                        help="输出视频路径")

    # 配置文件
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="配置文件路径 (默认: config.yaml)")

    # 特效参数
    parser.add_argument("--target_ids", "-t", default=None,
                        help="指定处理的人物ID，逗号分隔 (默认: 全部)")
    parser.add_argument("--thickness", type=int, default=5,
                        help="白边宽度 (奇数, 默认: 5)")

    # 模型参数 (默认None=读取config.yaml)
    parser.add_argument("--model", "-m", default=None,
                        help="YOLO模型路径 (默认: config.yaml or yolo11s-seg.pt)")
    parser.add_argument("--device", "-d", default=None,
                        choices=["cuda", "cpu"],
                        help="推理设备 (默认: config.yaml or cuda)")
    parser.add_argument("--conf", type=float, default=None,
                        help="检测置信度阈值 (默认: config.yaml or 0.3)")
    parser.add_argument("--track_buffer", type=int, default=None,
                        help="遮挡缓冲帧数 (默认: config.yaml or 60)")
    parser.add_argument("--expected_count", type=int, default=None,
                        help="预期人数 (默认: 自动)")
    parser.add_argument("--half", action="store_true",
                        help="启用FP16半精度推理 (需CUDA)")

    # 其他
    parser.add_argument("--quiet", action="store_true",
                        help="静默模式 (不显示进度条)")
    parser.add_argument("--blur_ksize", type=int, default=3,
                        help="遮罩预处理高斯核 (奇数, 默认: 3)")
    parser.add_argument("--temporal_window", type=int, default=3,
                        help="时域平滑窗口帧数 (默认: 3)")

    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置文件 (CLI 参数优先级高于配置文件)
    config = load_config(args.config)

    model_cfg = config.get("model", {})
    tracker_cfg = config.get("tracker", {})
    effects_cfg = config.get("effects", {})

    # 构建 TrackerConfig (CLI > config.yaml > 代码默认值)
    tracker_config = TrackerConfig(
        model_path=args.model or model_cfg.get("path") or "yolo11s-seg.pt",
        device=args.device or model_cfg.get("device") or "cuda",
        conf_threshold=args.conf if args.conf is not None else model_cfg.get("conf_threshold", 0.3),
        iou_threshold=model_cfg.get("iou_threshold", 0.35),
        imgsz=model_cfg.get("imgsz", 1280),
        retina_masks=model_cfg.get("retina_masks", True),
        expected_count=args.expected_count or tracker_cfg.get("expected_count"),
        track_buffer=args.track_buffer or tracker_cfg.get("track_buffer", 60),
        half=args.half or model_cfg.get("half", False),
        verbose=not args.quiet,
    )

    # 构建特效配置 (CLI > config.yaml > 代码默认值)
    effect_config = {
        "open_kernel_size": effects_cfg.get("open_kernel_size", 5),
        "body_expand_pixels": effects_cfg.get("body_expand_pixels", 8),
        "dilate_kernel_size": args.thickness or effects_cfg.get("dilate_kernel_size", 5),
        "blur_sigma": effects_cfg.get("blur_sigma", 1.0),
        "temporal_window": args.temporal_window or effects_cfg.get("temporal_window", 8),
    }

    # 解析 target_ids
    target_ids = None
    if args.target_ids:
        target_ids = [int(x.strip()) for x in args.target_ids.split(",")]

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)

    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    # 运行流水线
    pipeline = DanceAnonymizerPipeline(
        tracker_config=tracker_config,
        effect_config=effect_config,
    )

    print(f"{'='*60}")
    print(f"  舞蹈视频智能打码/特效渲染系统")
    print(f"{'='*60}")
    print(f"  输入: {args.input}")
    print(f"  输出: {args.output}")
    print(f"  设备: {args.device}")
    print(f"  模型: {args.model}")
    print(f"  白边宽度: {args.thickness}")
    print(f"  目标ID: {target_ids if target_ids else '全部'}")
    print(f"{'='*60}")

    pipeline.process(
        input_path=args.input,
        output_path=args.output,
        target_ids=target_ids,
        show_progress=not args.quiet,
    )

    print(f"\n处理完成! 输出文件: {args.output}")


if __name__ == "__main__":
    main()
