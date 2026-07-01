"""
Phase 4b: CLI 入口 — YOLO + SAM 2 混合架构
=============================================
用法:
    python main.py --input data/input/dance.mp4 --output data/output/result.mp4
    python main.py --input dance.mp4 --output result.mp4 --target_ids 1,3 --thickness 7
    python main.py --input dance.mp4 --output result.mp4 --device cpu
"""
import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tracker import TrackerConfig
from src.pipeline import DanceAnonymizerPipeline


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="舞蹈视频智能打码/特效渲染系统 (YOLO + SAM 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --input dance.mp4 --output result.mp4
  python main.py --input dance.mp4 --output result.mp4 --target_ids 1,3 --thickness 7
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="输入视频路径")
    parser.add_argument("--output", "-o", required=True, help="输出视频路径")
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")
    parser.add_argument("--target_ids", "-t", default=None,
                        help="指定处理的人物ID, 逗号分隔 (默认: 全部)")
    parser.add_argument("--thickness", type=int, default=5, help="白边宽度 (奇数, 默认: 5)")
    parser.add_argument("--model", "-m", default=None,
                        help="YOLO模型路径 (默认: config.yaml or yolo11s-seg.pt)")
    parser.add_argument("--device", "-d", default=None,
                        choices=["mps", "cuda", "cpu"],
                        help="推理设备 (默认: config.yaml or mps)")
    parser.add_argument("--conf", type=float, default=None,
                        help="检测置信度阈值 (默认: config.yaml or 0.3)")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    parser.add_argument("--temporal_window", type=int, default=3,
                        help="时域平滑窗口帧数 (默认: 3)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    model_cfg = config.get("model", {})
    effects_cfg = config.get("effects", {})
    engine_cfg = config.get("tracking", {})

    tracker_config = TrackerConfig(
        model_path=args.model or model_cfg.get("path") or "yolo11s-seg.pt",
        device=args.device or model_cfg.get("device") or "mps",
        conf_threshold=args.conf if args.conf is not None else model_cfg.get("conf_threshold", 0.3),
        iou_threshold=model_cfg.get("iou_threshold", 0.55),
        imgsz=model_cfg.get("imgsz", 1280),
        verbose=not args.quiet,
    )

    effect_config = {
        "dilate_kernel_size": args.thickness or effects_cfg.get("dilate_kernel_size", 5),
        "temporal_window": args.temporal_window or effects_cfg.get("temporal_window", 8),
    }

    engine_config = {
        "type": engine_cfg.get("engine", "sam2"),
        "model_path": engine_cfg.get("model_path",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "sam2_hiera_tiny.pt")),
    }

    target_ids = None
    if args.target_ids:
        target_ids = [int(x.strip()) for x in args.target_ids.split(",")]

    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    pipeline = DanceAnonymizerPipeline(
        tracker_config=tracker_config,
        effect_config=effect_config,
        engine_config=engine_config,
    )

    print(f"{'='*60}")
    print(f"  舞蹈视频智能打码/特效渲染系统")
    print(f"  架构: YOLO(首帧) + {engine_config['type']}(全片追踪)")
    print(f"{'='*60}")
    print(f"  输入: {args.input}")
    print(f"  输出: {args.output}")
    print(f"  引擎: {engine_config['type']}")
    print(f"  白边: {args.thickness}")
    print(f"  目标: {target_ids if target_ids else '全部'}")
    print(f"{'='*60}")

    pipeline.process(
        input_path=args.input,
        output_path=args.output,
        target_ids=target_ids,
        show_progress=not args.quiet,
    )

    print(f"\n处理完成! 输出: {args.output}")


if __name__ == "__main__":
    main()
