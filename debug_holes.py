"""
诊断脚本: 逐帧检查每个人的 alpha / edge / 渲染结果
输出 debug/ 目录下的图片, 帮助定位内部白边形状的来源
用法: python debug_holes.py data/input/vedio2.mp4
"""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tracker import DanceTracker, TrackerConfig
from src.effects import (
    TemporalMaskCache, calculate_depth_order,
    _extract_edge_alpha, _hex_to_bgr,
)

OUT = "debug"
os.makedirs(OUT, exist_ok=True)


def process_and_dump(video_path, frame_numbers=[30, 60, 90, 120]):
    cap = cv2.VideoCapture(video_path)
    tracker = DanceTracker(TrackerConfig(device="cpu", verbose=False))
    cache = TemporalMaskCache(window_size=5, ema_decay=0.3)
    smooth_history = {}

    for fn in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {fn}: read failed")
            continue

        h, w = frame.shape[:2]
        track_results = tracker.track(frame)
        print(f"\n=== Frame {fn}: {len(track_results)} tracks ===")

        # Dump 原始标注
        annotated = frame.copy()
        colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255)]
        for i, tr in enumerate(track_results):
            c = colors[i % len(colors)]
            cv2.rectangle(annotated, (tr.bbox[0], tr.bbox[1]),
                          (tr.bbox[2], tr.bbox[3]), c, 2)
            cv2.putText(annotated, f"ID:{tr.track_id}", (tr.bbox[0], tr.bbox[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        cv2.imwrite(f"{OUT}/frame{fn}_bbox.jpg", annotated)

        # 对每个 track 单独 dump alpha
        for tr in track_results:
            tid = tr.track_id
            alpha = tr.mask.astype(np.float32)
            print(f"  ID:{tid}  bbox={tr.bbox}  alpha_sum={alpha.sum():.0f}  alpha_max={alpha.max():.2f}  alpha_min_gt0={alpha[alpha>0.01].min():.3f}  foot_y={tr.foot_y:.0f}")

            # Raw alpha 热力图
            hot = cv2.applyColorMap((alpha * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(f"{OUT}/frame{fn}_id{tid}_raw_alpha.png", hot)

            # 硬掩码 (alpha > 0.3)
            hard = (alpha > 0.3).astype(np.uint8) * 255
            cv2.imwrite(f"{OUT}/frame{fn}_id{tid}_hard.png", hard)

            # 找所有轮廓 (包括内部)
            fh, fw = frame.shape[:2]
            contours_img = np.zeros((fh, fw, 3), dtype=np.uint8)
            all_contours, hierarchy = cv2.findContours(hard, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contours_img, all_contours, -1, (0, 255, 0), 2)
            # 标出内层轮廓
            if hierarchy is not None:
                for i, hi in enumerate(hierarchy[0]):
                    if hi[3] >= 0:  # 有父轮廓 = 内部洞
                        cv2.drawContours(contours_img, [all_contours[i]], -1, (255, 0, 0), 2)
            cv2.imwrite(f"{OUT}/frame{fn}_id{tid}_contours.png", contours_img)
            if hierarchy is not None:
                inner_count = sum(1 for hi in hierarchy[0] if hi[3] >= 0)
                print(f"    contours: {len(all_contours)} total, {inner_count} inner holes")

        # 深度排序
        depth_order, smooth_history = calculate_depth_order(
            track_results, temporal_window=1, smooth_history=smooth_history)
        tid_to_idx = {t.track_id: i for i, t in enumerate(track_results)}
        print(f"  depth_order (far→near): {depth_order}")

        # 逐人渲染 (模拟 apply_shadow_outline_effect)
        result = frame.copy()

        for tid in depth_order:
            idx = tid_to_idx.get(tid)
            if idx is None: continue
            tr = track_results[idx]
            alpha = tr.mask.astype(np.float32)
            if alpha.max() < 0.01: continue

            cache.update(tid, alpha, fn)
            blended_body = cache.get_blended_body(tid)
            if blended_body is None:
                blended_body = alpha

            blended_edge = cache.get_blended_edge(tid, edge_width=3)
            if blended_edge is None:
                blended_edge = _extract_edge_alpha(blended_body, 3)

            # 保存每个人的 alpha / edge / 混合前帧
            cv2.imwrite(f"{OUT}/frame{fn}_id{tid}_blended.png",
                        cv2.applyColorMap((blended_body*255).astype(np.uint8), cv2.COLORMAP_JET))
            cv2.imwrite(f"{OUT}/frame{fn}_id{tid}_edge.png",
                        (blended_edge*255).astype(np.uint8))

            # 渲染黑底 + 白边
            a = np.clip(blended_body, 0, 1)
            a3 = a[:, :, np.newaxis]
            result = (result.astype(np.float32) * (1 - a3)
                      + np.zeros((1,1,3), dtype=np.float32) * a3).astype(np.uint8)

            e = blended_edge
            if e.max() > 0.001:
                e3 = e[:, :, np.newaxis]
                result = (result.astype(np.float32) * (1 - e3)
                          + 255.0 * e3).astype(np.uint8)

        cv2.imwrite(f"{OUT}/frame{fn}_result.jpg", result)
        print(f"  → result saved: debug/frame{fn}_result.jpg")

    cap.release()
    print(f"\n全部诊断图输出到 {OUT}/ 目录")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/input/vedio2.mp4"
    process_and_dump(path)
