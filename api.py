"""
舞蹈视频智能打码/特效渲染系统 — FastAPI v5 (YOLO+SAM2)
=========================================================
交互: 上传 → 裁剪时长 → 调参预览 → 全片渲染
启动: uvicorn api:app --host 0.0.0.0 --port 8002
"""
import os, sys, uuid, json, base64, shutil, cv2, threading, asyncio
import numpy as np
from typing import Dict

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "sam2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "Cutie"))

from src.tracker import DanceTracker, TrackerConfig, TrackResult, auto_device
from src.pipeline import DanceAnonymizerPipeline
from src.effects import (
    process_frame_effects, calculate_depth_order,
    apply_shadow_outline_effect, draw_text_labels,
)
from src.utils import get_video_info

app = FastAPI(title="舞蹈视频智能打码 API v5", version="5.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(BASE_DIR, "data", "tasks")
os.makedirs(TASKS_DIR, exist_ok=True)

TASKS: Dict[str, dict] = {}
PROGRESS: Dict[str, dict] = {}
CANCEL_EVENTS: Dict[str, threading.Event] = {}


# ================================================================
#  /cancel/{task_id} — 取消进行中的渲染
# ================================================================

@app.post("/cancel/{task_id}")
async def cancel_render(task_id: str):
    evt = CANCEL_EVENTS.pop(task_id, None)
    if evt:
        evt.set()
        PROGRESS[task_id] = {"done": True, "running": False}
        return {"ok": True, "cancelled": True}
    return {"ok": True, "cancelled": False}


# ================================================================
#  /status/{task_id} — 轮询渲染进度
# ================================================================

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    p = PROGRESS.get(task_id, {})
    return {
        "step": p.get("step", 0),
        "step_name": p.get("step_name", ""),
        "step_total": p.get("step_total", 5),
        "frames_done": p.get("frames_done", 0),
        "frames_total": p.get("frames_total", 0),
        "elapsed": p.get("elapsed", 0),
        "eta": p.get("eta", 0),
        "fps": p.get("fps", 0),
        "done": p.get("done", False),
        "running": p.get("running", False),
        "error": p.get("error", ""),
    }


# ================================================================
#  /cleanup/{task_id}
# ================================================================

@app.delete("/cleanup/{task_id}")
async def cleanup(task_id: str):
    task = TASKS.pop(task_id, None)
    if task:
        task_dir = os.path.dirname(task["video_path"])
        shutil.rmtree(task_dir, ignore_errors=True)
    PROGRESS.pop(task_id, None)
    return {"ok": True}


# ================================================================
#  渲染辅助
# ================================================================

def _render_one_frame(frame, track_results, target_ids, params, labels_config=None, face_blur_ids=None, sticker_img=None, sticker_scale=0.40):
    """预览模式: 打码仅 target_ids, 标签覆盖所有人(自定义昵称或默认ID)。"""
    all_track_results = track_results

    # 美白
    if params.get("skin_whiten", 0) > 0:
        from src.effects import apply_skin_whiten
        frame = apply_skin_whiten(frame, all_track_results, params["skin_whiten"])

    # 面部贴纸
    if face_blur_ids:
        from src.effects import apply_face_blur
        frame = apply_face_blur(frame, all_track_results, face_blur_ids,
                                sticker_img=sticker_img, sticker_scale=sticker_scale)

    if target_ids:
        anonymize = [t for t in track_results if t.track_id in set(target_ids)]
    else:
        anonymize = track_results

    if anonymize:
        ordered_ids, _ = calculate_depth_order(anonymize, temporal_window=1)
        tid_to_idx = {t.track_id: i for i, t in enumerate(anonymize)}
        result = apply_shadow_outline_effect(
            frame=frame, depth_order=ordered_ids,
            track_id_to_idx=tid_to_idx, track_results=anonymize,
            dilate_kernel_size=params.get("thickness", 3),
            fill_mode=params.get("fill_mode", "solid"),
            fill_color=params.get("fill_color", "#000000"),
            border_color=params.get("border_color", "#FFFFFF"),
            opacity=params.get("opacity", 1.0),
        )
    else:
        result = frame.copy()

    result = draw_text_labels(result, all_track_results, labels_config, label_mode="all")

    # 拉腿：最后一步
    if params.get("leg_stretch_on") and params.get("leg_stretch", 0) > 0:
        from src.effects import apply_leg_stretch
        result = apply_leg_stretch(result, params["leg_zone_top"], params["leg_zone_bot"], params["leg_stretch"])

    return result


def _img_to_base64(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"


def _parse_sticker(sticker_data: str):
    """将 base64 贴纸数据解码为 OpenCV BGR/A 图像，失败返回 None。"""
    if not sticker_data or not sticker_data.startswith("data:image/"):
        return None
    try:
        img_data = base64.b64decode(sticker_data.split(",", 1)[1])
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        # cv2.imdecode 返回 BGR/BGRA，无需颜色转换
        return img
    except Exception:
        return None


DEFAULT_STICKER_PATH = os.path.join(BASE_DIR, "assets", "sticker-1.png")
_default_sticker_cache = None


def _get_default_sticker():
    """加载默认贴纸，缓存避免重复 IO。cv2.imread 返回 BGR/BGRA。"""
    global _default_sticker_cache
    if _default_sticker_cache is None:
        if os.path.exists(DEFAULT_STICKER_PATH):
            _default_sticker_cache = cv2.imread(DEFAULT_STICKER_PATH, cv2.IMREAD_UNCHANGED)
    return _default_sticker_cache


def _parse_ids(target_ids_str):
    if target_ids_str and target_ids_str.strip():
        return [int(x.strip()) for x in target_ids_str.split(",") if x.strip()]
    return []


# ================================================================
#  / — 前端
# ================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    path = os.path.join(BASE_DIR, "index.html")
    return HTMLResponse(open(path, encoding="utf-8").read())


# ================================================================
#  /analyze
# ================================================================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), trim_start: float = Form(0), trim_end: float = Form(0)):
    task_id = uuid.uuid4().hex[:10]
    task_dir = os.path.join(TASKS_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    video_path = os.path.join(task_dir, f"source{ext}")

    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 时长裁剪：用 ffmpeg 截取 trim_start ~ trim_end
        if trim_end > 0 and trim_end - trim_start > 0.5:
            from src.utils import _get_ffmpeg
            trimmed = os.path.join(task_dir, f"trimmed{ext}")
            import subprocess
            subprocess.run([
                _get_ffmpeg(), "-y", "-ss", str(trim_start), "-to", str(trim_end),
                "-i", video_path, "-c", "copy", trimmed
            ], capture_output=True, timeout=60)
            if os.path.exists(trimmed) and os.path.getsize(trimmed) > 1000:
                os.remove(video_path)
                os.rename(trimmed, video_path)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, best_frame = cap.read()
        cap.release()

        if not ret or best_frame is None:
            return JSONResponse({"error": "无法读取视频帧"}, 400)
        if fps <= 0: fps = 30.0
        if total_frames <= 0: return JSONResponse({"error": "视频文件损坏，无法读取帧信息！"}, 400)

        tracker = DanceTracker(TrackerConfig(device=auto_device(), verbose=False))
        raw_results = tracker.detect_first_frame(best_frame)
        raw_results.sort(key=lambda t: (t.bbox[0] + t.bbox[2]) / 2.0)

        # SAM 2 精修首帧 mask
        sam2_masks = {}
        try:
            import torch as _torch
            _orig_load = _torch.load
            if not _torch.cuda.is_available():
                _torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'map_location': 'cpu'})
            from hydra.core.global_hydra import GlobalHydra
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2_ckpt = os.path.join(BASE_DIR, "sam2_hiera_tiny.pt")
            sam2 = build_sam2("sam2_hiera_t.yaml", ckpt_path=sam2_ckpt, device=auto_device())
            predictor = SAM2ImagePredictor(sam2)
            predictor.set_image(best_frame)
            for tr in raw_results:
                x1, y1, x2, y2 = tr.bbox
                masks, _, _ = predictor.predict(
                    box=np.array([[x1, y1, x2, y2]]), multimask_output=False)
                sam2_masks[tr.track_id] = masks[0].astype(np.float32)
            del sam2, predictor
            if _torch.backends.mps.is_available():
                _torch.mps.empty_cache()
            _torch.load = _orig_load
        except Exception as e:
            pass  # SAM 2 精修跳过

        track_results = []
        for new_id, tr in enumerate(raw_results):
            mask = sam2_masks.get(tr.track_id, tr.mask)
            track_results.append(TrackResult(
                track_id=new_id, bbox=tr.bbox,
                confidence=tr.confidence, mask=mask, foot_y=tr.foot_y,
            ))
        if not track_results:
            return JSONResponse({"error": "未检测到人物，请确保首帧画面中有人物出现"}, 400)
        key_frame = best_frame
        available_ids = sorted([t.track_id for t in track_results])

        default_params = {"fill_mode": "blur", "fill_color": "#000000",
                           "border_color": "#FFFFFF", "opacity": 1.0, "thickness": 3}
        rendered = _render_one_frame(key_frame, track_results, available_ids, default_params)

        TASKS[task_id] = {
            "video_path": video_path, "ext": ext,
            "key_frame": key_frame.copy(),
            "track_results": track_results,
            "available_ids": available_ids,
            "total_frames": total_frames,
            "fps": fps,
        }

        return {
            "task_id": task_id,
            "image_base64": _img_to_base64(rendered),
            "available_ids": available_ids,
            "total_frames": total_frames, "fps": fps,
        }
    except Exception as e:
        shutil.rmtree(task_dir, ignore_errors=True)
        return JSONResponse({"error": str(e)}, 500)


# ================================================================
#  /preview_frame
# ================================================================

@app.post("/preview_frame")
async def preview_frame(
    task_id: str = Form(...),
    target_ids: str = Form(""),
    labels_config: str = Form(""),
    fill_mode: str = Form("solid"),
    fill_color: str = Form("#000000"),
    border_color: str = Form("#FFFFFF"),
    thickness: int = Form(3),
    opacity: float = Form(1.0),
    face_blur_ids: str = Form(""),
    sticker_data: str = Form(""),
    face_mode: str = Form("blur"),
    sticker_scale: float = Form(0.40),
    skin_whiten: int = Form(0),
    leg_stretch_on: str = Form("false"),
    leg_stretch: int = Form(0),
    leg_zone_top: float = Form(0.50),
    leg_zone_bot: float = Form(0.75),
):
    task = TASKS.get(task_id)
    if not task:
        return JSONResponse({"error": "task_id 无效"}, 404)
    parsed_ids = _parse_ids(target_ids)
    labels_cfg = json.loads(labels_config) if labels_config.strip() else None
    face_ids = _parse_ids(face_blur_ids)
    # 解析贴纸：优先用户上传，否则用默认贴纸
    sticker_img = _parse_sticker(sticker_data)
    if sticker_img is None and face_mode == "sticker":
        sticker_img = _get_default_sticker()
    try:
        params = {"fill_mode": fill_mode, "fill_color": fill_color,
                   "border_color": border_color, "opacity": opacity, "thickness": thickness,
                   "skin_whiten": skin_whiten,
                   "leg_stretch_on": leg_stretch_on == "true",
                   "leg_stretch": leg_stretch,
                   "leg_zone_top": leg_zone_top,
                   "leg_zone_bot": leg_zone_bot}
        rendered = _render_one_frame(task["key_frame"], task["track_results"],
                                      parsed_ids, params, labels_cfg,
                                      face_blur_ids=face_ids,
                                      sticker_img=sticker_img,
                                      sticker_scale=sticker_scale)
        return {"image_base64": _img_to_base64(rendered)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


#  /render (全片渲染, 异步可中断)
# ================================================================

@app.post("/render")
async def render(
    request: Request,
    task_id: str = Form(...),
    target_ids: str = Form(""),
    labels_config: str = Form(""),
    fill_mode: str = Form("solid"),
    fill_color: str = Form("#000000"),
    border_color: str = Form("#FFFFFF"),
    thickness: int = Form(3),
    opacity: float = Form(1.0),
    device: str = Form(""),
    follow_id: str = Form("-1"),
    face_blur_ids: str = Form(""),
    sticker_data: str = Form(""),
    face_mode: str = Form("blur"),
    sticker_scale: float = Form(0.40),
    skin_whiten: int = Form(0),
    leg_stretch_on: str = Form("false"),
    leg_stretch: int = Form(0),
    leg_zone_top: float = Form(0.50),
    leg_zone_bot: float = Form(0.75),
):
    task = TASKS.get(task_id)
    if not task:
        return JSONResponse({"error": "task_id 无效"}, 404)
    parsed_ids = _parse_ids(target_ids)
    labels_cfg = json.loads(labels_config) if labels_config.strip() else None
    if not parsed_ids:
        return JSONResponse({"error": "没有选中任何人"}, 400)

    follow_pid = int(follow_id) if follow_id and follow_id != "-1" else None
    face_ids = _parse_ids(face_blur_ids)
    sticker_img = _parse_sticker(sticker_data)
    if sticker_img is None and face_mode == "sticker":
        sticker_img = _get_default_sticker()

    output_path = os.path.join(os.path.dirname(task["video_path"]), "result.mp4")

    if PROGRESS.get(task_id, {}).get("running") and not PROGRESS[task_id].get("done"):
        return JSONResponse({"status": "running", "task_id": task_id})

    cancel_event = threading.Event()
    CANCEL_EVENTS[task_id] = cancel_event
    gen = PROGRESS.get(task_id, {}).get("generation", 0) + 1
    PROGRESS[task_id] = {"done": False, "running": True, "generation": gen,
                         "task_id": task_id, "output_path": output_path}

    def _on_progress(p):
        if PROGRESS.get(task_id, {}).get("generation") == gen:
            PROGRESS[task_id] = {**p, "done": False, "running": True, "generation": gen,
                                 "task_id": task_id, "output_path": output_path}

    try:
        pipeline = DanceAnonymizerPipeline(
            tracker_config=TrackerConfig(model_path="yolo11s-seg.pt",
                                          device=device or auto_device(), conf_threshold=0.3,
                                          verbose=False),
            effect_config={"dilate_kernel_size": max(1, min(thickness, 15))},
            engine_config={"type": "cutie", "model_path": "sam2_hiera_tiny.pt"},
        )

        result = {}
        def _run():
            try:
                pipeline.process(
                    input_path=task["video_path"], output_path=output_path,
                    target_ids=parsed_ids, labels_config=labels_cfg,
                    precomputed_detections=task.get("track_results"),
                    show_progress=False, cancel_event=cancel_event,
                    progress_callback=_on_progress,
                    fill_mode=fill_mode, fill_color=fill_color,
                    border_color=border_color, opacity=opacity,
                    follow_id=follow_pid,
                    face_blur_ids=face_ids,
                    sticker_img=sticker_img,
                    sticker_scale=sticker_scale,
                    skin_whiten=skin_whiten,
                    leg_stretch_on=leg_stretch_on == "true",
                    leg_stretch=leg_stretch,
                    leg_zone_top=leg_zone_top,
                    leg_zone_bot=leg_zone_bot,
                )
                result["status"] = "done"
                if PROGRESS.get(task_id, {}).get("generation") == gen:
                    PROGRESS[task_id] = {"done": True, "running": False, "generation": gen,
                        "task_id": task_id, "output_path": output_path}
            except Exception as e:
                err_msg = str(e)
                result["error"] = err_msg
                if PROGRESS.get(task_id, {}).get("generation") == gen:
                    PROGRESS[task_id] = {"done": True, "running": False, "generation": gen,
                        "task_id": task_id, "output_path": output_path,
                        "error": err_msg}

        thread = threading.Thread(target=_run)
        thread.start()

        return JSONResponse({"status": "started", "task_id": task_id})

    except Exception as e:
        cancel_event.set()
        if PROGRESS.get(task_id, {}).get("generation") == gen:
            PROGRESS[task_id] = {"done": True, "running": False, "generation": gen,
                                 "task_id": task_id, "output_path": output_path}
        return JSONResponse({"error": str(e)}, 500)


# ================================================================
#  /crop_result/{task_id} — 对成品视频应用画幅裁剪
# ================================================================

@app.post("/crop_result/{task_id}")
async def crop_result(task_id: str, crop_mode: str = Form(...), crop_offset: float = Form(0)):
    task = TASKS.get(task_id)
    if not task:
        return JSONResponse({"error": "task_id 无效"}, 404)
    result_path = os.path.join(os.path.dirname(task["video_path"]), "result.mp4")
    if not os.path.exists(result_path):
        return JSONResponse({"error": "成品视频不存在，请先生成视频"}, 404)

    RATIOS = {"1:1": 1.0, "4:3": 4/3, "16:9": 16/9, "9:16": 9/16}
    tr = RATIOS.get(crop_mode)
    if not tr:
        return JSONResponse({"error": f"不支持的裁剪比例: {crop_mode}"}, 400)

    cropped_path = os.path.join(os.path.dirname(task["video_path"]), "result_cropped.mp4")
    try:
        def _do_crop():
            cap = cv2.VideoCapture(result_path)
            sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            sr = sw / sh if sh > 0 else 1
            if abs(sr - tr) < 0.01:
                cap.release()
                shutil.copy2(result_path, cropped_path)
                return
            if tr > sr:
                nh = int(sw / tr); nh -= nh % 2
                margin = sh - nh
                shift = int(margin * crop_offset / 2)
                y1 = max(0, min(sh - nh, (sh - nh) // 2 + shift))
                x1, x2, y2 = 0, sw, y1 + nh
            else:
                nw = int(sh * tr); nw -= nw % 2
                margin = sw - nw
                shift = int(margin * crop_offset / 2)
                x1 = max(0, min(sw - nw, (sw - nw) // 2 + shift))
                y1, y2, x2 = 0, sh, x1 + nw
            out_w, out_h = x2 - x1, y2 - y1
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(cropped_path, fourcc, fps, (out_w, out_h))
            while True:
                ret, frame = cap.read()
                if not ret: break
                out.write(frame[y1:y2, x1:x2])
            cap.release(); out.release()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _do_crop)
        # 裁剪后合回音频
        from src.utils import has_audio_stream, merge_audio_with_moviepy
        src_audio = task["video_path"]  # 原片一定有音频
        if has_audio_stream(src_audio):
            try:
                merge_audio_with_moviepy(cropped_path, src_audio,
                                          cropped_path + ".audio.mp4")
                os.replace(cropped_path + ".audio.mp4", cropped_path)
            except Exception:
                pass  # 音频合成失败不阻塞，无声版也能用
        return {"ok": True, "path": cropped_path,
                "download_url": f"/result/{task_id}?cropped=1"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# ================================================================
#  /result/{task_id} — 下载已完成的视频
# ================================================================

@app.get("/result/{task_id}")
async def get_result(task_id: str, res: str = "original", fps: str = "original", cropped: str = "0"):
    p = PROGRESS.get(task_id, {})
    output_path = p.get("output_path", "")
    # 如果 PROGRESS 丢失（服务重启），尝试从磁盘恢复路径
    if not output_path:
        task_dir = os.path.join(TASKS_DIR, task_id)
        mp4_path = os.path.join(task_dir, "result.mp4")
        if os.path.exists(mp4_path):
            output_path = mp4_path
    if cropped == "1":
        cp = os.path.join(os.path.dirname(output_path) if output_path else os.path.join(TASKS_DIR, task_id), "result_cropped.mp4")
        if os.path.exists(cp):
            output_path = cp
    if not output_path or not os.path.exists(output_path):
        if not p.get("done"):
            return JSONResponse({"status": "not_ready"}, 202)
        return JSONResponse({"error": "输出文件不存在"}, 404)
    if p.get("error"):
        return JSONResponse({"error": p["error"]}, 500)
    if not output_path or not os.path.exists(output_path):
        return JSONResponse({"error": "输出文件不存在"}, 404)

    RES_MAP = {"2k": (2560, 1440), "1080p": (1920, 1080),
               "720p": (1280, 720), "480p": (854, 480)}

    def _transcode(_output_path, _res, _target_w, _target_h, _target_fps):
        suffix = f".{_res}" if _res != "original" else ""
        suffix += f".{_target_fps}fps" if _target_fps else ""
        transcode_path = _output_path + suffix + ".mp4"
        if os.path.exists(transcode_path):
            return transcode_path
        from moviepy.video.io.VideoFileClip import VideoFileClip
        clip = VideoFileClip(_output_path)
        try:
            if _res != "original":
                if clip.w > _target_w or clip.h > _target_h:
                    ratio = min(_target_w / clip.w, _target_h / clip.h)
                    w, h = int(clip.w * ratio), int(clip.h * ratio)
                    try: clip = clip.resized(newsize=(w, h))
                    except AttributeError: clip = clip.resize(newsize=(w, h))
            if _target_fps and abs(clip.fps - _target_fps) > 1:
                try: clip = clip.with_fps(_target_fps)
                except AttributeError: clip.fps = _target_fps
            clip.write_videofile(transcode_path, codec="libx264",
                                  audio_codec="aac", logger=None)
            return transcode_path
        finally:
            clip.close()

    need_resize = res in RES_MAP and res != "original"
    target_fps = int(fps) if fps.isdigit() else 0
    if need_resize or target_fps:
        target_w, target_h = RES_MAP.get(res, (99999, 99999))
        loop = asyncio.get_running_loop()
        final_path = await loop.run_in_executor(
            None, _transcode, output_path, res, target_w, target_h, target_fps)
    else:
        final_path = output_path

    download_name = "dance_anonymized_" + task_id + ".mp4"
    return FileResponse(
        path=final_path, media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="' + download_name + '"'})


# ================================================================
#  前端 HTML
# ================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
