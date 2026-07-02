"""
舞蹈视频智能打码/特效渲染系统 — FastAPI v5 (YOLO+SAM2)
=========================================================
交互: 上传 → 实时调参预览 → 3s片段 / 全片渲染 (任选其一, 可取消)
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

def _render_one_frame(frame, track_results, target_ids, params, labels_config=None):
    """预览模式: 打码仅 target_ids, 标签覆盖所有人(自定义昵称或默认ID)。"""
    all_track_results = track_results

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
    return result


def _img_to_base64(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"


def _parse_ids(target_ids_str):
    if target_ids_str and target_ids_str.strip():
        return [int(x.strip()) for x in target_ids_str.split(",") if x.strip()]
    return []


# ================================================================
#  / — 前端
# ================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


# ================================================================
#  /analyze
# ================================================================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    task_id = uuid.uuid4().hex[:10]
    task_dir = os.path.join(TASKS_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    video_path = os.path.join(task_dir, f"source{ext}")

    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

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
            print(f"[Analyze] SAM 2 精修跳过: {e}")

        track_results = []
        for new_id, tr in enumerate(raw_results):
            mask = sam2_masks.get(tr.track_id, tr.mask)
            track_results.append(TrackResult(
                track_id=new_id, bbox=tr.bbox,
                confidence=tr.confidence, mask=mask, foot_y=tr.foot_y,
            ))
        key_frame = best_frame
        available_ids = sorted([t.track_id for t in track_results])

        default_params = {"fill_mode": "solid", "fill_color": "#000000",
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
):
    task = TASKS.get(task_id)
    if not task:
        return JSONResponse({"error": "task_id 无效"}, 404)
    parsed_ids = _parse_ids(target_ids)
    labels_cfg = json.loads(labels_config) if labels_config.strip() else None
    try:
        params = {"fill_mode": fill_mode, "fill_color": fill_color,
                   "border_color": border_color, "opacity": opacity, "thickness": thickness}
        rendered = _render_one_frame(task["key_frame"], task["track_results"],
                                      parsed_ids, params, labels_cfg)
        return {"image_base64": _img_to_base64(rendered)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# ================================================================
#  /preview_snippet (3秒片段, 异步可中断)
# ================================================================

@app.post("/preview_snippet")
async def preview_snippet(
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
):
    task = TASKS.get(task_id)
    if not task:
        return JSONResponse({"error": "task_id 无效"}, 404)
    parsed_ids = _parse_ids(target_ids)
    labels_cfg = json.loads(labels_config) if labels_config.strip() else None
    video_path = task["video_path"]
    snippet_path = os.path.join(os.path.dirname(video_path), "snippet.mp4")
    snippet_frames = min(int(task["fps"] * 3), task["total_frames"])

    cancel_event = threading.Event()
    watcher_task = None
    try:
        # generation 计数器: 防止取消后旧线程覆盖新线程的进度
        gen = PROGRESS.get(task_id, {}).get("generation", 0) + 1
        PROGRESS[task_id] = {"done": False, "generation": gen}
        def _on_progress(p):
            if PROGRESS.get(task_id, {}).get("generation") == gen:
                PROGRESS[task_id] = {**p, "done": False, "generation": gen}

        async def disconnect_watcher():
            while not cancel_event.is_set():
                if await request.is_disconnected():
                    cancel_event.set()
                    return
                await asyncio.sleep(0.5)

        watcher_task = asyncio.create_task(disconnect_watcher())

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
                    input_path=video_path, output_path=snippet_path,
                    target_ids=parsed_ids, labels_config=labels_cfg,
                    precomputed_detections=task.get("track_results"),
                    max_frames=snippet_frames,
                    show_progress=False, cancel_event=cancel_event,
                    progress_callback=_on_progress,
                    fill_mode=fill_mode, fill_color=fill_color,
                    border_color=border_color, opacity=opacity,
                )
                result["status"] = "done"
                if PROGRESS.get(task_id, {}).get("generation") == gen:
                    PROGRESS[task_id] = {"done": True, "generation": gen}
            except Exception as e:
                result["error"] = str(e)
                if PROGRESS.get(task_id, {}).get("generation") == gen:
                    PROGRESS[task_id] = {"done": True, "error": str(e), "generation": gen}

        thread = threading.Thread(target=_run)
        thread.start()

        while thread.is_alive():
            if await request.is_disconnected():
                cancel_event.set()
                thread.join(timeout=5)
                PROGRESS[task_id] = {"done": True}
                return JSONResponse({"cancelled": True})
            await asyncio.sleep(0.3)

        thread.join()
        watcher_task.cancel()

        if result.get("error"):
            return JSONResponse({"error": result["error"]}, 500)
        if cancel_event.is_set():
            return JSONResponse({"cancelled": True})

        if not os.path.exists(snippet_path):
            return JSONResponse({"error": "输出文件未生成"}, 500)

        return FileResponse(path=snippet_path, media_type="video/mp4",
                            filename=f"snippet_{task_id}.mp4")

    except Exception as e:
        cancel_event.set()
        return JSONResponse({"error": str(e)}, 500)
    finally:
        if watcher_task is not None:
            watcher_task.cancel()


# ================================================================
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
):
    task = TASKS.get(task_id)
    if not task:
        return JSONResponse({"error": "task_id 无效"}, 404)
    parsed_ids = _parse_ids(target_ids)
    labels_cfg = json.loads(labels_config) if labels_config.strip() else None
    if not parsed_ids:
        return JSONResponse({"error": "没有选中任何人"}, 400)

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
#  /result/{task_id} — 下载已完成的视频
# ================================================================

@app.get("/result/{task_id}")
async def get_result(task_id: str, res: str = "original", fps: str = "original"):
    p = PROGRESS.get(task_id, {})
    if not p.get("done"):
        return JSONResponse({"status": "not_ready"}, 202)
    if p.get("error"):
        return JSONResponse({"error": p["error"]}, 500)
    output_path = p.get("output_path", "")
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
                    clip = clip.resized(newsize=(w, h))
            if _target_fps and abs(clip.fps - _target_fps) > 1:
                clip = clip.with_fps(_target_fps)
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

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no">
<title>舞蹈视频智能打码</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#18181B;color:#F4F4F5;display:flex;justify-content:center;min-height:100vh;padding:20px}
.container{max-width:820px;width:100%}
h1{text-align:center;color:#fff;font-size:1.5em;margin-bottom:20px}
.sub{text-align:center;color:#A1A1AA;margin-bottom:24px;font-size:.9em}
.card{background:#27272A;border-radius:12px;padding:20px;margin-bottom:24px;border:1px solid #3F3F46}
.card h2{font-size:1.05em;color:#A78BFA;margin-bottom:14px;font-weight:600}
input[type=text],input[type=number],select{height:38px;border-radius:6px;background:#3F3F46;border:1px solid transparent;color:#F4F4F5;font-size:.9em;outline:none;padding:0 12px;transition:border-color .2s}
input[type=text]:focus,input[type=number]:focus,select:focus{border-color:#6366F1}
select{cursor:pointer}
input[type=file]{width:100%;padding:10px 0;color:#A1A1AA;font-size:.9em;margin-bottom:10px}
label.block{display:block;font-size:.83em;color:#A1A1AA;margin-bottom:4px}
.btn{display:inline-flex;align-items:center;justify-content:center;height:44px;padding:0 24px;border-radius:8px;border:none;font-weight:700;font-size:.93em;cursor:pointer;transition:all .15s}
.btn:disabled{opacity:0.45;cursor:not-allowed}
.btn-primary{background:#7C3AED;color:#fff}.btn-primary:hover:not(:disabled){background:#6D28D9}
.btn-secondary{background:#3F3F46;color:#F4F4F5}.btn-secondary:hover:not(:disabled){background:#52525B}
.btn-ghost{background:transparent;color:#A1A1AA;border:1px solid #3F3F46}.btn-ghost:hover:not(:disabled){color:#fff;border-color:#52525B}
.btn-cancel{background:#F97316!important;color:#fff!important;animation:pulse .8s infinite alternate}
@keyframes pulse{from{opacity:1}to{opacity:0.7}}
.preview-card{flex-shrink:0;margin-bottom:12px}
.preview-card img,video{max-width:100%;max-height:50vh;object-fit:contain;background:#121214;border-radius:8px;border:1px solid #3F3F46;display:block;margin:0 auto}
.id-list{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
.id-row{display:flex;align-items:center;gap:12px;padding:12px 16px;background:#202022;border-radius:8px;width:100%;box-sizing:border-box}
.id-row .cb-label{display:flex;align-items:center;gap:8px;cursor:pointer;font-size:.9em;color:#D4D4D8;white-space:nowrap;min-width:70px}
.id-row .cb-label input[type=checkbox]{width:18px;height:18px;accent-color:#A78BFA;cursor:pointer}
.id-row .nick-input{flex:1;min-width:0;height:36px;margin:0}
.params-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.form-group{display:flex;flex-direction:column;gap:6px}
.form-group label.block{margin:0}
input[type=range]{width:100%;accent-color:#A78BFA}
.color-picker-wrapper{display:flex;align-items:center;gap:8px;background:#3F3F46;padding:4px 8px;border-radius:6px;height:38px}
.color-picker-wrapper input[type=color]{width:30px;height:28px;border:none;background:transparent;cursor:pointer;padding:0;border-radius:4px}
.color-picker-wrapper input[type=text]{flex:1;height:auto;background:transparent;border:none;padding:0;color:#F4F4F5;font-size:.9em}
.color-picker-wrapper input[type=text]:focus{border:none}
.action-bar{display:flex;gap:16px;justify-content:flex-end;margin-top:24px;border-top:1px solid #3F3F46;padding-top:20px}
#step1,#step2,#step3{display:none}
#step1.active,#step3.active{display:block}
#step2.active{display:flex;flex-direction:column;height:calc(100vh - 110px);overflow:hidden}
.scrollable-controls{flex:1;overflow-y:auto;padding-right:4px;padding-bottom:20px}
.scrollable-controls::-webkit-scrollbar{width:6px}
.scrollable-controls::-webkit-scrollbar-thumb{background:#52525B;border-radius:4px}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid #fff6;border-top-color:#fff;border-radius:50%;animation:spin .5s linear infinite;margin-right:6px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
#snippetStatus,#renderStatus{margin-top:10px;font-size:.85em}
#progressCard{display:none;margin-top:16px;padding:16px;background:#202022;border-radius:8px;border:1px solid #3F3F46}
#progressCard.active{display:block}
.progress-bar-bg{width:100%;height:10px;background:#3F3F46;border-radius:5px;overflow:hidden;margin:10px 0}
.progress-bar-fill{height:100%;background:linear-gradient(90deg,#A78BFA,#6366F1);border-radius:5px;transition:width .3s;width:0%}
.progress-info{display:flex;justify-content:space-between;color:#A1A1AA;font-size:.82em}
.progress-info span{color:#A78BFA;font-weight:700}
@media (max-width:768px){
  body{padding:8px}
  .container{max-width:100%}
  .card{padding:10px;margin-bottom:12px}
  h1{font-size:1.2em;margin-bottom:16px}
  .sub{font-size:.78em;margin-bottom:10px}
  .id-list{display:flex;flex-direction:column;gap:6px}
  .id-row{display:flex;flex-direction:row!important;align-items:center;justify-content:space-between;gap:10px;padding:8px 10px;width:100%;flex-wrap:nowrap!important}
  .id-row .cb-label{min-width:auto;font-size:.85em}
  .id-row .nick-input{flex:1;min-width:0;max-width:none;height:34px;margin:0;font-size:13px}
  .params-grid{display:flex!important;flex-direction:column;gap:6px}
  .form-group{display:flex;flex-direction:row!important;align-items:center;justify-content:space-between;background:#202022;padding:8px 10px;border-radius:8px;gap:10px}
  .form-group label.block{font-size:.82em;color:#A1A1AA;white-space:nowrap;min-width:60px;margin:0}
  .form-group select,.form-group input[type=number]{width:auto;min-width:80px;height:34px;font-size:13px;margin:0}
  .color-picker-wrapper{height:34px;padding:2px 6px;gap:4px}
  .color-picker-wrapper input[type=color]{width:24px;height:24px}
  .color-picker-wrapper input[type=text]{font-size:13px}
  .btn{height:40px;padding:0 14px;font-size:.85em}
  .action-bar{flex-direction:column;gap:8px;margin-top:16px;padding-top:14px}
  .action-bar .btn{width:100%}
  input[type=text],input[type=number],select{height:34px;font-size:13px}
  .form-group:has(input[type=range]){flex-direction:column!important;align-items:stretch}
  .form-group:has(input[type=range]) label.block{min-width:auto}
}
</style>
</head>
<body>
<div class="container">
<h1>舞蹈视频智能打码</h1>

<div id="step1" class="card active">
  <h2>Step 1 — 上传视频</h2>
  <input type="file" id="fileInput" accept="video/*">
  <button class="btn btn-primary" id="uploadBtn">上传并分析</button>
  <div id="uploadStatus"></div>
</div>

<div id="step2">
  <div class="card preview-card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
      <h2 style="margin:0">预览</h2>
      <button class="btn btn-ghost" id="reuploadBtn" style="height:34px;padding:0 14px;font-size:.8em">↻ 重新上传</button>
    </div>
    <img id="previewImg" alt="预览">
  </div>

  <div class="scrollable-controls">
  <div class="card">
    <h2>打码对象</h2>
    <div id="idCheckboxes" class="id-list"></div>
  </div>

  <div class="card">
    <h2>打码设置</h2>
    <div class="params-grid">
      <div class="form-group">
        <label class="block">填充模式</label>
        <select id="fillMode"><option value="solid">纯色</option><option value="gradient">渐变</option><option value="blur">模糊</option></select>
      </div>
      <div class="form-group">
        <label class="block">白边宽度</label>
        <input type="number" id="thickness" value="3" min="1" max="15">
      </div>
      <div class="form-group">
        <label class="block">填充色</label>
        <div class="color-picker-wrapper"><input type="color" id="fillColor" value="#000000"><input type="text" id="fillColorHex" value="#000000"></div>
      </div>
      <div class="form-group">
        <label class="block">边框色</label>
        <div class="color-picker-wrapper"><input type="color" id="borderColor" value="#FFFFFF"><input type="text" id="borderColorHex" value="#FFFFFF"></div>
      </div>
      <div class="form-group">
        <label class="block">透明度 <span id="opacityVal">100%</span></label>
        <input type="range" id="opacity" min="0" max="100" value="100">
      </div>
    </div>

    <div class="action-bar">
      <button class="btn btn-secondary" id="snippetBtn">生成 3 秒预览</button>
      <button class="btn btn-primary" id="renderBtn">生成完整视频</button>
    </div>
    <div id="snippetStatus"></div><div id="renderStatus"></div>
    </div>
    <div id="progressCard">
      <div id="progressStep" style="color:#A78BFA;font-size:.9em;font-weight:700"></div>
      <div class="progress-bar-bg"><div class="progress-bar-fill" id="progressFill"></div></div>
      <div class="progress-info">
        <span id="progressPct">0%</span>
        <span id="progressFps"></span>
        <span id="progressEta"></span>
      </div>
    </div>
  </div>
</div>

<div id="step3" class="card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
    <h2 style="margin:0">处理完成</h2>
    <button class="btn btn-ghost" onclick="location.reload()" style="height:34px;padding:0 14px;font-size:.85em">↻ 处理新视频</button>
  </div>
  <div id="resultArea"></div>
</div>
<div class="modal-overlay" id="qualityModal" style="display:none;position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,.6);backdrop-filter:blur(4px);justify-content:center;align-items:center">
  <div class="modal-card" style="background:#27272A;border-radius:12px;padding:24px;border:1px solid #3F3F46;min-width:300px;max-width:340px;text-align:center">
    <h3 style="color:#F4F4F5;margin-bottom:16px">选择下载画质</h3>
    <div style="display:flex;flex-direction:column;gap:16px;margin-bottom:24px">
      <div class="form-group" style="display:flex;justify-content:space-between;align-items:center">
        <label style="color:#A1A1AA;font-size:.9em">分辨率</label>
        <select id="downloadRes" style="width:140px;height:38px;background:#3F3F46;border:none;color:#F4F4F5;padding:0 12px;border-radius:6px;font-size:.9em">
          <option value="original">原始画质</option>
          <option value="2k">2K</option>
          <option value="1080p">1080P</option>
          <option value="720p">720P</option>
          <option value="480p">480P</option>
        </select>
      </div>
      <div class="form-group" style="display:flex;justify-content:space-between;align-items:center">
        <label style="color:#A1A1AA;font-size:.9em">帧率</label>
        <select id="downloadFps" style="width:140px;height:38px;background:#3F3F46;border:none;color:#F4F4F5;padding:0 12px;border-radius:6px;font-size:.9em">
          <option value="original">原始帧率</option>
          <option value="60">60 FPS</option>
          <option value="50">50 FPS</option>
          <option value="30">30 FPS</option>
          <option value="24">24 FPS</option>
        </select>
      </div>
    </div>
    <div style="display:flex;gap:10px">
      <button id="modalCancel" style="flex:1;padding:10px;background:transparent;border:1px solid #3F3F46;border-radius:8px;color:#A1A1AA;cursor:pointer;font-size:.9em">取消</button>
      <button id="modalConfirm" style="flex:1;padding:10px;background:#7C3AED;border:none;border-radius:8px;color:#fff;cursor:pointer;font-size:.9em;font-weight:700">确认下载</button>
    </div>
  </div>
</div>

<script>
let taskId = null, debounceTimer = null;
let snippetCtrl = null, renderCtrl = null;
let progressTimer = null;
let resultTaskId = null;
let snippetGenerated = false;

function showQualityModal(tid){
  resultTaskId=tid;
  var m=$('qualityModal'); if(m)m.style.display='flex';
}
function hideQualityModal(){
  var m=$('qualityModal'); if(m)m.style.display='none';
}
document.addEventListener('DOMContentLoaded', function(){
  var mc=$('modalCancel'), mf=$('modalConfirm'), mq=$('qualityModal');
  if(mc) mc.addEventListener('click', hideQualityModal);
  if(mq) mq.addEventListener('click', function(e){ if(e.target===this) hideQualityModal(); });
  if(mf) mf.addEventListener('click', function(){
    var res=$('downloadRes')?$('downloadRes').value:'original';
    var fps=$('downloadFps')?$('downloadFps').value:'original';
    hideQualityModal();
    var params=[];
    if(res!=='original') params.push('res='+res);
    if(fps!=='original') params.push('fps='+fps);
    var qs=params.length>0?'?'+params.join('&'):'';
    var a=document.createElement('a');
    a.href='/result/'+resultTaskId+qs;
    a.download='';
    a.style.display='none';
    document.body.appendChild(a);
    a.click();
    setTimeout(function(){ document.body.removeChild(a); }, 100);
  });
});

const $=id=>document.getElementById(id);
const BTN_SNIPPET='snippetBtn', BTN_RENDER='renderBtn';

function resetToStep1(){
  if(snippetCtrl){ snippetCtrl.abort(); snippetCtrl=null; }
  if(renderCtrl){ renderCtrl.abort(); renderCtrl=null; }
  if(taskId){
    const tid = taskId;
    fetch('/cleanup/'+tid, {method:'DELETE'}).catch(()=>{});
    taskId = null;
  }
  $('fileInput').value = '';
  clearTimeout(debounceTimer); debounceTimer = null;
  $('opacity').value = 100; $('opacityVal').textContent = '100%';
  $('fillMode').value = 'solid';
  $('fillColor').value = '#000000'; $('fillColorHex').value = '#000000';
  $('borderColor').value = '#FFFFFF'; $('borderColorHex').value = '#FFFFFF';
  $('thickness').value = 3;
  $('step2').classList.remove('active');
  $('step3').classList.remove('active');
  $('step1').classList.add('active');
  const uploadBtn = $('uploadBtn');
  uploadBtn.disabled = false;
  uploadBtn.textContent = '上传并分析';
  uploadBtn.classList.remove('btn-cancel');
  $('uploadStatus').innerHTML = '';
  $('previewImg').src = '';
  $('idCheckboxes').innerHTML = '';
  $('snippetStatus').innerHTML = '';
  $('renderStatus').innerHTML = '';
  $('resultArea').innerHTML = '';
  renderRunning = false;
  snippetGenerated = false;
  stopProgress();
  unlockButtons();
  const sb=$('snippetBtn'), rb=$('renderBtn');
  sb.classList.remove('btn-cancel'); sb.disabled=false;
  sb.innerHTML='生成 3 秒预览';
  rb.classList.remove('btn-cancel'); rb.disabled=false;
  rb.innerHTML='生成完整视频';
}

$('reuploadBtn').addEventListener('click', resetToStep1);

function buildLabelsConfig(){
  var cfg={};
  document.querySelectorAll('#idCheckboxes .nick-input').forEach(inp=>{
    var text=inp.value.trim(); if(!text)return;
    var tid=inp.closest('div').querySelector('.anon-cb').value;
    cfg[tid]={text:text};
  });
  return Object.keys(cfg).length>0?JSON.stringify(cfg):'';
}
function getParams(){
  var checks=document.querySelectorAll('#idCheckboxes .anon-cb:checked');
  return {target_ids:Array.from(checks).map(c=>c.value).join(','),labels_config:buildLabelsConfig(),fill_mode:$('fillMode').value,fill_color:$('fillColor').value,border_color:$('borderColor').value,thickness:$('thickness').value,opacity:$('opacity').value/100};
}

async function updatePreview(){
  if(!taskId)return;
  const p=getParams(),fd=new FormData();fd.append('task_id',taskId);
  for(const[k,v]of Object.entries(p))fd.append(k,String(v));
  try{const r=await fetch('/preview_frame',{method:'POST',body:fd});const d=await r.json();if(d.image_base64)$('previewImg').src=d.image_base64;}catch(e){}
}
function debounceUpdate(){
  clearTimeout(debounceTimer);
  debounceTimer=setTimeout(updatePreview,300);
  if(snippetGenerated){
    $('snippetBtn').disabled = false;
    $('snippetBtn').innerHTML = '重新生成预览视频';
  }
}

function lockButtons(activeBtnId){
  const other = activeBtnId===BTN_SNIPPET ? BTN_RENDER : BTN_SNIPPET;
  $(other).disabled = true;
}
function unlockButtons(){
  $(BTN_SNIPPET).disabled = false;
  $(BTN_RENDER).disabled = false;
}
function setCancelling(btn){
  btn.classList.add('btn-cancel');
  btn.innerHTML = '<span class="spinner"></span>取消';
}
function resetButton(btn, originalText, originalClass){
  btn.classList.remove('btn-cancel');
  btn.className = originalClass;
  btn.innerHTML = originalText;
}

function formatTime(s){
  if(!s||s<=0||s===Infinity)return '--';
  var m=Math.floor(s/60),sec=Math.floor(s%60);
  return m>0?m+'分'+sec+'秒':sec+'秒';
}
function startProgress(onDone){
  $('progressCard').classList.add('active');
  $('progressFill').style.width='0%'; $('progressPct').textContent='0%';
  $('progressFps').textContent=''; $('progressEta').textContent='';
  $('progressStep').textContent='准备中...';
  setTimeout(function(){ $('progressCard').scrollIntoView({behavior:'smooth',block:'end'}); },100);
  function poll(){
    if(!taskId)return;
    fetch('/status/'+taskId).then(r=>r.json()).then(p=>{
      if(p.done){
        stopProgress();
        if(onDone) onDone(p);
        return;
      }
      if(p.running || p.step>0){
        var names={1:'准备视频',2:'分析人物',3:'智能处理',4:'生成视频'};$('progressStep').textContent=(names[p.step]||'处理中')+' · '+p.step+'/'+p.step_total;
        if(p.frames_total>0){
          var pct=Math.round(p.frames_done/p.frames_total*100);
          $('progressFill').style.width=pct+'%';
          $('progressPct').textContent=pct+'% ('+p.frames_done+'/'+p.frames_total+'帧)';
          if(p.fps>0)$('progressFps').textContent=p.fps.toFixed(1)+' 帧/秒';
          $('progressEta').textContent='⏱ '+formatTime(p.eta);
        }else{$('progressFill').style.width='30%';$('progressPct').textContent='...';}
      }
      progressTimer=setTimeout(poll,800);
    }).catch(()=>{progressTimer=setTimeout(poll,1500);});
  }
  // 延迟首次 poll，等服务端更新 PROGRESS 后再开始轮询
  progressTimer=setTimeout(poll,300);
}
function stopProgress(){
  clearTimeout(progressTimer);progressTimer=null;
  $('progressCard').classList.remove('active');
}

function makeFetch(endpoint, btnId, statusId, onSuccess){
  const btn = $(btnId);
  const origText = btn.innerHTML;
  const origClass = btn.className;
  if (btnId===BTN_SNIPPET && snippetCtrl || btnId===BTN_RENDER && renderCtrl){
    if(!confirm('确定要取消当前任务吗？')) return;
    const ctrl = btnId===BTN_SNIPPET ? snippetCtrl : renderCtrl;
    ctrl.abort();
    stopProgress();
    fetch('/cancel/'+taskId, {method:'POST'}).catch(()=>{});
    return;
  }
  const ctrl = new AbortController();
  if(btnId===BTN_SNIPPET){ snippetCtrl=ctrl; snippetGenerated=false; }
  else renderCtrl=ctrl;
  lockButtons(btnId);
  setCancelling(btn);
  $(statusId).innerHTML = '';
  startProgress();
  const p = getParams();
  const fd = new FormData(); fd.append('task_id', taskId);
  for(const[k,v] of Object.entries(p)) fd.append(k, String(v));
  fetch(endpoint, {method:'POST', body:fd, signal:ctrl.signal})
  .then(async r => {
    if(!r.ok){const e=await r.json();$(statusId).innerHTML=`<span style="color:#e94560">${e.error||e.detail}</span>`;return;}
    onSuccess(r);
  })
  .catch(err => {
    if(err.name==='AbortError'){
    } else {
      $(statusId).innerHTML = `<span style="color:#e94560">错误: ${err.message}</span>`;
    }
    stopProgress();
  })
  .finally(() => {
    if(btnId===BTN_SNIPPET) snippetCtrl=null; else renderCtrl=null;
    resetButton(btn, origText, origClass);
    unlockButtons();
  });
}

$('snippetBtn').addEventListener('click', ()=>{
  makeFetch('/preview_snippet', BTN_SNIPPET, 'snippetStatus', async r => {
    stopProgress();
    const blob = await r.blob();
    const snippetUrl = URL.createObjectURL(blob);
    $('snippetStatus').innerHTML = `<video controls autoplay loop muted src="${snippetUrl}"></video>
      <br><a class="btn btn-snippet" href="${snippetUrl}" download="preview_${taskId}.mp4" style="display:inline-block;margin-top:8px;text-decoration:none;padding:6px 14px;font-size:.85em">下载预览片段</a>`;
    snippetGenerated = true;
    $('snippetBtn').innerHTML = '生成 3 秒预览';
  });
});

let renderRunning=false;
$('renderBtn').addEventListener('click', ()=>{
  if(renderRunning){
    if(!confirm('确定要取消当前任务吗？')) return;
    renderRunning=false;
    stopProgress();
    $('renderStatus').innerHTML='';
    unlockButtons();
    resetButton($('renderBtn'), '生成完整视频', 'btn btn-primary');
    fetch('/cancel/'+taskId, {method:'POST'}).catch(()=>{});
    return;
  }
  renderRunning=true;
  lockButtons(BTN_RENDER);
  setCancelling($('renderBtn'));
  $('renderStatus').innerHTML='';
  const p=getParams(), fd=new FormData(); fd.append('task_id', taskId);
  for(var k in p) fd.append(k, String(p[k]));
  fetch('/render', {method:'POST', body:fd})
  .then(async r=>{
    const d=await r.json();
    if(d.status==='started'){
      $('renderStatus').innerHTML='<span style="color:#A78BFA">处理中, 可切换应用, 完成后点击下载</span>';
      startProgress(function(p){
        renderRunning=false;
        unlockButtons();
        resetButton($('renderBtn'), '生成完整视频', 'btn btn-primary');
        if(p && p.error){
          $('renderStatus').innerHTML='<span style="color:#e94560">处理失败: '+p.error+'</span>';
          return;
        }
        var vUrl='/result/'+taskId;
        $('resultArea').innerHTML=
          '<video controls src="'+vUrl+'" style="max-width:100%;border-radius:8px;margin-bottom:12px"></video>'+
          '<button class="btn btn-primary" id="downloadBtn">下载视频</button>';
        $('downloadBtn').addEventListener('click', function(){ showQualityModal(taskId); });
        $('step2').classList.remove('active');
        $('step3').classList.add('active');
        $('renderStatus').innerHTML='';
      });
    }else{
      renderRunning=false;
      unlockButtons();
      resetButton($('renderBtn'), '生成完整视频', 'btn btn-primary');
    }
  })
  .catch(err=>{
    stopProgress();
    renderRunning=false;
    unlockButtons();
    resetButton($('renderBtn'), '生成完整视频', 'btn btn-primary');
    if(err.name!=='AbortError') $('renderStatus').innerHTML='<span style="color:#e94560">错误: '+err.message+'</span>';
  });
});

$('uploadBtn').addEventListener('click', async ()=>{
  const file=$('fileInput').files[0];if(!file)return alert('请选择视频');if(file.size>500*1024*1024){alert('上传失败：视频文件过大！请上传 500MB 以内的视频。');return;}
  if(file.size>500*1024*1024){alert('上传失败：视频文件过大！请上传 500MB 以内的视频。');return;}
  const btn=$('uploadBtn');btn.disabled=true;btn.innerHTML='<span class="spinner"></span>分析中...';
  const fd=new FormData();fd.append('file',file);
  try{
    const r=await fetch('/analyze',{method:'POST',body:fd});const d=await r.json();
    if(d.error){$('uploadStatus').innerHTML=`<span style="color:#e94560">${d.error}</span>`;alert(d.error);return;}
    taskId=d.task_id;$('previewImg').src=d.image_base64;
    const g=$('idCheckboxes');g.innerHTML='';
    d.available_ids.forEach(id=>{g.innerHTML+=`<div class="id-row"><label class="cb-label"><input type="checkbox" class="anon-cb" value="${id}" checked onchange="debounceUpdate()"> 人物${parseInt(id)+1}</label><input type="text" class="nick-input" placeholder="输入昵称..." maxlength="6" oninput="debounceUpdate()"></div>`;});
    $('step1').classList.remove('active');$('step2').classList.add('active');
  }catch(e){$('uploadStatus').textContent='错误: '+e.message;}
  finally{btn.disabled=false;btn.textContent='上传并分析';}
});

$('opacity').addEventListener('input',()=>{$('opacityVal').textContent=$('opacity').value+'%';debounceUpdate();});
$('fillColor').addEventListener('input',()=>{$('fillColorHex').value=$('fillColor').value;debounceUpdate();});
$('borderColor').addEventListener('input',()=>{$('borderColorHex').value=$('borderColor').value;debounceUpdate();});
['fillMode','thickness'].forEach(id=>$(id).addEventListener('change',debounceUpdate));
['fillColor','borderColor'].forEach(id=>$(id).addEventListener('input',debounceUpdate));
['fillColorHex','borderColorHex'].forEach(id=>$(id).addEventListener('change',function(){$(id.replace('Hex','')).value=this.value;debounceUpdate();}));
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
