"""
舞蹈视频智能打码/特效渲染系统 — FastAPI v4 (互斥+可中断)
=========================================================
交互: 上传 → 实时调参预览 → 3s片段 / 全片渲染 (任选其一, 可取消)
启动: uvicorn api:app --host 0.0.0.0 --port 8002
"""

import os, sys, uuid, json, base64, shutil, cv2, time, threading, asyncio
import numpy as np
from typing import Optional, List, Dict
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tracker import DanceTracker, TrackerConfig, TrackResult
from src.pipeline import DanceAnonymizerPipeline
from src.effects import (
    process_frame_effects, TemporalMaskCache, calculate_depth_order,
)
from src.utils import VideoReader, VideoWriter, get_video_info

app = FastAPI(title="舞蹈视频智能打码 API v4", version="4.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(BASE_DIR, "data", "tasks")
os.makedirs(TASKS_DIR, exist_ok=True)

TASKS: Dict[str, dict] = {}


# ================================================================
#  /cleanup/{task_id} — 释放服务端资源
# ================================================================

@app.delete("/cleanup/{task_id}")
async def cleanup(task_id: str):
    task = TASKS.pop(task_id, None)
    if task:
        task_dir = os.path.dirname(task["video_path"])
        shutil.rmtree(task_dir, ignore_errors=True)
    return {"ok": True}


# ================================================================
#  渲染辅助
# ================================================================

def _render_one_frame(frame, track_results, target_ids, params):
    """对单帧应用特效 (无时序缓存), 并标注人物ID。"""
    if target_ids:
        target_set = set(target_ids)
        track_results = [t for t in track_results if t.track_id in target_set]
    if not track_results:
        return frame.copy()

    ordered_ids, _ = calculate_depth_order(track_results, temporal_window=1)
    tid_to_idx = {t.track_id: i for i, t in enumerate(track_results)}
    cache = TemporalMaskCache(window_size=1, ema_decay=0.0)

    from src.effects import apply_shadow_outline_effect
    result = apply_shadow_outline_effect(
        frame=frame, depth_order=ordered_ids,
        track_id_to_idx=tid_to_idx, track_results=track_results,
        temporal_cache=cache, frame_idx=0,
        dilate_kernel_size=params.get("thickness", 3),
        fill_mode=params.get("fill_mode", "solid"),
        fill_color=params.get("fill_color", "#000000"),
        border_color=params.get("border_color", "#FFFFFF"),
        opacity=params.get("opacity", 1.0),
        target_ids=target_ids,
    )

    # ★ 标注人物 ID
    for tr in track_results:
        x1, y1, x2, y2 = tr.bbox
        label = f"ID:{tr.track_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
        cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.putText(result, label, (x1 + 3, y1 - 5), font, 0.6, (255, 255, 255), 2)
    return result


def _img_to_base64(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"


def _parse_ids(target_ids_str, task):
    if target_ids_str.strip():
        return [int(x.strip()) for x in target_ids_str.split(",") if x.strip()]
    return task["available_ids"]


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
        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # ★ 扫描 12 帧, model.predict() 快速统计人数
        from ultralytics import YOLO
        scanner = YOLO("yolo11s-seg.pt")
        best_frame, best_count, best_idx = None, 0, 0
        sample_count = min(12, total_frames)
        sample_positions = [int(total_frames * i / (sample_count + 1)) for i in range(1, sample_count + 1)]

        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                continue
            preds = scanner.predict(frame, classes=[0], conf=0.3, device="cpu", verbose=False)
            n = len(preds[0].boxes) if preds[0].boxes is not None else 0
            if n > best_count:
                best_count = n
                best_frame = frame.copy()
                best_idx = pos

        if best_frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if ret: best_frame, best_idx = frame, 0
        cap.release()

        if best_frame is None:
            return JSONResponse({"error": "无法读取视频帧"}, 400)

        # 在最佳帧上跑完整 tracker (获取 mask + track_id)
        tracker = DanceTracker(TrackerConfig(device="cpu", verbose=False))
        track_results = tracker.track(best_frame)
        key_frame = best_frame
        available_ids = sorted([t.track_id for t in track_results])
        target_fr = best_idx

        default_params = {"fill_mode": "solid", "fill_color": "#000000",
                           "border_color": "#FFFFFF", "opacity": 1.0, "thickness": 3}
        rendered = _render_one_frame(key_frame, track_results, available_ids, default_params)

        TASKS[task_id] = {
            "video_path": video_path, "ext": ext,
            "key_frame": key_frame.copy(),
            "track_results": track_results,
            "available_ids": available_ids,
            "total_frames": total_frames,
            "fps": fps, "key_frame_idx": target_fr,
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
    fill_mode: str = Form("solid"),
    fill_color: str = Form("#000000"),
    border_color: str = Form("#FFFFFF"),
    thickness: int = Form(3),
    opacity: float = Form(1.0),
):
    task = TASKS.get(task_id)
    if not task:
        return JSONResponse({"error": "task_id 无效"}, 404)
    parsed_ids = _parse_ids(target_ids, task)
    try:
        params = {"fill_mode": fill_mode, "fill_color": fill_color,
                   "border_color": border_color, "opacity": opacity, "thickness": thickness}
        rendered = _render_one_frame(task["key_frame"], task["track_results"], parsed_ids, params)
        return {"image_base64": _img_to_base64(rendered)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# ================================================================
#  /preview_snippet (异步, 可中断)
# ================================================================

@app.post("/preview_snippet")
async def preview_snippet(
    request: Request,
    task_id: str = Form(...),
    target_ids: str = Form(""),
    fill_mode: str = Form("solid"),
    fill_color: str = Form("#000000"),
    border_color: str = Form("#FFFFFF"),
    thickness: int = Form(3),
    opacity: float = Form(1.0),
    device: str = Form("cpu"),
):
    task = TASKS.get(task_id)
    if not task:
        return JSONResponse({"error": "task_id 无效"}, 404)
    parsed_ids = _parse_ids(target_ids, task)
    video_path = task["video_path"]
    snippet_path = os.path.join(os.path.dirname(video_path), "snippet.mp4")
    raw_path = snippet_path + ".raw.mp4"
    fps = task["fps"]

    try:
        snippet_frames = min(int(fps * 3), task["total_frames"])
        cap = cv2.VideoCapture(video_path)
        w, h = int(cap.get(3)), int(cap.get(4))
        tracker = DanceTracker(TrackerConfig(device=device, verbose=False))
        out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        temporal_cache = TemporalMaskCache(window_size=4, ema_decay=0.4)

        for i in range(snippet_frames):
            # ★ 客户端断开检测
            if await request.is_disconnected():
                break

            ret, frame = cap.read()
            if not ret:
                break
            track_results = tracker.track(frame)
            result, _, temporal_cache = process_frame_effects(
                frame=frame, track_results=track_results,
                temporal_cache=temporal_cache, frame_idx=i,
                target_ids=parsed_ids,
                fill_mode=fill_mode, fill_color=fill_color,
                border_color=border_color, opacity=opacity,
                dilate_kernel_size=thickness,
            )
            out.write(result)
        cap.release()
        out.release()

        if await request.is_disconnected():
            os.remove(raw_path)
            return JSONResponse({"cancelled": True})

        # moviepy H.264 重编码
        from moviepy.video.io.VideoFileClip import VideoFileClip
        clip = VideoFileClip(raw_path)
        clip.write_videofile(snippet_path, codec="libx264", audio_codec="aac", logger=None)
        clip.close()
        os.remove(raw_path)

        return FileResponse(path=snippet_path, media_type="video/mp4",
                            filename=f"snippet_{task_id}.mp4")
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# ================================================================
#  /render (异步, 可中断)
# ================================================================

@app.post("/render")
async def render(
    request: Request,
    task_id: str = Form(...),
    target_ids: str = Form(""),
    fill_mode: str = Form("solid"),
    fill_color: str = Form("#000000"),
    border_color: str = Form("#FFFFFF"),
    thickness: int = Form(3),
    opacity: float = Form(1.0),
    device: str = Form("cpu"),
):
    task = TASKS.get(task_id)
    if not task:
        return JSONResponse({"error": "task_id 无效"}, 404)
    parsed_ids = _parse_ids(target_ids, task)
    if not parsed_ids:
        return JSONResponse({"error": "没有选中任何人"}, 400)

    output_path = os.path.join(os.path.dirname(task["video_path"]), "result.mp4")

    # threading.Event 作为取消信号
    cancel_event = threading.Event()

    # 后台线程: 轮询客户端断开状态
    async def disconnect_watcher():
        while not cancel_event.is_set():
            if await request.is_disconnected():
                cancel_event.set()
                return
            await asyncio.sleep(0.5)

    watcher_task = asyncio.create_task(disconnect_watcher())

    try:
        pipeline = DanceAnonymizerPipeline(
            tracker_config=TrackerConfig(model_path="yolo11s-seg.pt",
                                          device=device, conf_threshold=0.3,
                                          verbose=False),
            effect_config={"body_expand_pixels": 0,
                           "dilate_kernel_size": max(1, min(thickness, 15))},
        )
        # 在单独线程中跑 pipeline (因为 pipeline 是同步的)
        result = {}
        def _run():
            try:
                pipeline.process(
                    input_path=task["video_path"], output_path=output_path,
                    target_ids=parsed_ids, show_progress=False,
                    cancel_event=cancel_event,
                    fill_mode=fill_mode, fill_color=fill_color,
                    border_color=border_color, opacity=opacity,
                )
                result["status"] = "done"
            except Exception as e:
                result["error"] = str(e)

        thread = threading.Thread(target=_run)
        thread.start()

        # 等待渲染完成或断开
        while thread.is_alive():
            if await request.is_disconnected():
                cancel_event.set()
                thread.join(timeout=5)
                return JSONResponse({"cancelled": True})
            await asyncio.sleep(0.3)

        thread.join()
        watcher_task.cancel()

        if result.get("error"):
            return JSONResponse({"error": result["error"]}, 500)
        if cancel_event.is_set():
            return JSONResponse({"cancelled": True})

        if not os.path.exists(output_path):
            return JSONResponse({"error": "输出文件未生成"}, 500)

        return FileResponse(path=output_path, media_type="video/mp4",
                            filename=f"anonymized_{task_id}.mp4")

    except Exception as e:
        cancel_event.set()
        return JSONResponse({"error": str(e)}, 500)
    finally:
        watcher_task.cancel()


# ================================================================
#  前端 HTML
# ================================================================

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>舞蹈视频智能打码 v4</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0b0b0f;color:#ddd;display:flex;justify-content:center;min-height:100vh;padding:20px}
.container{max-width:780px;width:100%}
h1{text-align:center;color:#fff;font-size:1.5em;margin-bottom:4px}
.sub{text-align:center;color:#777;margin-bottom:24px;font-size:.9em}
.card{background:#16161e;border-radius:12px;padding:24px;margin-bottom:16px;border:1px solid #252530}
.card h2{font-size:1.1em;color:#e94560;margin-bottom:12px}
input[type=file],input[type=text],input[type=number],select{width:100%;padding:10px 14px;border-radius:8px;border:1px solid #333;background:#0d0d16;color:#ddd;font-size:.9em;outline:none;margin-bottom:10px}
input:focus,select:focus{border-color:#e94560}
label{display:block;font-size:.85em;color:#aaa;margin:8px 0 4px}
.hint{font-size:.78em;color:#555;margin-bottom:8px}
.btn{display:inline-block;padding:10px 24px;border-radius:8px;border:none;font-weight:700;font-size:.95em;cursor:pointer;margin:4px;transition:.15s}
.btn:disabled{opacity:0.45;cursor:not-allowed}
.btn-snippet{background:#2d6a4f;color:#fff}.btn-snippet:hover:not(:disabled){background:#1b4332}
.btn-render{background:#e94560;color:#fff}.btn-render:hover:not(:disabled){background:#c73650}
.btn-cancel{background:#ff6b35!important;color:#fff!important;animation:pulse .8s infinite alternate}
@keyframes pulse{from{opacity:1}to{opacity:0.7}}
.btn-outline{background:transparent;color:#e94560;border:1px solid #e94560}
.row{display:flex;gap:12px;flex-wrap:wrap}.row>*{flex:1;min-width:120px}
.col-half{flex:1 1 calc(50% - 8px);min-width:150px}
.col-third{flex:1 1 calc(33% - 8px);min-width:100px}
#step1,#step2,#step3{display:none}
#step1.active,#step2.active,#step3.active{display:block}
.preview-img{max-width:100%;border-radius:8px;border:1px solid #333}
.checkbox-group{display:flex;flex-wrap:wrap;gap:8px}
.checkbox-group label{display:flex;align-items:center;gap:6px;background:#0d0d16;border:1px solid #333;border-radius:8px;padding:6px 14px;cursor:pointer;font-size:.9em;color:#ddd;margin:0;transition:.15s}
.checkbox-group input[type=checkbox]{width:16px;height:16px;accent-color:#e94560}
.color-row{display:flex;align-items:center;gap:8px}
input[type=color]{width:40px;height:36px;border:none;background:transparent;cursor:pointer;padding:0}
.color-hex{flex:1}
input[type=range]{width:100%;accent-color:#e94560}
video{max-width:100%;border-radius:8px;margin-top:8px}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid #fff6;border-top-color:#fff;border-radius:50%;animation:spin .5s linear infinite;margin-right:6px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
#snippetStatus,#renderStatus{margin-top:10px;font-size:.85em}
.btn-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
</style>
</head>
<body>
<div class="container">
<h1>舞蹈视频智能打码</h1>
<p class="sub">上传 → 实时调参 → 3秒预览 / 全片渲染 (互斥 · 可取消)</p>

<div id="step1" class="card active">
  <h2>Step 1 — 上传视频</h2>
  <input type="file" id="fileInput" accept="video/*">
  <button class="btn btn-render" id="uploadBtn">上传并分析</button>
  <div id="uploadStatus"></div>
</div>

<div id="step2" class="card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
    <h2 style="margin:0">Step 2 — 实时调参预览</h2>
    <button class="btn btn-outline btn-sm" id="reuploadBtn" style="margin:0">↻ 重新上传</button>
  </div>
  <img id="previewImg" class="preview-img" alt="预览">

  <label>打码对象</label>
  <div id="idCheckboxes" class="checkbox-group"></div>

  <label>调参 <span style="color:#e94560;font-size:.8em">(修改即刷新)</span></label>
  <div class="row">
    <div class="col-third">
      <label>模式</label><select id="fillMode"><option value="solid">纯色</option><option value="gradient">渐变</option><option value="blur">模糊</option></select>
    </div>
    <div class="col-third">
      <label>透明度 <span id="opacityVal">100%</span></label><input type="range" id="opacity" min="0" max="100" value="100">
    </div>
    <div class="col-third">
      <label>白边</label><input type="number" id="thickness" value="3" min="1" max="15">
    </div>
  </div>
  <div class="row">
    <div class="col-half">
      <label>填充色</label><div class="color-row"><input type="color" id="fillColor" value="#000000"><input type="text" id="fillColorHex" class="color-hex" value="#000000"></div>
    </div>
    <div class="col-half">
      <label>边框色</label><div class="color-row"><input type="color" id="borderColor" value="#FFFFFF"><input type="text" id="borderColorHex" class="color-hex" value="#FFFFFF"></div>
    </div>
  </div>

  <div class="btn-row">
    <button class="btn btn-snippet" id="snippetBtn">生成3秒预览</button>
    <button class="btn btn-render" id="renderBtn">生成完整视频</button>
  </div>
  <div id="snippetStatus"></div><div id="renderStatus"></div>
</div>

<div id="step3" class="card">
  <h2>Step 3 — 渲染完成</h2>
  <div id="resultArea"></div>
  <button class="btn btn-outline" onclick="location.reload()">处理新视频</button>
</div>
</div>

<script>
let taskId = null, debounceTimer = null;
let snippetCtrl = null, renderCtrl = null;  // AbortController

const $=id=>document.getElementById(id);
const BTN_SNIPPET='snippetBtn', BTN_RENDER='renderBtn';

// ======== resetToStep1 — 严格三层复位 ========
function resetToStep1(){
  // 1. 网络层: 中断正在进行的请求 + 服务端清理
  if(snippetCtrl){ snippetCtrl.abort(); snippetCtrl=null; }
  if(renderCtrl){ renderCtrl.abort(); renderCtrl=null; }
  if(taskId){
    const tid = taskId;
    fetch('/cleanup/'+tid, {method:'DELETE'}).catch(()=>{});
    taskId = null;
  }

  // 2. 数据层: 清空标识、文件input、参数归零
  $('fileInput').value = '';
  clearTimeout(debounceTimer); debounceTimer = null;
  $('opacity').value = 100; $('opacityVal').textContent = '100%';
  $('fillMode').value = 'solid';
  $('fillColor').value = '#000000'; $('fillColorHex').value = '#000000';
  $('borderColor').value = '#FFFFFF'; $('borderColorHex').value = '#FFFFFF';
  $('thickness').value = 3;

  // 3. UI层: 切换面板 + 复位上传按钮 + 清理动态DOM
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

  // 按钮互斥复位
  unlockButtons();
  const sb=$('snippetBtn'), rb=$('renderBtn');
  sb.classList.remove('btn-cancel'); sb.disabled=false;
  sb.innerHTML='生成3秒预览';
  rb.classList.remove('btn-cancel'); rb.disabled=false;
  rb.innerHTML='生成完整视频';
}

// ======== 按钮绑定 ========
$('reuploadBtn').addEventListener('click', resetToStep1);

function getParams(){
  const checks=document.querySelectorAll('#idCheckboxes input[type=checkbox]:checked');
  return {target_ids:Array.from(checks).map(c=>c.value).join(','),fill_mode:$('fillMode').value,fill_color:$('fillColor').value,border_color:$('borderColor').value,thickness:$('thickness').value,opacity:$('opacity').value/100};
}

async function updatePreview(){
  if(!taskId)return;
  const p=getParams(),fd=new FormData();fd.append('task_id',taskId);
  for(const[k,v]of Object.entries(p))fd.append(k,String(v));
  try{const r=await fetch('/preview_frame',{method:'POST',body:fd});const d=await r.json();if(d.image_base64)$('previewImg').src=d.image_base64;}catch(e){}
}
function debounceUpdate(){clearTimeout(debounceTimer);debounceTimer=setTimeout(updatePreview,300);}

// ======== 互斥按钮 + AbortController ========
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
  btn.className = btn.className.replace(/btn-cancel/g,'');
  btn.className = originalClass;
  btn.innerHTML = originalText;
}

function makeFetch(endpoint, btnId, statusId, onSuccess){
  const btn = $(btnId);
  const origText = btn.innerHTML;
  const origClass = btn.className;

  // 如果正在请求中 → 取消
  if (btnId===BTN_SNIPPET && snippetCtrl || btnId===BTN_RENDER && renderCtrl){
    const ctrl = btnId===BTN_SNIPPET ? snippetCtrl : renderCtrl;
    ctrl.abort();
    return;
  }

  // 新建 AbortController
  const ctrl = new AbortController();
  if(btnId===BTN_SNIPPET) snippetCtrl=ctrl; else renderCtrl=ctrl;

  // UI: 锁定另一按钮, 当前按钮变取消态
  lockButtons(btnId);
  setCancelling(btn);
  $(statusId).innerHTML = '';

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
      $(statusId).innerHTML = '<span style="color:#ff6b35">已取消</span>';
    } else {
      $(statusId).innerHTML = `<span style="color:#e94560">错误: ${err.message}</span>`;
    }
  })
  .finally(() => {
    if(btnId===BTN_SNIPPET) snippetCtrl=null; else renderCtrl=null;
    resetButton(btn, origText, origClass);
    unlockButtons();
  });
}

// ======== 按钮绑定 ========
$('snippetBtn').addEventListener('click', ()=>{
  makeFetch('/preview_snippet', BTN_SNIPPET, 'snippetStatus', async r => {
    const blob = await r.blob();
    const snippetUrl = URL.createObjectURL(blob);
    $('snippetStatus').innerHTML = `<video controls autoplay loop muted src="${snippetUrl}"></video>
      <br><a class="btn btn-snippet" href="${snippetUrl}" download="preview_${taskId}.mp4" style="display:inline-block;margin-top:8px;text-decoration:none;padding:6px 14px;font-size:.85em">下载预览片段</a>`;
  });
});

$('renderBtn').addEventListener('click', ()=>{
  makeFetch('/render', BTN_RENDER, 'renderStatus', async r => {
    const blob = await r.blob();
    const videoUrl = URL.createObjectURL(blob);
    $('resultArea').innerHTML = `<video controls src="${videoUrl}" style="max-width:100%;border-radius:8px"></video>
      <br><a class="btn btn-render" href="${videoUrl}" download="anonymized_${taskId}.mp4" style="display:inline-block;margin-top:12px;text-decoration:none">
        下载视频
      </a>`;
    $('step2').classList.remove('active');
    $('step3').classList.add('active');
  });
});

// ======== 上传 ========
$('uploadBtn').addEventListener('click', async ()=>{
  const file=$('fileInput').files[0];if(!file)return alert('请选择视频');
  const btn=$('uploadBtn');btn.disabled=true;btn.innerHTML='<span class="spinner"></span>分析中...';
  const fd=new FormData();fd.append('file',file);
  try{
    const r=await fetch('/analyze',{method:'POST',body:fd});const d=await r.json();
    if(d.error){$('uploadStatus').innerHTML=`<span style="color:#e94560">${d.error}</span>`;return;}
    taskId=d.task_id;$('previewImg').src=d.image_base64;
    const g=$('idCheckboxes');g.innerHTML='';
    d.available_ids.forEach(id=>{g.innerHTML+=`<label><input type="checkbox" value="${id}" checked onchange="debounceUpdate()"> ID: ${id}</label>`;});
    $('step1').classList.remove('active');$('step2').classList.add('active');
  }catch(e){$('uploadStatus').textContent='错误: '+e.message;}
  finally{btn.disabled=false;btn.textContent='上传并分析';}
});

// ======== 参数绑定 ========
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
