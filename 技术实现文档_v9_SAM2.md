# 舞蹈视频智能打码系统 — v9 技术实现文档（SAM 2 混合架构）

> 逐行对应实际代码 | 2026-06-27 | YOLO(首帧) + SAM 2(全片追踪)

---

## 1. 项目结构

```
dance-anonymizer/
├── main.py                # CLI 入口 (130行)
├── api.py                 # FastAPI Web 服务 + SPA 前端 (700+行)
├── config.yaml            # 全局配置文件
├── requirements.txt       # 依赖清单
├── sam2_hiera_tiny.pt     # SAM 2 预训练权重 (~149MB)
├── yolo11s-seg.pt         # YOLO 分割模型 (自动下载)
├── src/
│   ├── __init__.py
│   ├── utils.py           # 视频 I/O + 音频合成 (203行)
│   ├── tracker.py         # 首帧 YOLO 检测器 (150行)
│   ├── effects.py         # 特效渲染 + 深度排序 (259行)
│   └── pipeline.py        # SAM 2 处理流水线 (314行)
└── data/
    ├── input/
    ├── output/
    └── tasks/             # Web 服务任务存储
```

---

## 2. 架构总览

```
                        输入视频
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    步骤1: 全片抽帧    步骤2: SAM2初始化  步骤3: YOLO首帧检测
    cv2.imwrite()      build_sam2_video_  DanceTracker
    → _sam2_frames/      predictor()        .detect_first_frame()
    00000.jpg ...       → predictor         → List[(bbox, id)]
    00NNN.jpg           → inference_state       │
          │                │                   ▼
          │                │           predictor.add_new_points_or_box()
          │                │           注册每个 bbox 到 SAM 2
          │                │                   │
          │                ▼                   │
          │          步骤4: SAM 2 propagate_in_video()
          │          ┌──────────────────────────┘
          │          │  逐帧产出: (frame_idx, [obj_ids], mask_logits)
          │          │
          │          ├─→ torch.sigmoid(logits) → soft alpha [0,1]
          │          ├─→ compute_foot_y() + compute_bbox_from_mask()
          │          ├─→ process_frame_effects() → 填充 + 白边
          │          └─→ VideoWriter.write()
          │                      │
          ▼                      ▼
    步骤5: 音频合并(moviepy) + 清理 _sam2_frames/
          → 成品 MP4 (libx264 + AAC)
```

### 与 v8 (纯 YOLO) 架构对比

| 维度 | v8 纯 YOLO | v9 YOLO + SAM 2 |
|------|-----------|----------------|
| **每帧检测** | YOLO model.track(BoT-SORT) | SAM 2 propagate (无需重复推理) |
| **Mask 来源** | 手工 4 步 Soft Alpha 转换 | SAM 2 sigmoid → 天然精准 |
| **时序一致性** | EMA 35%历史混合 → 拖影 | SAM 2 记忆体 → 零闪烁 |
| **遮挡恢复** | fallback ×0.5 衰减补丁 | SAM 2 内置记忆, 天然恢复 |
| **TemporalMaskCache** | 存在 (复杂状态管理) | **已删除** |
| **TrackerConfig 字段** | 11 个 (含 dead code) | 6 个 |
| **effect_config 键** | 5 键 | 2 键 |

---

## 3. src/tracker.py — 首帧检测器 (150行)

### 3.1 数据结构

```python
@dataclass
class TrackResult:
    """单个人物结果 — mask 由 SAM 2 提供, 此处仅保留兼容结构"""
    track_id: int                           # 从0开始自增
    bbox: Tuple[int, int, int, int]         # xyxy (已膨胀)
    confidence: float                       # YOLO 检测置信度
    mask: np.ndarray                        # (H,W) float32 alpha [0,1]
    foot_y: float = 0.0                     # 脚部y坐标

@dataclass
class TrackerConfig:
    """首帧检测器配置 (仅 YOLO 部分)"""
    model_path: str = "yolo11s-seg.pt"
    device: str = "mps"                     # mps / cuda / cpu
    conf_threshold: float = 0.3
    iou_threshold: float = 0.55
    imgsz: int = 1280
    verbose: bool = True
```

**已删除的旧字段**: `track_buffer`, `iou_match_threshold`, `expected_count`, `ghost_frames`, `half`, `retina_masks`, `min_alpha_sum`, `body_expand_pixels`

### 3.2 DanceTracker 类

```python
class DanceTracker:
    """首帧 YOLO 检测器 — 仅为 SAM 2 提供初始 bbox 注册。"""
```

仅两个方法：
- `detect_first_frame(frame) → List[TrackResult]` — YOLO model.predict(), 提取 bbox + foot_y + 临时 alpha
- `track(frame)` — 兼容旧接口, 内部调用 detect_first_frame()

**关键变化**: 不再有 `_fallback_alphas`、`_last_bbox`、BoT-SORT、时序 fallback、4步 Soft Alpha 转换。YOLO mask 仅用于计算出脚位置, SAM 2 会重新生成最终 mask。

### 3.3 辅助函数

```python
def compute_foot_y(mask: np.ndarray) -> float:
    """从 mask 计算脚部 y 坐标 (alpha 行权重加权底部)"""
    # mask.sum() < 100 → 返回 H
    # 否则: row_weights = mask.sum(axis=1), 取 row_weights > 0.01 的最大行 index

def compute_bbox_from_mask(mask: np.ndarray, expand_ratio=0.05) -> Tuple[int,int,int,int]:
    """从二值/soft mask 提取包围框 (mask>0.3 二值化 + 5%膨胀)"""
```

---

## 4. src/pipeline.py — SAM 2 处理流水线 (314行)

### 4.1 DanceAnonymizerPipeline

```python
class DanceAnonymizerPipeline:
    def __init__(self,
                 tracker_config: TrackerConfig = TrackerConfig(),
                 effect_config: Optional[dict] = None,    # {dilate_kernel_size, temporal_window}
                 sam2_config: Optional[dict] = None):     # {model_path}
```

### 4.2 process() 方法 — 5 步流水线

```
process(input_path, output_path, target_ids, show_progress,
        cancel_event, progress_callback, fill_mode, fill_color,
        border_color, opacity) → str (output_path)
```

**步骤 1: 全片抽帧** (L79-104)
- cv2.VideoCapture 逐帧读取
- 写为 `00000.jpg`, `00001.jpg`, ... 到 `_sam2_frames/` 临时目录
- JPEG quality 95
- progress_callback: `{step:1, step_name:"抽帧"}`

**步骤 2: SAM 2 初始化** (L106-124)
```python
from sam2.build_sam import build_sam2_video_predictor
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
predictor = build_sam2_video_predictor("sam2_hiera_t.yaml", ckpt_path=sam2_model, device=device)
inference_state = predictor.init_state(video_path=frames_dir)
```
- progress_callback: `{step:2, step_name:"加载模型"}`

**步骤 3: YOLO 首帧检测 + SAM 2 注册** (L126-157)
```python
detections = self._tracker.detect_first_frame(first_frame)
for det in detections:
    predictor.add_new_points_or_box(
        inference_state, frame_idx=0, obj_id=tid,
        box=[float(x1), float(y1), float(x2), float(y2)])
```
- progress_callback: `{step:3, step_name:"首帧检测"}`

**步骤 4: SAM 2 传播 + 逐帧渲染** (L159-272)
```python
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    # 逐帧处理:
    for i, obj_id in enumerate(out_obj_ids):
        mask = torch.sigmoid(out_mask_logits[i]).cpu().squeeze().numpy()
        # resize 到视频分辨率 (SAM 2 内部分辨率可能不同)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), INTER_LINEAR)
        mask = np.clip(mask, 0.0, 1.0)
        # 计算 foot_y + bbox → TrackResult

    # 特效渲染 (无 EMA)
    result_frame, _ = process_frame_effects(frame, track_results, ...)
    writer.write(result_frame)

    # 每 5 帧上报进度:
    progress_callback({
        step:4, step_name:"SAM2渲染", step_total:5,
        frames_done, frames_total, elapsed, eta, fps
    })
```
- SAM 2 propagate 是生成器, 无需手动管理帧索引
- mask_logits 是单一 Tensor (N, H_s, W_s), 非 list
- sigmoid 后直接得到天然平滑的 soft alpha [0,1]

**步骤 5: 音频合并 + 清理** (L280-308)
```python
if has_audio_stream(input_path):
    merge_audio_with_moviepy(tmp_video, input_path, output_path)
else:
    os.rename(tmp_video, output_path)
shutil.rmtree(frames_dir)  # 清理临时帧目录
```
- progress_callback: `{step:5, step_name:"合成音频"}`

### 4.3 progress_callback 协议

回调函数签名: `fn(dict) → None`

dict 字段:
| 字段 | 类型 | 说明 |
|------|------|------|
| step | int | 当前步骤 1-5 |
| step_name | str | 步骤名 |
| step_total | int | 总步骤数 (5) |
| frames_done | int | 步骤4已完成帧数 |
| frames_total | int | 总帧数 |
| elapsed | float | 已耗时(秒) |
| eta | float | 预计剩余(秒) |
| fps | float | 处理速度 |

---

## 5. src/effects.py — 特效渲染模块 (259行)

### 5.1 白边提取

```python
def extract_edge_alpha(alpha: np.ndarray, edge_width: int = 3) -> np.ndarray:
    """dilate → subtract → GaussianBlur。输入/输出均为 float32 [0,1]"""
    # 1. alpha > 0.2 二值化
    # 2. MORPH_ELLIPSE(ksize, ksize) dilate
    # 3. edge = dilated - hard
    # 4. GaussianBlur(edge, ksize, σ=ksize/2.5) → 抗锯齿
    # 5. clip [0,1]

def blend_body(alpha: np.ndarray) -> np.ndarray:
    """GaussianBlur(5×5, σ=1.0) 轻量柔化"""

def blend_edge(alpha: np.ndarray, edge_width: int = 3) -> np.ndarray:
    """GaussianBlur(7×7, σ=1.5) 平滑 → extract_edge_alpha()"""
```

### 5.2 深度排序

```python
def calculate_depth_order(track_results, temporal_window=5, smooth_history=None):
    """30帧缓冲取 foot_y 中位数永久冻结"""
    BUFFER_FRAMES = 30
    # 缓冲期: 动态排序
    # 期满: np.median(foot_y) → 升序 (远→近) 冻结
    # 新ID: 按 foot_y 插入正确位置
```

### 5.3 渲染核心

```python
def apply_shadow_outline_effect(
    frame, depth_order, track_id_to_idx, track_results,
    dilate_kernel_size=3, fill_mode="solid",
    fill_color="#000000", border_color="#FFFFFF", opacity=1.0,
) -> np.ndarray:
    """单 Pass 远→近渲染 (无 EMA, 直接使用 SAM 2 mask)"""
    # for tid in depth_order (远→近):
    #   1. blend_body(alpha) → 实体填充 (solid/gradient/blur) × opacity
    #   2. blend_edge(alpha) → 白边 alpha 混合
    # 近处人物自动覆盖远处, 白边紧贴实体
```

### 5.4 便捷封装

```python
def process_frame_effects(
    frame, track_results, smooth_history=None, frame_idx=0,
    dilate_kernel_size=3, temporal_window=8,
    target_ids=None, fill_mode="solid",
    fill_color="#000000", border_color="#FFFFFF", opacity=1.0,
) -> Tuple[np.ndarray, dict]:
    """对单帧应用特效 (SAM 2 版本, 无 temporal_cache)"""
    # 1. target_ids 过滤
    # 2. calculate_depth_order()
    # 3. apply_shadow_outline_effect()
```

**已删除**: `TemporalMaskCache` 类 (40行), `temporal_cache` 参数, `cleanup_stale()` 逻辑

---

## 6. api.py — FastAPI Web 服务 (700+行)

### 6.1 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | SPA 前端页面 (内嵌 HTML/CSS/JS) |
| `/analyze` | POST | 上传视频 → 扫描12帧选最佳 → YOLO首帧检测 → 返回预览图+IDs |
| `/preview_frame` | POST | 单帧重渲染 (无EMA, 预览调参) |
| `/preview_snippet` | POST | 3秒 SAM 2 片段 (async, 可中断) |
| `/render` | POST | 全片 SAM 2 渲染 (async, threading.Event 取消) |
| `/status/{task_id}` | GET | 轮询渲染进度 (step, frames, eta, fps) |
| `/cleanup/{task_id}` | DELETE | 释放服务端资源 |

### 6.2 /analyze 流程

```
1. 保存上传视频
2. 扫描 12 均匀采样帧, YOLO predict 统计人数 → 选最佳帧
3. DanceTracker.detect_first_frame(best_frame) → track_results
4. 默认参数渲染预览图 + 标注 ID 标签
5. 存储任务状态到 TASKS dict
6. 返回 {task_id, image_base64, available_ids, total_frames, fps}
```

### 6.3 /preview_snippet & /render

均使用 `DanceAnonymizerPipeline.process()`, 通过 `progress_callback` 写入 `PROGRESS[task_id]` dict, 前端轮询 `/status/{task_id}` 获取进度。

### 6.4 进度轮询协议

前端每 800ms 请求 `/status/{task_id}`:
```json
{
    "step": 4, "step_name": "SAM2渲染", "step_total": 5,
    "frames_done": 135, "frames_total": 300,
    "elapsed": 22.5, "eta": 27.3, "fps": 6.0,
    "done": false
}
```

### 6.5 前端 UI (SPA)

三步流程 + 进度条:
- **Step 1**: 上传视频
- **Step 2**: 调参预览 (预览图 + ID复选框 + 填充色/边框色/白边宽度/透明度)
  - 生成3秒预览 / 生成完整视频 按钮 (互斥, 可取消)
  - **进度卡片** (点击后出现): 步骤名 + 进度条 + 百分比 + FPS + 预计剩余时间
- **Step 3**: 结果展示 (视频播放器 + 下载)

---

## 7. main.py — CLI 入口 (130行)

```bash
python main.py -i input.mp4 -o output.mp4                    # 默认全片
python main.py -i input.mp4 -o output.mp4 -t 1,3              # 指定ID
python main.py -i input.mp4 -o output.mp4 -d cpu --thickness 7 # CPU + 粗白边
```

参数: `--input`, `--output`, `--config`, `--target_ids`, `--thickness`, `--model`, `--device` (mps/cuda/cpu), `--conf`, `--quiet`, `--temporal_window`

配置优先级: `CLI args > config.yaml > 代码默认值`

---

## 8. config.yaml

```yaml
model:
  path: "yolo11s-seg.pt"
  device: "mps"
  conf_threshold: 0.3
  iou_threshold: 0.55
  imgsz: 1280

sam2:
  model_path: "sam2_hiera_tiny.pt"

effects:
  dilate_kernel_size: 3
  temporal_window: 8

paths:
  input_dir: "data/input"
  output_dir: "data/output"
```

---

## 9. src/utils.py — 视频 I/O (203行, 未改动)

| 类/函数 | 功能 |
|---------|------|
| `VideoReader` | 上下文管理器, 逐帧读取 (frames() 生成器) |
| `VideoWriter` | 上下文管理器, 逐帧写出 |
| `has_audio_stream()` | moviepy 检测音频流 |
| `merge_audio_with_moviepy()` | libx264 + AAC 合成 |
| `get_video_info()` | 返回 {width, height, fps, total_frames, fourcc} |

---

## 10. 完整参数表

### 10.1 YOLO 首帧检测 (TrackerConfig)

| 参数 | 默认值 | 可配置 | 说明 |
|------|--------|--------|------|
| model_path | yolo11s-seg.pt | ✅ | YOLO 模型 |
| device | mps | ✅ | mps/cuda/cpu |
| conf_threshold | 0.3 | ✅ | 检测置信度 |
| iou_threshold | 0.55 | ✅ | NMS IoU |
| imgsz | 1280 | ✅ | 推理分辨率 |
| verbose | True | ✅ | 日志输出 |

### 10.2 SAM 2 (sam2_config)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| model_path | sam2_hiera_tiny.pt | SAM 2 权重路径 |
| config | sam2_hiera_t.yaml | Hydra 配置名 (硬编码) |

### 10.3 特效 (effect_config)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| dilate_kernel_size | 3 | 白边宽度 |
| temporal_window | 8 | 深度排序窗口 |

### 10.4 Pipeline process() 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| input_path | str | 必需 | 输入视频 |
| output_path | str | 必需 | 输出视频 |
| target_ids | List[int] | None (全部) | 目标人物ID |
| show_progress | bool | True | 打印日志 |
| cancel_event | threading.Event | None | 取消信号 |
| progress_callback | fn(dict) | None | Web进度上报 |
| fill_mode | str | "solid" | solid/gradient/blur |
| fill_color | str | "#000000" | 填充色 |
| border_color | str | "#FFFFFF" | 边框色 |
| opacity | float | 1.0 | 不透明度 |

---

## 11. Python 依赖

| 包 | 版本 | 用途 |
|----|------|------|
| torch | >=2.5.0 | 深度学习框架 |
| torchvision | >=0.20.0 | 视觉工具 |
| ultralytics | >=8.0.0 | YOLO 检测 |
| sam-2 | 1.0 (GitHub) | SAM 2 视频分割 |
| opencv-python | >=4.8.0 | 图像/视频处理 |
| moviepy | >=1.0.3 | 音视频合成 |
| numpy | >=1.24.0 | 数值计算 |
| pyyaml | >=6.0 | 配置解析 |
| tqdm | >=4.65.0 | 进度条 |
| fastapi | >=0.100.0 | Web API |
| uvicorn | >=0.20.0 | ASGI 服务器 |
| python-multipart | >=0.0.5 | 文件上传 |

---

## 12. 环境搭建

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装 Python 3.11
uv python install 3.11

# 3. 创建虚拟环境
cd dance-anonymizer/..
uv venv --python 3.11 .venv
source .venv/bin/activate

# 4. 安装依赖
uv pip install torch torchvision ultralytics opencv-python moviepy pyyaml tqdm numpy fastapi uvicorn python-multipart
uv pip install /tmp/sam2  # SAM 2 需手动 clone + install

# 5. 下载权重
# YOLO: 首次运行时自动下载 yolo11s-seg.pt
# SAM 2:
curl -L "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt" -o sam2_hiera_tiny.pt

# 6. 启动服务
cd dance-anonymizer
python -m uvicorn api:app --host 0.0.0.0 --port 8002

# 7. 或 CLI 模式
python main.py -i data/input/dance.mp4 -o data/output/result.mp4
```

---

## 13. 已知局限

1. **SAM 2 首次加载慢**: 初始化需加载 ~149MB 权重 + 读取全部 JPEG 帧
2. **磁盘占用**: 全片抽帧会临时占用约等于视频体积的 JPEG 空间 (渲染完自动清理)
3. **MPS 兼容性**: SAM 2 在 Apple Silicon MPS 上可能有算子兼容问题, 必要时 fallback CPU
4. **首帧依赖**: 如果首帧有人物被遮挡/未检测到, 整个视频该人物都不会被追踪
5. **深度排序冻结**: 30帧缓冲后永久冻结, 无法适应人物前后交叉换位
6. **不支持中途新增人物**: SAM 2 注册仅在首帧, propagate 过程中无法动态添加目标
7. **sam2_hiera_tiny 精度**: 轻量模型, 边缘可能不如 large 版本精准
