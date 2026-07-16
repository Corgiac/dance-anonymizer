# DanceAnon — AI 智能舞蹈视频打码工具

上传视频即可自动识别追踪画面中的人物，支持全身打码、面部贴纸、美白拉腿、自动跟随运镜，Web 界面实时预览并导出。

## 功能

- AI 人物追踪 — 快速舞蹈不丢人
- 全身打码 — 模糊/纯色/渐变，边框与透明度可调
- 面部贴纸 — 内置贴纸 + 自定义上传，大小可调
- 自动跟随 — 镜头平滑跟随选中人物
- 美白拉腿 — 强度实时可调
- 裁剪时长与画幅 — 拖拽选取片段，多比例裁剪

## 安装与启动

### Mac

打开终端，依次运行：

```bash
cd 项目目录
bash scripts/mac/setup.sh
bash scripts/mac/run.sh
```

浏览器打开 `http://localhost:8002`。

可选：`brew install ffmpeg` 以获得音频合成支持。

### Windows

进入 `scripts/windows/` 文件夹：先双击 `setup.bat`，然后双击 `run.bat`。

浏览器打开 `http://localhost:8002`。

可选：从 https://ffmpeg.org 下载 ffmpeg 并添加到 PATH，以获得音频合成支持。

> 没有 NVIDIA 显卡的电脑首次处理可能需等待 5-10 分钟，不是卡死。

### 手动安装

```bash
cd 项目目录
python3 -m venv .venv
source .venv/bin/activate      # Mac
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
python -m uvicorn api:app --host 0.0.0.0 --port 8002
```

## 使用说明

1. 点「选取视频」选择一个视频
2. 拖动裁剪条选取片段，点「上传并分析」
3. 勾选要打码的人（默认全选）
4. 调节填充模式、颜色、透明度、美白、拉腿
5. 可选：给未打码的人添加面部贴纸
6. 可选：点「跟随」让镜头跟随某个人物
7. 点「生成完整视频」，等待处理完成
8. 在结果页可选裁剪画幅比例，然后保存下载

> 💡 **提示**：需要打码的人必须在视频首帧内才能被 AI 追踪。建议把视频开头截到所有人物都在画面里的位置。同时人物建议控制在 8 人以内。

## 可能遇到的问题

### 网页打不开 / 显示「无法连接」

1. 确认终端窗口还开着（关掉窗口服务就停了）
2. 确认地址是 `http://localhost:8002`（不是 https）

### 提示 `address already in use`

```bash
# Mac
lsof -ti:8002 | xargs kill -9
bash scripts/mac/run.sh

# Windows
关掉所有终端窗口，重新双击 run.bat
```

### 处理速度慢

| 设备 | 视频时长 | 处理耗时 |
|------|---------|---------|
| Mac (Apple M 芯片) | 25 秒 | ~160 秒 |
| Windows (纯 CPU) | 10 秒 | ~400 秒 |

有 NVIDIA 显卡可大幅加速。

## 技术栈

Python · FastAPI · OpenCV · YOLO · CUTIE · SAM2 · PyTorch
