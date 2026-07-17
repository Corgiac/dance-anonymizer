# DanceAnon — AI 智能舞蹈视频打码工具

上传视频即可自动识别追踪画面中的人物，支持全身打码、面部贴纸、美白拉腿、自动跟随运镜，Web 界面实时预览并导出。

## 系统要求

- **操作系统**：macOS / Windows / Linux
- **Python**：3.10 或以上
- **内存**：建议 8GB 以上
- **显卡**：Apple M 系列芯片（MPS 加速）、NVIDIA 显卡（CUDA 加速）。纯 CPU 也能运行，只是较慢。

### 安装 Python

如果电脑上还没有 Python 3.10+，先安装：

**Mac**：打开终端，输入：

```bash
brew install python@3.11
```

没有 Homebrew？先访问 https://brew.sh 按提示安装，再执行上面命令。或者直接从 https://www.python.org/downloads/ 下载 macOS 安装包。

**Windows**：访问 https://www.python.org/downloads/ 下载安装包，**安装时务必勾选「Add Python to PATH」**，然后一直下一步即可。

## 快速开始

### 第一步：下载项目

点击页面右上角绿色 `Code` 按钮 → `Download ZIP`，解压到任意文件夹。

### 第二步：安装环境

#### Mac 用户

1. 打开「终端」（在 启动台 → 其他 → 终端，或按 `Cmd + 空格` 搜索"终端"）
2. 把解压后的文件夹拖到终端窗口里，会显示类似 `/Users/xxx/dance-anonymizer` 的路径
3. 依次复制粘贴以下命令（每行粘贴后按回车）：

```bash
cd （这里把文件夹拖进来，会自动填路径）
bash scripts/mac/setup.sh
```

看到 `安装完成！` 就说明装好了。

> ⚠️ **必要步骤**：还需下载 AI 模型文件 `sam2_hiera_tiny.pt`（约 150MB），放到项目根目录（和 `api.py` 同级）。
> 下载地址：https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt

#### Windows 用户

1. 进入解压后的文件夹 → `scripts` → `windows`
2. **双击 `setup.bat`**，等待安装完成
3. 看到 `Setup complete!` 后按任意键关闭窗口

> ⚠️ **必要步骤**：还需下载 AI 模型文件 `sam2_hiera_tiny.pt`（约 150MB），放到项目根目录（和 `api.py` 同级）。
> 下载地址：https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt

### 第三步：启动服务

#### Mac 用户

在终端输入：

```bash
cd （这里把文件夹拖进来）
bash scripts/mac/run.sh
```

#### Windows 用户

回到 `scripts/windows/` 文件夹，**双击 `run.bat`**。

看到 `Uvicorn running on http://0.0.0.0:8002` 说明启动成功。

### 第四步：打开网页

浏览器地址栏输入 **`http://localhost:8002`** 回车。

> 📱 **手机也能用**：手机和电脑连同一个 WiFi，手机浏览器输入 `http://电脑IP:8002`。
>
> Mac 查看 IP：系统设置 → 网络 → Wi-Fi → 详细信息 → IP 地址
> Windows 查看 IP：`Win + R` → 输入 `cmd` → 输入 `ipconfig` → 找 IPv4 地址

## 使用指南

### 1. 上传视频

点击「选取或拖拽视频到此处」，选择一个舞蹈视频。支持 mp4、mov 等常见格式。

### 2. 裁剪时长

拖动紫色横条上的白色圆点，选取需要处理的片段。视频越短处理越快。

> ⚠️ **重要**：需要打码的人必须出现在首帧画面中，否则 AI 无法追踪。

### 3. 配置打码

上传分析后进入配置页：

- **勾选人物**：默认全选，取消勾选的人不会被全身打码
- **填充模式**：模糊 / 纯色 / 渐变
- **边线宽度**：人物边缘白色描边的粗细
- **填充色 / 边框色**：点击可展开色板选择
- **透明度**：打码区域的透明度

### 4. 面部贴纸（可选）

取消勾选某个人物后，旁边会出现「面部」按钮。点亮后可在该人物面部叠加贴纸：

- 默认使用内置贴纸
- 可点击「自定义」上传自己的图片（支持 PNG / JPG / WebP，PNG 透明背景效果最好）
- 「大小」滑条可调节贴纸尺寸

### 5. 美白 & 拉腿（可选）

- **美白**：0-100% 滑条，在人物皮肤区域提亮
- **拉腿**：勾选「启用」，调节程度和区域范围，在人物腿部区域纵向拉伸

### 6. 自动跟随（可选）

点击某个人物旁边的「跟随」按钮（变紫色即启用）。生成视频时镜头会自动跟随该人物移动，始终保持居中。

### 7. 生成视频

点击「生成完整视频」，等待进度条跑完。然后进入结果页：

- 可直接播放预览
- 可先选裁剪比例（1:1 / 4:3 / 16:9 / 9:16）再下载
- 点击「保存视频」下载到本地

## 常见问题

### 安装时报错 / pip 安装失败

1. **Python 版本太低**：本项目需要 Python 3.10 或以上。在终端/命令行输入 `python --version` 检查版本。如果低于 3.10，去 https://www.python.org/downloads/ 下载最新版。
2. **网络问题（国内用户）**：如果开了 VPN 或代理，先关掉再试。setup 脚本会自动换清华镜像重试。
3. **手动安装**：如果脚本一直失败，参考下方「手动安装」步骤。

### 网页打不开 / 显示"无法连接"

1. 确认终端窗口（黑窗口）还开着——关掉窗口服务就停了
2. 确认地址是 `http://localhost:8002`（不是 https）
3. 如果修改过端口，确认用的是哪个端口

### 提示 `address already in use`（端口被占用）

**Mac**：在终端输入以下命令，再重新运行 `run.sh`：
```bash
lsof -ti:8002 | xargs kill -9
bash scripts/mac/run.sh
```

**Windows**：关掉所有命令提示符窗口，重新双击 `run.bat`。

### 处理到一半报错 / 卡住

刷新网页，重新上传视频再试一次。如果频繁出现，尝试处理短一点的视频片段。

### 处理速度很慢

| 设备 | 视频时长 | 处理耗时 |
|------|---------|---------|
| Mac (Apple M 芯片，GPU 加速) | 25 秒 | ~160 秒 |
| Windows (NVIDIA 显卡，GPU 加速) | 25 秒 | ~120 秒 |
| Windows / Mac (纯 CPU) | 10 秒 | ~400 秒 |

有独立显卡可以大幅加速。视频内同时出现的人物建议在 8 人以内。

### 生成的视频没有声音

运行安装脚本会自动装好 ffmpeg。如仍有问题，重新运行一次 setup 脚本即可。

### 手机访问不了

1. 确认手机和电脑连接的是**同一个 WiFi**
2. 网址格式是 `http://电脑IP:8002`（不是 localhost）
3. Mac 防火墙可能阻止访问：系统设置 → 网络 → 防火墙 → 关闭

## 手动安装（进阶）

如果一键脚本执行失败，可以手动安装：

### Mac

```bash
cd 项目文件夹路径
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn api:app --host 0.0.0.0 --port 8002
```

### Windows

```bash
cd 项目文件夹路径
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn api:app --host 0.0.0.0 --port 8002
```

## 模型文件说明

本项目依赖以下 AI 模型：

- `yolo11s-seg.pt` — 人物检测（项目自带）
- `sam2_hiera_tiny.pt` — SAM2 首帧 mask 精修（需自行下载）
- CUTIE 模型 — 人物跨帧追踪（`vendor/Cutie/weights/` 目录，项目自带）

其中 `sam2_hiera_tiny.pt` 因超过 GitHub 100MB 限制未包含在仓库中，请下载后放到项目根目录：

- 下载地址：https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt

## 技术栈

Python · FastAPI · OpenCV · YOLO · CUTIE · SAM2 · PyTorch
