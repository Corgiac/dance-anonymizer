# 舞蹈视频智能打码 🎭

上传舞蹈视频 → 勾选人物 → AI 自动追踪全身生成打码/特效。手机电脑都能用。

## 快速开始

### 1. 环境要求

- Python 3.10+
- macOS / Linux（Windows 未测试）
- 8GB 以上内存

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/Corgiac/dance-anonymizer.git
cd dance-anonymizer

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装 Python 依赖
pip install -r requirements.txt

# 安装 SAM2（首帧抠图精修）
pip install git+https://github.com/facebookresearch/sam2.git

# 克隆 Cutie（全片追踪引擎）
git clone https://github.com/hkchengrex/Cutie.git vendor/Cutie

# 安装 Cutie 依赖
pip install -e vendor/Cutie
```


### 4. 启动

```bash
uvicorn api:app --host 0.0.0.0 --port 8002
```

打开浏览器访问 `http://localhost:8002`。

手机访问：电脑和手机连同一 WiFi，手机浏览器打开 `http://<电脑IP>:8002`。

## 配置

编辑 `config.yaml`：

```yaml
model:
  device: "mps"    # macOS Apple Silicon。NVIDIA 显卡改为 "cuda"，无 GPU 改为 "cpu"
tracking:
  engine: "cutie"   # 追踪引擎（cutie / sam2）
```

## 效果演示

| 纯色打码 | 模糊效果 | 渐变填充 |
|---------|---------|---------|
| 纯黑覆盖 | 高斯模糊 | 渐变过渡 |

## 项目结构

```
dance-anonymizer/
├── api.py              # FastAPI Web 服务 + 前端界面
├── main.py             # CLI 命令行入口
├── config.yaml         # 全局配置
├── requirements.txt    # Python 依赖
├── src/
│   ├── effects.py      # 特效渲染（打码/白边/标签）
│   ├── engine.py       # 追踪引擎（SAM2/Cutie）
│   ├── pipeline.py     # 处理流水线
│   ├── tracker.py      # YOLO 首帧检测
│   └── utils.py        # 视频 I/O 工具
└── vendor/Cutie/       # Cutie 追踪引擎（需手动克隆）
```

## 常见问题

**Q: 启动后访问不了？**
A: 检查防火墙是否放行 8002 端口，确认启动命令包含 `--host 0.0.0.0`。

**Q: 处理很慢？**
A: 确保 `config.yaml` 中 `device` 设置正确。macOS M 系列芯片用 `mps`，NVIDIA 显卡用 `cuda`。

**Q: 视频生成后无法下载？**
A: 渲染完成后会在页面底部显示下载按钮，点击选择画质即可下载。
