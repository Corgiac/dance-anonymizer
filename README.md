# 舞蹈视频智能打码 🎭

上传舞蹈视频 → 勾选人物 → 一键生成打码。AI 自动追踪全身，快速动作也不会漏。

<img src="assets/dem.jpg" alt="手机端展示" width="320">

<img src="assets/result.GIF" alt="打码效果展示" width="640">

---

## ? 完全没接触过代码，怎么用？

跟着下面步骤走，**全程复制粘贴**，10 分钟搞定。遇到报错直接翻到最下面的「常见问题」。

---

## 第一步：安装 Python（已经装过的跳过）

### Mac 用户

1. 打开 https://www.python.org/downloads/
2. 点那个黄色大按钮下载
3. 双击下载的文件，一路点「继续」→「安装」
4. 安装完关掉窗口

### Windows 用户

1. 打开 https://www.python.org/downloads/
2. 点黄色大按钮下载
3. **? 重要**：安装界面第一页，**勾选底部的「Add Python to PATH」**，然后再点 Install
4. 装完重启电脑（让 PATH 生效）

> ? 怎么确认装好了？  
> Mac：打开「终端」（在启动台搜 Terminal），输入 `python3 --version`，回车。出现 `Python 3.x.x` 就对了。  
> Windows：按 `Win+R`，输入 `cmd`，回车。在黑窗口输入 `python --version`，回车。出现 `Python 3.x.x` 就对了。

---

## 第二步：下载项目

### 如果你有 Git

```bash
git clone https://github.com/Corgiac/dance-anonymizer.git
cd dance-anonymizer
```

### 如果没用过 Git（更简单）

1. 打开 https://github.com/Corgiac/dance-anonymizer
2. 点绿色的 **Code** 按钮 → **Download ZIP**
3. 解压到你喜欢的文件夹（比如桌面）
4. 文件夹名字改成 `dance-anonymizer`

---

## 第三步：一键安装

### Mac 用户

打开「终端」，把下面三行**一行一行**复制进去，每行按回车：

```bash
cd ~/Desktop/dance-anonymizer
```

> ? 如果文件夹不在桌面，把路径换成你的实际位置。拖拽文件夹到终端窗口可以自动填入路径。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e vendor/Cutie
```

看到 `Successfully installed ...` 就说明装好了。

### Windows 用户

按 `Win+R`，输入 `cmd`，回车。在黑窗口里**一行一行**复制，每行按回车：

```bash
cd C:\Users\你的用户名\Desktop\dance-anonymizer
```

> ? 把「你的用户名」换成你电脑的用户名。或者直接拖拽文件夹到黑窗口自动填入路径。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e vendor/Cutie
```

---

## 第四步：启动

**每次使用都要先做这一步**。

### Mac

```bash
cd ~/Desktop/dance-anonymizer
source .venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8002
```

### Windows

```bash
cd C:\Users\你的用户名\Desktop\dance-anonymizer
.venv\Scripts\activate
uvicorn api:app --host 0.0.0.0 --port 8002
```

看到 `Uvicorn running on http://0.0.0.0:8002` 就说明启动成功了。

---

## 第五步：打开网页

浏览器地址栏输入：**`http://localhost:8002`**，回车。

? **手机也能用**：手机和电脑连同一个 WiFi，手机浏览器输入 `http://电脑IP:8002`。  
（Mac：系统设置 → 网络 → 看 IP 地址。Windows：`Win+R` → `cmd` → `ipconfig` → 找 IPv4 地址）

---

## 怎么用

1. 点「上传并分析」，选一个舞蹈视频
2. 勾选你要打码的人（默认全选）
3. 调颜色、白边、透明度（实时预览）
4. 点「生成 3 秒预览」先试一下效果
5. 满意就点「生成完整视频」，等着就行
6. 完成后点「下载视频」

---

## 常见问题

### ? 提示 `pip: command not found`

Python 没装或者没勾选 PATH。回到第一步重装，Windows 一定要勾选「Add Python to PATH」。


### ? 提示 `No module named 'cutie'`

```bash
pip install -e vendor/Cutie
```

### ? 网页打不开 / 显示「无法连接」

1. 确认黑窗口还开着（关掉窗口服务就停了）
2. 确认黑窗口里最后一行是 `Uvicorn running on ...`
3. 确认网址写的是 `http://localhost:8002`（不是 https）

### ? 处理一半报错了 / 卡住了

刷新网页，上传视频重新来一次。大概率是显存不够，尝试处理短一点的视频。

### ? 手机上访问不了

1. 确认手机和电脑连的是**同一个 WiFi**
2. 确认网址格式是 `http://IP:8002`（不是 localhost）
3. Mac 防火墙关了试试：系统设置 → 网络 → 防火墙 → 关闭

### ? 处理速度很慢

正常。1 分钟视频大概需要 3-5 分钟。如果不满意，有 NVIDIA 显卡的话会自动加速。

---

## 给程序员朋友

```bash
git clone https://github.com/Corgiac/dance-anonymizer.git
cd dance-anonymizer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e vendor/Cutie
uvicorn api:app --host 0.0.0.0 --port 8002
```
