# 橙色线圈轮廓检测 — Streamlit 部署版

加热膜铜片 / 橙色线圈轮廓检测的 Web 调参界面。本目录是为部署到 [Streamlit Community Cloud](https://share.streamlit.io) 准备的**独立项目**，不依赖父目录任何文件。

## 文件说明

| 文件 | 作用 |
|---|---|
| `streamlit_app.py` | Streamlit 入口（Streamlit Cloud 默认识别这个文件名） |
| `detector.py` | 核心算法（HSV → Canny → 形态学 → 面积过滤） |
| `pyproject.toml` | Python 项目定义和依赖 |
| `uv.lock` | uv 锁定的精确依赖版本（**必须提交到 git**） |
| `packages.txt` | 系统级 apt 包，云端 Linux 需要中文字体和 OpenCV 运行时 |
| `.streamlit/config.toml` | 上传大小限制、主题等 |

## 本地运行

```bash
cd streamlit-coil-detect
uv sync                       # 创建 .venv 并按 uv.lock 装依赖
uv run streamlit run streamlit_app.py
```

浏览器自动打开 <http://localhost:8501>。

新增/升级依赖：

```bash
uv add some-package           # 新增
uv lock --upgrade-package streamlit   # 升级单个包
uv lock --upgrade             # 升级全部
```

> 没装 uv？`pip install uv` 或参考 <https://docs.astral.sh/uv/getting-started/installation/>。

## 部署到 Streamlit Cloud

1. 把本目录推到 GitHub（**单独一个仓库**最省事，也可以放在父仓库的子目录里）。
2. 登录 <https://share.streamlit.io>，点 **New app**。
3. 填表：
   - Repository：你的仓库
   - Branch：`main`
   - Main file path：`streamlit_app.py`（如果放在子目录就填 `streamlit-coil-detect/streamlit_app.py`）
4. 点 Deploy，约 1~2 分钟构建完成（uv 比 pip 快得多）。

构建期间云端会：
- `apt install` 读取 `packages.txt`（中文字体 + OpenCV 运行时库）
- 检测到 `uv.lock` → 用 **uv** 还原依赖（Streamlit Cloud 现在优先识别 `uv.lock`，比 `requirements.txt` 还靠前）
- 启动 `streamlit run streamlit_app.py`

> ⚠️ 不要在仓库里同时放 `uv.lock` 和 `requirements.txt`/`Pipfile`/`environment.yml`——Streamlit Cloud 只会用它找到的第一个，多个文件会让人混淆。

## 调参速览

界面分三个折叠面板，共 19 个滑动条：

1. **板整体 HSV 范围**：定位两块基板的外接矩形
2. **橙色线圈 HSV 范围**：从板内部筛出橙色线圈区域（**最常调**）
3. **Canny / 形态学 / 面积**：边缘检测 + mask 清理 + 轮廓面积阈值

详细调参方法见同目录下的 [`调参指南.md`](./调参指南.md) 和 [`去噪保轮廓指南.md`](./去噪保轮廓指南.md)。
