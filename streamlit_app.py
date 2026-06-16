"""橙色线圈轮廓检测 — Streamlit Web 调参界面（独立部署版）。

本地运行：
    uv sync
    uv run streamlit run streamlit_app.py

部署到 Streamlit Cloud：
    把本目录推到 GitHub，然后在 https://share.streamlit.io 关联仓库。
    Cloud 会按 uv.lock 还原依赖，按 packages.txt 装中文字体等系统包。

参数持久化策略：
    Streamlit Cloud 容器是临时文件系统，不能直接写盘。所以参数通过
    "下载 params.json" 让用户存到本地，下次 "上传 params.json" 一键恢复。
    本地用户也可以把下载的文件直接放回项目目录，下次启动会自动加载。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from detector import DEFAULTS, Params, process_image

PARAMS_JSON = "params.json"

PARAM_KEYS = [f.name for f in fields(Params)]

st.set_page_config(page_title="橙色线圈轮廓检测调参", layout="wide")


def _load_saved_params() -> Params:
    """启动时尝试从同目录的 params.json 加载（本地用户把下载的文件放回目录就能复用）。"""
    try:
        with open(PARAMS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Params(**{k: int(v) for k, v in data.items() if k in PARAM_KEYS})
    except FileNotFoundError:
        return DEFAULTS
    except Exception as e:
        st.warning(f"读取 {PARAMS_JSON} 失败: {e}，使用默认值")
        return DEFAULTS


def _ensure_state():
    if "params_loaded" in st.session_state:
        return
    init = _load_saved_params()
    for k in PARAM_KEYS:
        st.session_state[k] = getattr(init, k)
    st.session_state["params_loaded"] = True


def _reset_to_defaults():
    for k in PARAM_KEYS:
        st.session_state[k] = getattr(DEFAULTS, k)


def _params_to_json_bytes() -> bytes:
    p = Params(**{k: int(st.session_state[k]) for k in PARAM_KEYS})
    return json.dumps(p.__dict__, indent=2, ensure_ascii=False).encode("utf-8")


def _apply_uploaded_params(raw: bytes) -> tuple[bool, str]:
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        return False, f"解析失败：{e}"
    if not isinstance(data, dict):
        return False, "JSON 顶层不是对象"
    applied = 0
    for k in PARAM_KEYS:
        if k in data:
            try:
                st.session_state[k] = int(data[k])
                applied += 1
            except Exception:
                pass
    if applied == 0:
        return False, "JSON 里没有任何已知参数字段"
    return True, f"已应用 {applied}/{len(PARAM_KEYS)} 个参数"


_ensure_state()

st.title("橙色线圈轮廓检测")
st.caption(
    "上传图片，拖滑动条调参，结果实时刷新。"
    "**下载 params.json** 把当前参数存到本地，下次 **上传 params.json** 一键恢复。"
)

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    uploaded = st.file_uploader("输入图片", type=["jpg", "jpeg", "png", "bmp", "webp"])

    with st.expander("板整体 HSV 范围", expanded=False):
        st.slider("板 H min（色相下限）", 0, 179, key="board_h_min")
        st.slider("板 H max（色相上限）", 0, 179, key="board_h_max")
        st.slider("板 S min（饱和度下限）", 0, 255, key="board_s_min")
        st.slider("板 S max（饱和度上限）", 0, 255, key="board_s_max")
        st.slider("板 V min（明度下限）", 0, 255, key="board_v_min")
        st.slider("板 V max（明度上限）", 0, 255, key="board_v_max")

    with st.expander("橙色线圈 HSV 范围", expanded=True):
        st.slider("线圈 H min（色相下限）", 0, 179, key="coil_h_min")
        st.slider("线圈 H max（色相上限）", 0, 179, key="coil_h_max")
        st.slider("线圈 S min（饱和度下限）", 0, 255, key="coil_s_min")
        st.slider("线圈 S max（饱和度上限）", 0, 255, key="coil_s_max")
        st.slider("线圈 V min（明度下限）", 0, 255, key="coil_v_min")
        st.slider("线圈 V max（明度上限）", 0, 255, key="coil_v_max")

    with st.expander("Canny / 形态学 / 面积", expanded=True):
        st.slider("Canny low（边缘弱阈值）", 0, 255, key="canny_lo")
        st.slider("Canny high（边缘强阈值）", 0, 500, key="canny_hi")
        st.slider("开运算核（奇数，去小白点）", 1, 21, step=2, key="open_ksize")
        st.slider("闭运算核（奇数，连断口）", 1, 31, step=2, key="close_ksize")
        st.slider("闭运算迭代次数", 1, 8, key="close_iters")
        st.slider("最小轮廓面积（去小点）", 0, 5000, step=10, key="min_area")
        st.slider("最大轮廓面积（去超大块）", 100, 50000, step=100, key="max_area")

    btn_dl, btn_reset = st.columns(2)
    btn_dl.download_button(
        "下载 params.json",
        data=_params_to_json_bytes(),
        file_name=PARAMS_JSON,
        mime="application/json",
        type="primary",
        width="stretch",
    )
    if btn_reset.button("恢复默认", width="stretch"):
        _reset_to_defaults()
        st.rerun()

    up = st.file_uploader(
        "上传 params.json 恢复参数",
        type=["json"],
        key="params_upload",
        help="把之前下载的 params.json 拖进来，自动应用到所有滑动条。",
    )
    if up is not None:
        raw = up.getvalue()
        sig = hashlib.md5(raw).hexdigest()
        if st.session_state.get("_last_params_sig") != sig:
            ok, msg = _apply_uploaded_params(raw)
            st.session_state["_last_params_sig"] = sig
            if ok:
                st.toast(msg, icon="✅")
                st.rerun()
            else:
                st.toast(msg, icon="⚠️")

with col_right:
    if uploaded is None:
        st.info("先在左侧上传一张图片。")
    else:
        pil = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
        img_rgb = np.array(pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        p = Params(**{k: int(st.session_state[k]) for k in PARAM_KEYS})
        result_bgr, info = process_image(img_bgr, p)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        st.image(result_rgb, caption="检测结果", width="stretch")

        msg = (
            f"轮廓数: **{info['contour_count']}**　"
            f"匹配框中心: ({info['center_x']:.2f}, {info['center_y']:.2f})　"
            f"图像尺寸: {info['width']} × {info['height']}"
        )
        if "error" in info:
            st.warning(info["error"])
        st.markdown(msg)

        buf = BytesIO()
        Image.fromarray(result_rgb).save(buf, format="PNG")
        st.download_button(
            "下载结果图",
            data=buf.getvalue(),
            file_name="result.png",
            mime="image/png",
        )
