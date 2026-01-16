# transunet_viewer_ct_mn_ft.py
import sys
import os
import re
import math
import contextlib
from typing import Tuple, Optional, Dict, Any, List

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QGroupBox, QHBoxLayout, QVBoxLayout, QSpinBox, QTabWidget, QGridLayout,
    QFrame, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


SIZE = 350

# =========================
# CKPT paths (absolute from this .py)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRANSUNET_CKPT_PATH = os.path.join(
    BASE_DIR,
    "runs_transunet_ct_mn_ft/exp_20260115_215111/checkpoints/best_0.8188.pt"
)

MN_TRANSUNET_CKPT_PATH = os.path.join(
    BASE_DIR,
    "runs_transunet_mn/exp_20260115_215915/checkpoints/best_0.7520.pt"
)

# 初始預設（若 ckpt 自動推回成功，這些會被覆寫）
TRANSUNET_IMG_SIZE = (256, 256)   # (H, W)
TRANSUNET_IN_CHANNELS = 2         # T1 + T2
TRANSUNET_NUM_CLASSES = 3         # multi-label: 0=CT, 1=MN, 2=FT
TRANSUNET_BASE = 32

TRANSUNET_MIXED_PRECISION = True  # CUDA 才會開

TRANSUNET_THR_CT = 0.5
TRANSUNET_THR_MN = 0.2
TRANSUNET_THR_FT = 0.3

MN_KEEP_ONE_COMPONENT = True
MN_W_AREA = 0.1
MN_W_TOP = 0.3
MN_W_RIGHT = 0.3
MN_MIN_AREA = 0

CT_KEEP_ONE_COMPONENT = True
CT_MIN_AREA = 200  # 小於這面積的連通元件直接丟掉

FT_FROM_CT_DARK_IN_T2_ENABLE = True
FT_T2_MIN_PIXELS_IN_CT = 200
FT_T2_ROI_P_LOW = 1.0
FT_T2_ROI_P_HIGH = 99.0

# 對比強化方式：Sigmoid
FT_T2_USE_SIGMOID_CONTRAST = True
FT_T2_SIGMOID_K = 8.0  # 越大對比越強

# 若不用 Sigmoid：改用線性對比
FT_T2_LINEAR_ALPHA = 1.8
FT_T2_LINEAR_BETA = 0.0

# 用強化後的 ROI 取「暗」門檻：PCT=40 表示取最暗 40% 當 FT
FT_T2_DARK_PCT_IN_CT_FOR_FT = 40.0

# FT 通常在 CT 內部，先把 CT 往內縮避免邊界誤判
FT_CT_ERODE_K = 3
FT_CT_ERODE_IT = 1

# 暗區 mask 清理
FT_OPEN_K = 3
FT_OPEN_IT = 1
FT_CLOSE_K = 7
FT_CLOSE_IT = 0
FT_KEEP_LARGEST = False

# 若 CT 太小/估不到門檻 → 要不要回退用 model 的 FT
FT_FALLBACK_TO_MODEL_IF_FAIL = True

# ============================================================
# CT contour smooth
# ============================================================
CT_CONTOUR_SMOOTH_ENABLE = True
CT_CONTOUR_MEDIAN_K = 5
CT_CONTOUR_CLOSE_K = 5
CT_CONTOUR_CLOSE_IT = 1
CT_CONTOUR_EPS_RATIO = 0.003
CT_CONTOUR_THICKNESS = 1


# =========================
# AMP compatibility helper
# =========================
try:
    from torch import amp as _amp
    _HAS_TORCH_AMP = True
except Exception:
    _HAS_TORCH_AMP = False
    _amp = None


def get_autocast(device_type: str, enabled: bool):
    """
    ✅ 修正：CPU 上不要硬開 cuda.amp.autocast，避免例外
    """
    if not enabled:
        return contextlib.nullcontext()

    if _HAS_TORCH_AMP:
        # torch.amp.autocast supports device_type="cuda"/"cpu"
        return _amp.autocast(device_type=device_type, enabled=True)

    # fallback：沒有 torch.amp 就只支援 cuda autocast
    if device_type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)

    return contextlib.nullcontext()


def make_image_box(size=SIZE):
    lbl = QLabel()
    lbl.setFixedSize(size, size)
    lbl.setFrameStyle(QFrame.Box | QFrame.Plain)
    lbl.setLineWidth(3)
    lbl.setAlignment(Qt.AlignCenter)
    return lbl


def draw_mask(image, mask):
    masked_image = image.copy()
    mask_bool = mask.astype(bool)
    masked_image[mask_bool] = (0, 255, 0)
    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)


def _ensure_odd(k: int) -> int:
    k = int(max(1, k))
    if k % 2 == 0:
        k += 1
    return k


def _smooth_mask_for_contour(mask_u8_255: np.ndarray,
                             median_k: int,
                             close_k: int,
                             close_it: int) -> np.ndarray:
    m = mask_u8_255.copy()
    mk = _ensure_odd(median_k)
    if mk >= 3:
        m = cv2.medianBlur(m, mk)

    ck = _ensure_odd(close_k)
    if ck >= 3 and close_it > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=int(close_it))

    m = (m > 0).astype(np.uint8) * 255
    return m


def _draw_contours_aa(bgr: np.ndarray,
                      contours,
                      thickness: int = 1,
                      eps_ratio: Optional[float] = None):
    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue
        if eps_ratio is not None and eps_ratio > 0:
            peri = cv2.arcLength(cnt, True)
            eps = float(eps_ratio) * peri
            cnt = cv2.approxPolyDP(cnt, eps, True)
        if cnt is None or len(cnt) < 3:
            continue
        cv2.polylines(
            bgr, [cnt], isClosed=True,
            color=(0, 0, 255),
            thickness=int(thickness),
            lineType=cv2.LINE_AA
        )


def draw_predict_mask(base_img, gt_mask, pred_mask, kind: str = ""):
    overlay = draw_mask(base_img, gt_mask)

    mask_u8 = (pred_mask.astype(np.uint8) * 255)

    if kind == "CT" and CT_CONTOUR_SMOOTH_ENABLE:
        mask_u8 = _smooth_mask_for_contour(
            mask_u8,
            median_k=CT_CONTOUR_MEDIAN_K,
            close_k=CT_CONTOUR_CLOSE_K,
            close_it=CT_CONTOUR_CLOSE_IT
        )
    else:
        mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    if kind == "CT" and CT_CONTOUR_SMOOTH_ENABLE:
        _draw_contours_aa(
            bgr, contours,
            thickness=CT_CONTOUR_THICKNESS,
            eps_ratio=CT_CONTOUR_EPS_RATIO
        )
    else:
        _draw_contours_aa(bgr, contours, thickness=1, eps_ratio=None)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def norm01_robust(img_u8: np.ndarray) -> np.ndarray:
    img = img_u8.astype(np.float32)
    lo, hi = np.percentile(img, (1.0, 99.0))
    if hi <= lo:
        lo = float(img.min())
        hi = float(img.max()) + 1e-6
    img = (img - lo) / (hi - lo + 1e-6)
    img = np.clip(img, 0.0, 1.0)
    return img.astype(np.float32)


def keep_one_component_largest(mask_u8: np.ndarray, min_area: int = 0) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    if not np.any(m):
        return m

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    if num == 2:
        area = int(stats[1, cv2.CC_STAT_AREA])
        if area < int(min_area):
            return np.zeros_like(m)
        return m

    candidates = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            candidates.append((i, area))

    if candidates:
        best_i = max(candidates, key=lambda x: x[1])[0]
        return (labels == best_i).astype(np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_i = 1 + int(np.argmax(areas))
    return (labels == best_i).astype(np.uint8)


def _morph_open_close(mask_u8: np.ndarray,
                      open_k: int, open_it: int,
                      close_k: int, close_it: int) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8) * 255

    ok = _ensure_odd(open_k)
    ck = _ensure_odd(close_k)

    if open_k > 0 and open_it > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=int(open_it))

    if close_k > 0 and close_it > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=int(close_it))

    return (m > 0).astype(np.uint8)


def _keep_largest_cc(mask_u8: np.ndarray) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    if not np.any(m):
        return m
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    out = (labels == idx).astype(np.uint8)
    return out


def _erode_mask(mask_u8: np.ndarray, k: int, it: int) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8) * 255
    k = _ensure_odd(k)
    if k >= 3 and it > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.erode(m, ker, iterations=int(it))
    return (m > 0).astype(np.uint8)


def _roi_robust_norm01(img_u8: np.ndarray, roi_mask_u8: np.ndarray,
                       p_low: float, p_high: float) -> Optional[Tuple[np.ndarray, float, float]]:
    roi = (roi_mask_u8 > 0)
    if not np.any(roi):
        return None

    vals = img_u8[roi].astype(np.float32)
    if vals.size < int(FT_T2_MIN_PIXELS_IN_CT):
        return None

    lo, hi = np.percentile(vals, (float(p_low), float(p_high)))
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max()) + 1e-6

    img = img_u8.astype(np.float32)
    norm = (img - lo) / (hi - lo + 1e-6)
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
    return norm, lo, hi


def _contrast_sigmoid(x01: np.ndarray, k: float) -> np.ndarray:
    k = float(k)
    z = k * (x01.astype(np.float32) - 0.5)
    y = 1.0 / (1.0 + np.exp(-z))
    return y.astype(np.float32)


def _contrast_linear(x01: np.ndarray, alpha: float, beta: float = 0.0) -> np.ndarray:
    a = float(alpha)
    b = float(beta)
    y = (x01.astype(np.float32) - 0.5) * a + 0.5 + b
    y = np.clip(y, 0.0, 1.0)
    return y.astype(np.float32)


def enhance_t2_in_ct_roi(t2_u8: np.ndarray, ct_mask_u8: np.ndarray) -> Optional[np.ndarray]:
    out = _roi_robust_norm01(t2_u8, ct_mask_u8, FT_T2_ROI_P_LOW, FT_T2_ROI_P_HIGH)
    if out is None:
        return None
    t2_norm, _, _ = out

    if FT_T2_USE_SIGMOID_CONTRAST:
        t2_enh01 = _contrast_sigmoid(t2_norm, FT_T2_SIGMOID_K)
    else:
        t2_enh01 = _contrast_linear(t2_norm, FT_T2_LINEAR_ALPHA, FT_T2_LINEAR_BETA)

    t2_enh_u8 = (t2_enh01 * 255.0 + 0.5).astype(np.uint8)
    return t2_enh_u8


def estimate_dark_thr_from_enhanced_t2_in_ct(enh_t2_u8: np.ndarray, ct_mask_u8: np.ndarray) -> Optional[float]:
    roi = (ct_mask_u8 > 0)
    if not np.any(roi):
        return None
    vals = enh_t2_u8[roi].astype(np.float32)
    if vals.size < int(FT_T2_MIN_PIXELS_IN_CT):
        return None
    thr = float(np.percentile(vals, float(FT_T2_DARK_PCT_IN_CT_FOR_FT)))
    return thr


def derive_ft_from_ct_dark_in_t2(t2_u8: np.ndarray, ct_mask_u8: np.ndarray) -> np.ndarray:
    ct = (ct_mask_u8 > 0).astype(np.uint8)
    if not np.any(ct):
        return np.zeros_like(ct)

    ct_inner = ct
    if FT_CT_ERODE_K > 0 and FT_CT_ERODE_IT > 0:
        ct_inner = _erode_mask(ct, FT_CT_ERODE_K, FT_CT_ERODE_IT)
        if not np.any(ct_inner):
            ct_inner = ct

    enh = enhance_t2_in_ct_roi(t2_u8, ct_inner)
    if enh is None:
        return np.zeros_like(ct)

    thr = estimate_dark_thr_from_enhanced_t2_in_ct(enh, ct_inner)
    if thr is None:
        return np.zeros_like(ct)

    ft = ((enh.astype(np.float32) <= thr) & (ct_inner > 0)).astype(np.uint8)

    ft = _morph_open_close(ft, FT_OPEN_K, FT_OPEN_IT, FT_CLOSE_K, FT_CLOSE_IT)

    if FT_KEEP_LARGEST:
        ft = _keep_largest_cc(ft)

    return (ft > 0).astype(np.uint8)


def keep_one_component_weighted(mask_u8: np.ndarray,
                                w_area: float = 0.5,
                                w_top: float = 0.25,
                                w_right: float = 0.25,
                                min_area: int = 30) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    if not np.any(m):
        return m

    H, W = m.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 2:
        return m

    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        comps.append((i, area, cx, cy))

    if not comps:
        return _keep_largest_cc(m)

    max_area = max(c[1] for c in comps) + 1e-6
    denom_h = float(max(1, H - 1))
    denom_w = float(max(1, W - 1))

    best_i = comps[0][0]
    best_score = -1e9

    for (i, area, cx, cy) in comps:
        area_norm = float(area) / float(max_area)
        topness = 1.0 - float(cy) / denom_h
        rightness = float(cx) / denom_w
        score = float(w_area) * area_norm + float(w_top) * topness + float(w_right) * rightness
        if score > best_score:
            best_score = score
            best_i = i

    out = (labels == best_i).astype(np.uint8)
    return out


# ============================================================
# ✅ ckpt state_dict 對齊工具
# ============================================================
def _strip_known_prefixes(sd: Dict[str, torch.Tensor], prefixes: List[str]) -> Dict[str, torch.Tensor]:
    if not isinstance(sd, dict) or not sd:
        return sd

    out = sd
    for p in prefixes:
        keys = list(out.keys())
        if not keys:
            break
        cnt = sum(1 for k in keys if isinstance(k, str) and k.startswith(p))
        if cnt >= int(0.8 * len(keys)):
            out = {k[len(p):]: v for k, v in out.items() if isinstance(k, str)}
    return out


def _extract_state_dict_and_cfg(ck: Any) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
    cfg = None
    if isinstance(ck, dict):
        cfg = ck.get("cfg", None)
        if "model_state" in ck and isinstance(ck["model_state"], dict):
            sd = ck["model_state"]
        elif "state_dict" in ck and isinstance(ck["state_dict"], dict):
            sd = ck["state_dict"]
        elif "model_state_dict" in ck and isinstance(ck["model_state_dict"], dict):
            sd = ck["model_state_dict"]
        else:
            maybe_sd = {k: v for k, v in ck.items() if isinstance(v, torch.Tensor)}
            sd = maybe_sd if maybe_sd else ck
    else:
        sd = ck

    if not isinstance(sd, dict):
        raise ValueError("Checkpoint is not a state_dict-like object.")

    sd = _strip_known_prefixes(sd, prefixes=["module.", "model.", "net."])
    return sd, (cfg if isinstance(cfg, dict) else None)


def _find_key_endswith(sd: Dict[str, torch.Tensor], suffix: str) -> Optional[str]:
    for k in sd.keys():
        if isinstance(k, str) and k.endswith(suffix):
            return k
    return None


def _infer_trans_blocks(sd: Dict[str, torch.Tensor], prefix: str = "trans_blocks.") -> int:
    max_i = -1
    pat = re.compile(r".*" + re.escape(prefix) + r"(\d+)\.")
    for k in sd.keys():
        if not isinstance(k, str):
            continue
        m = pat.match(k)
        if m:
            idx = int(m.group(1))
            max_i = max(max_i, idx)
    return max_i + 1 if max_i >= 0 else 0


def _factorize_token_len(L: int) -> Tuple[int, int]:
    """
    把 token_len L 分解成 (h, w) 最接近 sqrt 的一組
    """
    L = int(L)
    if L <= 0:
        return 16, 16
    r = int(np.sqrt(L))
    for h in range(r, 0, -1):
        if L % h == 0:
            w = L // h
            return int(h), int(w)
    return r, max(1, L // max(1, r))


def _pick_heads(embed_dim: int, preferred: int = 8) -> int:
    embed_dim = int(embed_dim)
    preferred = int(max(1, preferred))
    if embed_dim % preferred == 0:
        return preferred
    for h in [8, 6, 4, 3, 2, 1, 12, 16]:
        if embed_dim % h == 0:
            return int(h)
    return 1


def _is_vit_transunet_ckpt(sd: Dict[str, torch.Tensor]) -> bool:
    if _find_key_endswith(sd, "patch_proj.weight") is not None:
        return True
    if _find_key_endswith(sd, "bridge.net.0.weight") is not None:
        return True
    if _find_key_endswith(sd, "pos_embed") is not None and any(isinstance(k, str) and k.startswith("trans_blocks.") for k in sd.keys()):
        return True
    return False


def _infer_vittransunet_cfg_from_ckpt(
    sd: Dict[str, torch.Tensor],
    cfg: Optional[Dict[str, Any]],
    default_img_size=(256, 256),
    default_heads=8,
    default_mlp_ratio=4.0,
    default_dropout=0.1,
) -> Dict[str, Any]:
    """
    ✅ 修正重點：
    - 回傳 dict 一定包含：
      POS_HW / PATCH_IN_CH / BRIDGE_OUT_CH / BRIDGE_K
    - 這樣 load_transunet_once() 不會再 KeyError
    """

    # base + in_ch
    k_d1 = _find_key_endswith(sd, "d1.net.0.weight")
    if k_d1 is None:
        raise KeyError("Cannot find key 'd1.net.0.weight' in checkpoint state_dict.")
    w_d1 = sd[k_d1]
    base = int(w_d1.shape[0])
    in_ch = int(w_d1.shape[1])

    # out channels
    k_out = _find_key_endswith(sd, "out.weight")
    if k_out is None:
        raise KeyError("Cannot find key 'out.weight' in checkpoint state_dict.")
    out_ch = int(sd[k_out].shape[0])

    # vit_dim from patch_proj
    k_pp = _find_key_endswith(sd, "patch_proj.weight")
    if k_pp is None:
        raise KeyError("Cannot find key 'patch_proj.weight' in checkpoint state_dict.")
    vit_dim = int(sd[k_pp].shape[0])

    # patch_in_ch from patch_proj input
    patch_in_ch = int(sd[k_pp].shape[1])

    # vit_depth from trans_blocks
    vit_depth = _infer_trans_blocks(sd, prefix="trans_blocks.")
    if vit_depth <= 0:
        raise KeyError("Cannot infer vit_depth: no trans_blocks.* found in ckpt.")

    # img size from cfg OR pos_embed length
    img_size = tuple(default_img_size)

    # 如果 training cfg 裡面有 IMG_SIZE → 直接用（最準）
    if isinstance(cfg, dict) and "IMG_SIZE" in cfg:
        v = cfg["IMG_SIZE"]
        if isinstance(v, (list, tuple)) and len(v) == 2:
            img_size = (int(v[0]), int(v[1]))

    # 否則用 pos_embed 的 token_len 推回 (H/16,W/16)
    k_pe = _find_key_endswith(sd, "pos_embed")
    if k_pe is not None and not (isinstance(cfg, dict) and "IMG_SIZE" in cfg):
        token_len = int(sd[k_pe].shape[1])
        ph, pw = _factorize_token_len(token_len)   # (H/16, W/16)
        img_size = (int(ph * 16), int(pw * 16))

    # heads/mlp/dropout/thr from cfg (training keys)
    if isinstance(cfg, dict):
        vit_heads = int(cfg.get("VIT_HEADS", default_heads))
        vit_mlp_ratio = float(cfg.get("VIT_MLP_RATIO", default_mlp_ratio))
        vit_dropout = float(cfg.get("VIT_DROPOUT", default_dropout))

        thr_ct = float(cfg.get("THR_CT", TRANSUNET_THR_CT))
        thr_mn = float(cfg.get("THR_MN", TRANSUNET_THR_MN))
        thr_ft = float(cfg.get("THR_FT", TRANSUNET_THR_FT))
    else:
        vit_heads = int(default_heads)
        vit_mlp_ratio = float(default_mlp_ratio)
        vit_dropout = float(default_dropout)
        thr_ct, thr_mn, thr_ft = TRANSUNET_THR_CT, TRANSUNET_THR_MN, TRANSUNET_THR_FT

    vit_heads = _pick_heads(vit_dim, preferred=vit_heads)

    # ✅ POS_HW 一定要補：它就是 H/16, W/16
    pos_hw = (int(img_size[0] // 16), int(img_size[1] // 16))

    # ✅ bridge kernel/out_ch：從 bridge.net.0.weight 推（推不到就用預設）
    bridge_out_ch = int(vit_dim)
    bridge_k = 3
    k_b0 = _find_key_endswith(sd, "bridge.net.0.weight")
    if k_b0 is not None:
        wb = sd[k_b0]
        if hasattr(wb, "shape") and len(wb.shape) == 4:
            bridge_out_ch = int(wb.shape[0])
            bridge_k = int(wb.shape[2])

    return {
        "IMG_SIZE": (int(img_size[0]), int(img_size[1])),
        "IN_CHANNELS": int(in_ch),
        "OUT_CHANNELS": int(out_ch),
        "BASE": int(base),

        "VIT_DIM": int(vit_dim),
        "VIT_DEPTH": int(vit_depth),
        "VIT_HEADS": int(vit_heads),
        "VIT_MLP_RATIO": float(vit_mlp_ratio),
        "VIT_DROPOUT": float(vit_dropout),

        # ✅ 這些是你一直缺的 key
        "PATCH_IN_CH": int(patch_in_ch),
        "POS_HW": (int(pos_hw[0]), int(pos_hw[1])),
        "BRIDGE_OUT_CH": int(bridge_out_ch),
        "BRIDGE_K": int(bridge_k),

        "THR_CT": float(thr_ct),
        "THR_MN": float(thr_mn),
        "THR_FT": float(thr_ft),
    }


# ============================================================
# Model blocks  (MATCH train_transunet_ct_mn_ft_weighted.py)
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """
    Same as training:
    LN → MHA → residual
    LN → MLP → residual
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.norm1(x)
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + attn_out
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x


class ViTTransUNet(nn.Module):
    """
    EXACT SAME as train_transunet_ct_mn_ft_weighted.py
    Keys will match:
      d1.*, d2.*, d3.*, d4.*,
      patch_proj.*,
      pos_embed,
      trans_blocks.*,
      bridge.net.*,
      u4.*, c4.net.*,
      u3.*, c3.net.*,
      u2.*, c2.net.*,
      u1.*, c1.net.*,
      out.*
    """
    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 3,
        base: int = 32,
        img_size: Tuple[int, int] = (256, 256),
        vit_dim: int = 512,
        vit_depth: int = 6,
        vit_heads: int = 8,
        vit_mlp_ratio: float = 4.0,
        vit_dropout: float = 0.1,
    ):
        super().__init__()
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.base = int(base)

        assert vit_dim % vit_heads == 0, f"vit_dim({vit_dim}) must be divisible by vit_heads({vit_heads})."

        # CNN Encoder
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.p4 = nn.MaxPool2d(2)  # H/16

        # Patch embedding
        self.patch_proj = nn.Conv2d(base * 8, vit_dim, kernel_size=1, bias=True)

        h16 = self.img_size[0] // 16
        w16 = self.img_size[1] // 16
        self.token_len = int(h16 * w16)
        self.vit_dim = int(vit_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.token_len, vit_dim))
        self.drop = nn.Dropout(vit_dropout)

        self.trans_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=vit_dim,
                num_heads=vit_heads,
                mlp_ratio=vit_mlp_ratio,
                dropout=vit_dropout
            )
            for _ in range(vit_depth)
        ])

        # Bridge (DoubleConv)
        self.bridge = DoubleConv(vit_dim, vit_dim)

        # UNet Decoder
        self.u4 = nn.ConvTranspose2d(vit_dim, base * 8, 2, stride=2)
        self.c4 = DoubleConv(base * 16, base * 8)

        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv(base * 8, base * 4)

        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 4, base * 2)

        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, kernel_size=1, bias=True)

    def _to_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        return feat.flatten(2).transpose(1, 2)  # (N, H*W, D)

    def _to_feat(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        d4 = self.d4(self.p3(d3))

        f = self.p4(d4)          # (N, base*8, H/16, W/16)
        f = self.patch_proj(f)   # (N, vit_dim, H/16, W/16)

        n, d, h, w = f.shape
        tokens = self._to_tokens(f)  # (N, L, D)

        # pos embedding (safe fallback interpolate)
        if tokens.shape[1] == self.pos_embed.shape[1]:
            tokens = tokens + self.pos_embed
        else:
            # ✅ 修正：不要假設 pos_embed 一定是 square
            pe = self.pos_embed  # (1, L0, D)
            L0 = int(pe.shape[1])
            ph0, pw0 = _factorize_token_len(L0)  # (H0, W0)

            pe2 = pe.transpose(1, 2).reshape(1, d, ph0, pw0)
            pe2 = F.interpolate(pe2, size=(h, w), mode="bilinear", align_corners=False)
            pe2 = pe2.flatten(2).transpose(1, 2)
            tokens = tokens + pe2

        tokens = self.drop(tokens)

        for blk in self.trans_blocks:
            tokens = blk(tokens)

        feat = self._to_feat(tokens, h, w)
        feat = self.bridge(feat)

        x = self.u4(feat)
        x = torch.cat([x, d4], dim=1)
        x = self.c4(x)

        x = self.u3(x)
        x = torch.cat([x, d3], dim=1)
        x = self.c3(x)

        x = self.u2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.c2(x)

        x = self.u1(x)
        x = torch.cat([x, d1], dim=1)
        x = self.c1(x)

        return self.out(x)


# ============================================================
# Global models
# ============================================================
TRANS_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANS_MODEL = None
TRANS_MODEL_KIND = None

_TRANS_IMG_SIZE = TRANSUNET_IMG_SIZE
_TRANS_IN_CH = TRANSUNET_IN_CHANNELS
_TRANS_OUT_CH = TRANSUNET_NUM_CLASSES
_TRANS_BASE = TRANSUNET_BASE

_TRANS_VIT_DIM = 512
_TRANS_VIT_DEPTH = 2
_TRANS_VIT_HEADS = 8
_TRANS_VIT_MLP = 4.0
_TRANS_VIT_DROP = 0.1
_TRANS_POS_HW = (TRANSUNET_IMG_SIZE[0] // 16, TRANSUNET_IMG_SIZE[1] // 16)

_TRANS_BRIDGE_OUT = 512
_TRANS_BRIDGE_K = 3
_TRANS_PATCH_IN = TRANSUNET_BASE * 8

_THR_CT = TRANSUNET_THR_CT
_THR_MN = TRANSUNET_THR_MN
_THR_FT = TRANSUNET_THR_FT

MN_MODEL = None
_MN_IMG_SIZE = (256, 256)
_MN_IN_CH = 2
_MN_OUT_CH = 1
_MN_BASE = TRANSUNET_BASE
_MN_THR = 0.5


def load_transunet_once():
    """
    ✅ 自動判斷 ViTTransUNet ckpt 並自動推回必要參數
    """
    global TRANS_MODEL, TRANS_MODEL_KIND
    global _TRANS_IMG_SIZE, _TRANS_IN_CH, _TRANS_OUT_CH, _TRANS_BASE, _TRANS_POS_HW
    global _TRANS_VIT_DIM, _TRANS_VIT_DEPTH, _TRANS_VIT_HEADS, _TRANS_VIT_MLP, _TRANS_VIT_DROP
    global _TRANS_BRIDGE_OUT, _TRANS_BRIDGE_K, _TRANS_PATCH_IN
    global _THR_CT, _THR_MN, _THR_FT

    if TRANS_MODEL is not None:
        return

    if not os.path.isfile(TRANSUNET_CKPT_PATH):
        raise FileNotFoundError(f"TransUNet ckpt not found: {TRANSUNET_CKPT_PATH}")

    ck = torch.load(TRANSUNET_CKPT_PATH, map_location=TRANS_DEVICE)
    sd, cfg = _extract_state_dict_and_cfg(ck)

    if not _is_vit_transunet_ckpt(sd):
        raise RuntimeError("This viewer version expects ViTTransUNet ckpt, but the ckpt looks different.")

    TRANS_MODEL_KIND = "vit"
    auto = _infer_vittransunet_cfg_from_ckpt(
        sd=sd,
        cfg=cfg,
        default_img_size=_TRANS_IMG_SIZE,
        default_heads=_TRANS_VIT_HEADS,
        default_mlp_ratio=_TRANS_VIT_MLP,
        default_dropout=_TRANS_VIT_DROP,
    )

    # ✅ 一定有這些 key（已在 infer 函式補齊）
    _TRANS_IMG_SIZE = tuple(auto["IMG_SIZE"])
    _TRANS_IN_CH = int(auto["IN_CHANNELS"])
    _TRANS_OUT_CH = int(auto["OUT_CHANNELS"])
    _TRANS_BASE = int(auto["BASE"])
    _TRANS_PATCH_IN = int(auto["PATCH_IN_CH"])

    _TRANS_VIT_DIM = int(auto["VIT_DIM"])
    _TRANS_VIT_DEPTH = int(auto["VIT_DEPTH"])
    _TRANS_VIT_HEADS = int(auto["VIT_HEADS"])
    _TRANS_VIT_MLP = float(auto["VIT_MLP_RATIO"])
    _TRANS_VIT_DROP = float(auto["VIT_DROPOUT"])
    _TRANS_POS_HW = tuple(auto["POS_HW"])

    _TRANS_BRIDGE_OUT = int(auto["BRIDGE_OUT_CH"])
    _TRANS_BRIDGE_K = int(auto["BRIDGE_K"])

    _THR_CT = float(auto["THR_CT"])
    _THR_MN = float(auto["THR_MN"])
    _THR_FT = float(auto["THR_FT"])

    model = ViTTransUNet(
        in_ch=_TRANS_IN_CH,
        out_ch=_TRANS_OUT_CH,
        base=_TRANS_BASE,
        img_size=_TRANS_IMG_SIZE,
        vit_dim=_TRANS_VIT_DIM,
        vit_depth=_TRANS_VIT_DEPTH,
        vit_heads=_TRANS_VIT_HEADS,
        vit_mlp_ratio=_TRANS_VIT_MLP,
        vit_dropout=_TRANS_VIT_DROP,
    ).to(TRANS_DEVICE)
    model.eval()

    model.load_state_dict(sd, strict=True)
    TRANS_MODEL = model

    print(f"[ViTTransUNet] loaded: {TRANSUNET_CKPT_PATH}")
    print(f"[ViTTransUNet] device: {TRANS_DEVICE}")
    print(f"[ViTTransUNet] img_size={_TRANS_IMG_SIZE} base={_TRANS_BASE} in_ch={_TRANS_IN_CH} out_ch={_TRANS_OUT_CH}")
    print(f"[ViTTransUNet] patch_in={_TRANS_PATCH_IN} vit_dim={_TRANS_VIT_DIM} depth={_TRANS_VIT_DEPTH} heads={_TRANS_VIT_HEADS}")
    print(f"[ViTTransUNet] pos_hw={_TRANS_POS_HW}")
    print(f"[ViTTransUNet] bridge_out={_TRANS_BRIDGE_OUT} bridge_k={_TRANS_BRIDGE_K}")
    print(f"[ViTTransUNet] thr(CT,MN,FT)=({_THR_CT},{_THR_MN},{_THR_FT})")


@torch.no_grad()
def transunet_predict_multilabel_masks(t1_u8: np.ndarray, t2_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    load_transunet_once()

    Hm, Wm = _TRANS_IMG_SIZE
    t1_rs = cv2.resize(t1_u8, (Wm, Hm), interpolation=cv2.INTER_LINEAR)
    t2_rs = cv2.resize(t2_u8, (Wm, Hm), interpolation=cv2.INTER_LINEAR)

    t1_n = norm01_robust(t1_rs)
    t2_n = norm01_robust(t2_rs)

    x = np.stack([t1_n, t2_n], axis=0)  # (2,H,W)
    x = torch.from_numpy(x).unsqueeze(0).to(TRANS_DEVICE)  # (1,2,H,W)

    use_amp = bool(TRANSUNET_MIXED_PRECISION and TRANS_DEVICE.type == "cuda")
    with get_autocast(device_type=TRANS_DEVICE.type, enabled=use_amp):
        logits = TRANS_MODEL(x)       # (1,3,H,W)
        probs = torch.sigmoid(logits)

        pred = torch.zeros_like(probs)
        pred[:, 0] = (probs[:, 0] >= _THR_CT).float()
        pred[:, 1] = (probs[:, 1] >= _THR_MN).float()
        pred[:, 2] = (probs[:, 2] >= _THR_FT).float()

    pred_small = pred.squeeze(0).detach().cpu().numpy().astype(np.uint8)  # (3,H,W)
    ct_s = pred_small[0]
    mn_s = pred_small[1]
    ft_s = pred_small[2]

    ct = cv2.resize(ct_s, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    mn = cv2.resize(mn_s, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    ft = cv2.resize(ft_s, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    return ct, mn, ft


def predict_masks_transunet(t1_img_u8, t2_img_u8):
    """
    回傳順序（維持你原本 Viewer 用法）：
      ct_pred, ft_pred, mn_pred
    """
    if t1_img_u8 is None or t2_img_u8 is None:
        z = np.zeros((SIZE, SIZE), dtype=np.uint8)
        return z, z, z

    ct, mn, ft = transunet_predict_multilabel_masks(t1_img_u8, t2_img_u8)
    return ct, ft, mn


def dice_coef(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    s = gt.sum() + pred.sum()
    if s == 0:
        return 1.0
    return 2.0 * inter / s


def safe_sort_numeric_first(files):
    def key(p):
        s = os.path.splitext(os.path.basename(p))[0]
        if s.isdigit():
            return (0, int(s))
        return (1, s)
    return sorted(files, key=key)


# ============================================================
# UI
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Segmentation Viewer (ViTTransUNet bridge fixed)")
        self.resize(1300, 700)

        self.t1_images = []
        self.t2_images = []

        self.gt_masks = {"CT": [], "FT": [], "MN": []}
        self.pred_masks = {"CT": [], "FT": [], "MN": []}
        self.dice_scores = {"CT": [], "FT": [], "MN": []}

        self.idx = 0
        self.show_pred = False

        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ========== 左邊：T1 / T2 ==========
        left_box = QGroupBox()
        left_layout = QVBoxLayout(left_box)

        left_layout.addWidget(QLabel("T1"))
        self.lbl_t1 = make_image_box()
        left_layout.addWidget(self.lbl_t1)

        left_layout.addWidget(QLabel("T2"))
        self.lbl_t2 = make_image_box()
        left_layout.addWidget(self.lbl_t2)
        left_layout.addStretch()

        btn_load_t1 = QPushButton("Load T1 folder")
        btn_prev = QPushButton("←")
        btn_next = QPushButton("→")

        btn_load_t1.clicked.connect(self.load_t1_folder)
        btn_prev.clicked.connect(self.prev_img)
        btn_next.clicked.connect(self.next_img)

        h1 = QHBoxLayout()
        h1.addWidget(btn_load_t1)
        h1.addStretch()
        h1.addWidget(btn_prev)
        h1.addWidget(btn_next)
        left_layout.addLayout(h1)

        btn_load_t2 = QPushButton("Load T2 folder")
        btn_load_t2.clicked.connect(self.load_t2_folder)

        self.spin_idx = QSpinBox()
        self.spin_idx.setMinimum(0)
        self.spin_idx.setMaximum(0)
        self.spin_idx.setValue(0)
        self.spin_idx.valueChanged.connect(self.go_index)

        self.lbl_filename = QLabel("")

        h2 = QHBoxLayout()
        h2.addWidget(btn_load_t2)
        h2.addStretch()
        h2.addWidget(self.spin_idx)
        h2.addWidget(self.lbl_filename)
        left_layout.addLayout(h2)

        # ========== 右邊：Tabs ==========
        right_box = QGroupBox()
        right_layout = QVBoxLayout(right_box)

        self.tabs = QTabWidget()
        self.tab_t1 = QWidget()
        self.tab_t2 = QWidget()
        self.tabs.addTab(self.tab_t1, "T1")
        self.tabs.addTab(self.tab_t2, "T2")
        right_layout.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self.on_tab_changed)

        self.result_boxes = {"T1": {}, "T2": {}}
        self.dice_labels = {"T1": {}, "T2": {}}
        self.build_tab("T1", self.tab_t1)
        self.build_tab("T2", self.tab_t2)

        bottom_layout = QHBoxLayout()

        btn_ct_mask = QPushButton("Load CT Mask folder")
        btn_ft_mask = QPushButton("Load FT Mask folder")
        btn_mn_mask = QPushButton("Load MN Mask folder")
        btn_predict = QPushButton("Predict")

        btn_ct_mask.clicked.connect(lambda: self.load_mask_folder("CT"))
        btn_ft_mask.clicked.connect(lambda: self.load_mask_folder("FT"))
        btn_mn_mask.clicked.connect(lambda: self.load_mask_folder("MN"))
        btn_predict.clicked.connect(self.predict_all)

        bottom_layout.addWidget(btn_ct_mask)
        bottom_layout.addWidget(btn_ft_mask)
        bottom_layout.addWidget(btn_mn_mask)
        bottom_layout.addSpacing(40)
        bottom_layout.addWidget(btn_predict)
        bottom_layout.addStretch()

        right_layout.addLayout(bottom_layout)

        main_layout.addWidget(left_box, 1)
        main_layout.addWidget(right_box, 3)

    def build_tab(self, tab_name: str, container: QWidget):
        layout = QVBoxLayout(container)
        grid = QGridLayout()
        grid.setHorizontalSpacing(80)

        titles = ["CT", "FT", "MN"]
        for col, key in enumerate(titles):
            lbl_title = QLabel(key)
            box = make_image_box()
            lbl_dice = QLabel("Dice coefficient:")

            self.result_boxes[tab_name][key] = box
            self.dice_labels[tab_name][key] = lbl_dice

            grid.addWidget(lbl_title, 0, col, alignment=Qt.AlignCenter)
            grid.addWidget(box, 1, col, alignment=Qt.AlignCenter)
            grid.addWidget(lbl_dice, 2, col, alignment=Qt.AlignCenter)

        layout.addLayout(grid)
        layout.addStretch()

    def update_spin_range(self):
        lengths = [len(self.t1_images), len(self.t2_images)]
        for lst in self.gt_masks.values():
            lengths.append(len(lst))
        max_len = max(lengths) if lengths else 0
        self.spin_idx.setMaximum(max(0, max_len - 1))

    def load_folder_images(self, folder):
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]
        return safe_sort_numeric_first(files)

    def load_t1_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select T1 Folder")
        if folder:
            self.t1_images = self.load_folder_images(folder)
            self.idx = 0
            self.spin_idx.setValue(0)
            self.update_spin_range()
            self.update_base_images()

    def load_t2_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select T2 Folder")
        if folder:
            self.t2_images = self.load_folder_images(folder)
            self.idx = 0
            self.spin_idx.setValue(0)
            self.update_spin_range()
            self.update_base_images()

    def load_mask_folder(self, kind: str):
        folder = QFileDialog.getExistingDirectory(self, f"Select {kind} Mask Folder")
        if not folder:
            return

        files = self.load_folder_images(folder)
        size = (SIZE, SIZE)
        masks = []

        for path in files:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
            mask_bin = (img > 127).astype(np.uint8)

            if kind == "CT" and CT_KEEP_ONE_COMPONENT:
                mask_bin = keep_one_component_largest(mask_bin, min_area=CT_MIN_AREA)

            if kind == "MN" and MN_KEEP_ONE_COMPONENT:
                mask_bin = keep_one_component_weighted(mask_bin, MN_W_AREA, MN_W_TOP, MN_W_RIGHT, MN_MIN_AREA)

            masks.append(mask_bin)

        self.gt_masks[kind] = masks
        self.pred_masks[kind] = []
        self.dice_scores[kind] = []

        self.show_pred = False
        self.update_spin_range()
        self.update_base_images()

    def prev_img(self):
        if self.idx > 0:
            self.idx -= 1
            self.spin_idx.blockSignals(True)
            self.spin_idx.setValue(self.idx)
            self.spin_idx.blockSignals(False)
            self.update_base_images()

    def next_img(self):
        if self.idx < self.spin_idx.maximum():
            self.idx += 1
            self.spin_idx.blockSignals(True)
            self.spin_idx.setValue(self.idx)
            self.spin_idx.blockSignals(False)
            self.update_base_images()

    def go_index(self, value):
        self.idx = value
        self.update_base_images()

    def update_filename_label(self):
        tab_name = "T1" if self.tabs.currentIndex() == 0 else "T2"
        base_list = self.t1_images if tab_name == "T1" else self.t2_images

        if base_list and self.idx < len(base_list):
            path = base_list[self.idx]
            name = os.path.basename(path)
            self.lbl_filename.setText(name)
        else:
            self.lbl_filename.setText("")

    def update_base_images(self):
        size = SIZE

        if self.t1_images and self.idx < len(self.t1_images):
            pix = QPixmap(self.t1_images[self.idx]).scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_t1.setPixmap(pix)
        else:
            self.lbl_t1.clear()

        if self.t2_images and self.idx < len(self.t2_images):
            pix = QPixmap(self.t2_images[self.idx]).scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_t2.setPixmap(pix)
        else:
            self.lbl_t2.clear()

        self.update_results()
        self.update_filename_label()

    def on_tab_changed(self, index):
        self.update_results()
        self.update_filename_label()

    def update_results(self):
        tab_name = "T1" if self.tabs.currentIndex() == 0 else "T2"
        base_list = self.t1_images if tab_name == "T1" else self.t2_images

        if not base_list or self.idx >= len(base_list):
            for kind in ["CT", "FT", "MN"]:
                self.result_boxes[tab_name][kind].clear()
                self.dice_labels[tab_name][kind].setText("Dice coefficient:")
            return

        base_path = base_list[self.idx]
        base_img = cv2.imread(base_path)
        if base_img is None:
            for kind in ["CT", "FT", "MN"]:
                self.result_boxes[tab_name][kind].clear()
                self.dice_labels[tab_name][kind].setText("Dice coefficient:")
            return

        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        base_img = cv2.resize(base_img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)

        for kind in ["CT", "FT", "MN"]:
            box = self.result_boxes[tab_name][kind]
            dice_label = self.dice_labels[tab_name][kind]

            mask_to_use = None
            dice_text = "Dice coefficient:"

            if self.show_pred and self.pred_masks[kind]:
                if self.idx < len(self.pred_masks[kind]):
                    mask_to_use = self.pred_masks[kind][self.idx]
                    if self.idx < len(self.dice_scores[kind]):
                        dice_text = f"Dice coefficient: {self.dice_scores[kind][self.idx]:.3f}"

            elif self.gt_masks[kind]:
                if self.idx < len(self.gt_masks[kind]):
                    mask_to_use = self.gt_masks[kind][self.idx]
                    dice_text = "Dice coefficient: -"

            if mask_to_use is None:
                box.clear()
                dice_label.setText("Dice coefficient:")
                continue

            if not self.show_pred:
                overlay_np = draw_mask(base_img, mask_to_use)
            else:
                gt = self.gt_masks[kind][self.idx] if self.idx < len(self.gt_masks[kind]) else None
                pred = self.pred_masks[kind][self.idx] if self.idx < len(self.pred_masks[kind]) else None
                if gt is None or pred is None:
                    overlay_np = draw_mask(base_img, mask_to_use)
                    dice_text = "Dice coefficient: -"
                else:
                    overlay_np = draw_predict_mask(base_img, gt, pred, kind=kind)

            overlay_np = np.ascontiguousarray(overlay_np)
            h, w, ch = overlay_np.shape
            bytes_per_line = ch * w
            qimg = QImage(overlay_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)

            box.setPixmap(pix)
            dice_label.setText(dice_text)

    def predict_all(self):
        size = (SIZE, SIZE)

        if not self.t1_images or not self.t2_images:
            QMessageBox.warning(self, "缺資料", "請先載入 T1 folder 與 T2 folder")
            return

        try:
            for kind in ["CT", "FT", "MN"]:
                self.pred_masks[kind] = []
                self.dice_scores[kind] = []

            max_n = min(len(self.t1_images), len(self.t2_images))

            for i in range(max_n):
                t1_path = self.t1_images[i]
                t2_path = self.t2_images[i]

                t1_img = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
                t2_img = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)

                if t1_img is None or t2_img is None:
                    ct_pred = np.zeros(size, dtype=np.uint8)
                    ft_pred = np.zeros(size, dtype=np.uint8)
                    mn_pred = np.zeros(size, dtype=np.uint8)
                else:
                    t1_img = cv2.resize(t1_img, size, interpolation=cv2.INTER_LINEAR)
                    t2_img = cv2.resize(t2_img, size, interpolation=cv2.INTER_LINEAR)

                    ct_pred, ft_model_pred, mn_pred = predict_masks_transunet(t1_img, t2_img)

                    ct_pred = (ct_pred > 0).astype(np.uint8)
                    ft_model_pred = (ft_model_pred > 0).astype(np.uint8)
                    mn_pred = (mn_pred > 0).astype(np.uint8)

                    if CT_KEEP_ONE_COMPONENT:
                        ct_pred = keep_one_component_largest(ct_pred, min_area=CT_MIN_AREA)

                    if FT_FROM_CT_DARK_IN_T2_ENABLE:
                        ft_ct = derive_ft_from_ct_dark_in_t2(t2_img, ct_pred)
                        if np.any(ft_ct):
                            ft_pred = ft_ct
                        else:
                            ft_pred = ft_model_pred if FT_FALLBACK_TO_MODEL_IF_FAIL else np.zeros(size, dtype=np.uint8)
                    else:
                        ft_pred = ft_model_pred

                    if MN_KEEP_ONE_COMPONENT:
                        mn_pred = keep_one_component_weighted(mn_pred, MN_W_AREA, MN_W_TOP, MN_W_RIGHT, MN_MIN_AREA)

                    ct_pred = (ct_pred > 0).astype(np.uint8)
                    ft_pred = (ft_pred > 0).astype(np.uint8)
                    mn_pred = (mn_pred > 0).astype(np.uint8)

                for kind, pred in [("CT", ct_pred), ("FT", ft_pred), ("MN", mn_pred)]:
                    pred = (pred > 0).astype(np.uint8)
                    self.pred_masks[kind].append(pred)

                    if i < len(self.gt_masks[kind]):
                        gt = self.gt_masks[kind][i]
                        d = dice_coef(gt, pred)
                    else:
                        d = 0.0
                    self.dice_scores[kind].append(d)

        except Exception as e:
            QMessageBox.critical(self, "Predict 發生錯誤", str(e))
            return

        self.show_pred = True
        self.update_results()


if __name__ == "__main__":
    try:
        load_transunet_once()
    except Exception as e:
        print("[Model] load failed:", e)

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
