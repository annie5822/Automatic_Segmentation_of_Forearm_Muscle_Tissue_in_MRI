# transunet_viewer_ct_mn_ft.py
import sys
import os
from typing import Tuple, Optional

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
# TransUNet ckpt
# =========================
TRANSUNET_CKPT_PATH = "runs_transunet_ct_mn_ft/exp_20260104_235936/checkpoints/best_0.8445.pt"
MN_TRANSUNET_CKPT_PATH = "runs_transunet_mn/exp_20260106_195827/checkpoints/best_0.7486.pt"

# 推論用：固定使用 training 的 IMG_SIZE / IN_CHANNELS / OUT_CHANNELS
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
CT_MIN_AREA = 200  # 小於這面積的連通元件直接丟掉；想更嚴格就加大（例如 500 / 1000）

FT_FROM_CT_DARK_IN_T2_ENABLE = True
FT_T2_MIN_PIXELS_IN_CT = 200
FT_T2_ROI_P_LOW = 1.0
FT_T2_ROI_P_HIGH = 99.0

# 對比強化方式：Sigmoid
FT_T2_USE_SIGMOID_CONTRAST = True
FT_T2_SIGMOID_K = 8.0  # 越大對比越強

# 若不用 Sigmoid：改用線性對比
FT_T2_LINEAR_ALPHA = 1.8  # 1.2~2.5
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
FT_KEEP_LARGEST = False  # FT 可分成多塊，不要只留最大

# 若 CT 太小/估不到門檻 → 要不要回退用 model 的 FT
FT_FALLBACK_TO_MODEL_IF_FAIL = True

# ============================================================
# CT pred 邊框太鋸齒 → 圓滑化 + anti-alias
# ============================================================
CT_CONTOUR_SMOOTH_ENABLE = True
CT_CONTOUR_MEDIAN_K = 5          # 3/5/7... 越大越平滑
CT_CONTOUR_CLOSE_K = 5           # close 讓邊緣比較連續
CT_CONTOUR_CLOSE_IT = 1
CT_CONTOUR_EPS_RATIO = 0.003     # approxPolyDP 的 epsilon 比例：越大越圓滑
CT_CONTOUR_THICKNESS = 1         # 紅線粗細

# =========================
# AMP compatibility helper
# =========================
try:
    from torch import amp as _amp  # torch>=2.0
    _HAS_TORCH_AMP = True
except Exception:
    _HAS_TORCH_AMP = False
    _amp = None


def get_autocast(device_type: str, enabled: bool):
    if _HAS_TORCH_AMP:
        return _amp.autocast(device_type=device_type, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled=enabled)


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
    """
    只在 ROI 內計算 percentile，回傳：
      norm_img01 (float32, same shape),
      lo, hi
    若 ROI 太小/空，回 None
    """
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
    """
    x01 in [0,1]
    y = sigmoid(k*(x-0.5))
    k 越大對比越強
    """
    k = float(k)
    z = k * (x01.astype(np.float32) - 0.5)
    y = 1.0 / (1.0 + np.exp(-z))
    return y.astype(np.float32)


def _contrast_linear(x01: np.ndarray, alpha: float, beta: float = 0.0) -> np.ndarray:
    """
    y = clip((x-0.5)*alpha + 0.5 + beta, 0, 1)
    alpha>1 對比更強
    """
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
    """
    1) 取 CT ROI
    2) 在 ROI 內把 T2 做對比強化
    3) 用強化後 ROI 的 percentile 取暗部
    """
    ct = (ct_mask_u8 > 0).astype(np.uint8)
    if not np.any(ct):
        return np.zeros_like(ct)

    # 可選：CT 往內縮，避免邊界雜訊被選到 FT
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

    # 清理
    ft = _morph_open_close(ft, FT_OPEN_K, FT_OPEN_IT, FT_CLOSE_K, FT_CLOSE_IT)

    if FT_KEEP_LARGEST:
        ft = _keep_largest_cc(ft)

    return (ft > 0).astype(np.uint8)



def keep_one_component_weighted(mask_u8: np.ndarray,
                                w_area: float = 0.5,
                                w_top: float = 0.25,
                                w_right: float = 0.25,
                                min_area: int = 30) -> np.ndarray:
    """
    在連通元件中只保留一塊：
      score = w_area * area_norm + w_top * topness + w_right * rightness
      - area_norm：area / max_area
      - topness：  1 - cy/(H-1)     (越靠上越好)
      - rightness：cx/(W-1)         (越靠右越好)
    """
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
# TransUNet (CNN encoder + Transformer bottleneck + UNet decoder)
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
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
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


class TransUNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 3,
        base: int = 32,
        img_size: Tuple[int, int] = (256, 256),
        trans_layers: int = 2,
        trans_heads: int = 4,
        trans_mlp_ratio: float = 4.0,
        trans_dropout: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size

        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.p4 = nn.MaxPool2d(2)

        self.mid = DoubleConv(base * 8, base * 16)

        self.embed_dim = base * 16
        h16 = img_size[0] // 16
        w16 = img_size[1] // 16
        self.token_len = h16 * w16

        self.pos_embed = nn.Parameter(torch.zeros(1, self.token_len, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.trans_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=self.embed_dim,
                num_heads=trans_heads,
                mlp_ratio=trans_mlp_ratio,
                dropout=trans_dropout
            )
            for _ in range(trans_layers)
        ])

        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.c4 = DoubleConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def _to_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        return feat.flatten(2).transpose(1, 2)

    def _to_feat(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        d4 = self.d4(self.p3(d3))
        mid = self.mid(self.p4(d4))

        n, d, h, w = mid.shape
        tokens = self._to_tokens(mid)

        if tokens.shape[1] == self.token_len:
            tokens = tokens + self.pos_embed
        else:
            pe = self.pos_embed
            L0 = pe.shape[1]
            s0 = int(np.sqrt(L0))
            pe2 = pe.transpose(1, 2).reshape(1, d, s0, s0)
            pe2 = F.interpolate(pe2, size=(h, w), mode="bilinear", align_corners=False)
            pe2 = pe2.flatten(2).transpose(1, 2)
            tokens = tokens + pe2

        for blk in self.trans_blocks:
            tokens = blk(tokens)

        mid_t = self._to_feat(tokens, h, w)

        x = self.u4(mid_t)
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


TRANS_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANS_MODEL = None


_TRANS_IMG_SIZE = TRANSUNET_IMG_SIZE
_TRANS_IN_CH = TRANSUNET_IN_CHANNELS
_TRANS_OUT_CH = TRANSUNET_NUM_CLASSES
_TRANS_LAYERS = 2
_TRANS_HEADS = 4
_TRANS_MLP = 4.0
_TRANS_DROP = 0.1
_TRANS_BASE = TRANSUNET_BASE


_THR_CT = TRANSUNET_THR_CT
_THR_MN = TRANSUNET_THR_MN
_THR_FT = TRANSUNET_THR_FT


MN_MODEL = None
_MN_IMG_SIZE = (256, 256)
_MN_IN_CH = 2
_MN_OUT_CH = 1
_MN_LAYERS = 2
_MN_HEADS = 4
_MN_MLP = 4.0
_MN_DROP = 0.1
_MN_BASE = TRANSUNET_BASE
_MN_THR = 0.5


def load_transunet_once():
    global TRANS_MODEL
    global _TRANS_IMG_SIZE, _TRANS_IN_CH, _TRANS_OUT_CH, _TRANS_LAYERS, _TRANS_HEADS, _TRANS_MLP, _TRANS_DROP, _TRANS_BASE
    global _THR_CT, _THR_MN, _THR_FT

    if TRANS_MODEL is not None:
        return

    if not os.path.isfile(TRANSUNET_CKPT_PATH):
        raise FileNotFoundError(f"TransUNet ckpt not found: {TRANSUNET_CKPT_PATH}")

    ck = torch.load(TRANSUNET_CKPT_PATH, map_location=TRANS_DEVICE)

    cfg = None
    if isinstance(ck, dict):
        cfg = ck.get("cfg", None)

    if isinstance(cfg, dict):
        img_size = cfg.get("IMG_SIZE", _TRANS_IMG_SIZE)
        if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            _TRANS_IMG_SIZE = (int(img_size[0]), int(img_size[1]))

        _TRANS_IN_CH = int(cfg.get("IN_CHANNELS", _TRANS_IN_CH))
        _TRANS_OUT_CH = int(cfg.get("OUT_CHANNELS", _TRANS_OUT_CH))

        _TRANS_LAYERS = int(cfg.get("TRANS_NUM_LAYERS", _TRANS_LAYERS))
        _TRANS_HEADS = int(cfg.get("TRANS_NUM_HEADS", _TRANS_HEADS))
        _TRANS_MLP = float(cfg.get("TRANS_MLP_RATIO", _TRANS_MLP))
        _TRANS_DROP = float(cfg.get("TRANS_DROPOUT", _TRANS_DROP))

        if "THR_CT" in cfg:
            _THR_CT = float(cfg["THR_CT"])
        if "THR_MN" in cfg:
            _THR_MN = float(cfg["THR_MN"])
        if "THR_FT" in cfg:
            _THR_FT = float(cfg["THR_FT"])

        _TRANS_BASE = TRANSUNET_BASE

    model = TransUNet(
        in_ch=_TRANS_IN_CH,
        out_ch=_TRANS_OUT_CH,
        base=_TRANS_BASE,
        img_size=_TRANS_IMG_SIZE,
        trans_layers=_TRANS_LAYERS,
        trans_heads=_TRANS_HEADS,
        trans_mlp_ratio=_TRANS_MLP,
        trans_dropout=_TRANS_DROP,
    ).to(TRANS_DEVICE)
    model.eval()

    sd = ck["model_state"] if (isinstance(ck, dict) and "model_state" in ck) else ck
    model.load_state_dict(sd, strict=True)

    TRANS_MODEL = model
    print(f"[TransUNet] loaded: {TRANSUNET_CKPT_PATH}")
    print(f"[TransUNet] device: {TRANS_DEVICE}")
    print(f"[TransUNet] img_size={_TRANS_IMG_SIZE} in_ch={_TRANS_IN_CH} out_ch={_TRANS_OUT_CH}")
    print(f"[TransUNet] trans: layers={_TRANS_LAYERS} heads={_TRANS_HEADS} mlp={_TRANS_MLP} drop={_TRANS_DROP}")
    print(f"[TransUNet] thr(CT,MN,FT)=({_THR_CT},{_THR_MN},{_THR_FT})")


def load_mn_transunet_once():
    global MN_MODEL
    global _MN_IMG_SIZE, _MN_IN_CH, _MN_OUT_CH, _MN_LAYERS, _MN_HEADS, _MN_MLP, _MN_DROP, _MN_BASE, _MN_THR

    if MN_MODEL is not None:
        return

    if not os.path.isfile(MN_TRANSUNET_CKPT_PATH):
        # 只影響 MN；不動其它流程
        print(f"[MN-TransUNet] ckpt not found, fallback to multi-label MN: {MN_TRANSUNET_CKPT_PATH}")
        MN_MODEL = None
        return

    ck = torch.load(MN_TRANSUNET_CKPT_PATH, map_location=TRANS_DEVICE)

    cfg = None
    if isinstance(ck, dict):
        cfg = ck.get("cfg", None)

    if isinstance(cfg, dict):
        img_size = cfg.get("IMG_SIZE", _MN_IMG_SIZE)
        if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            _MN_IMG_SIZE = (int(img_size[0]), int(img_size[1]))

        _MN_IN_CH = int(cfg.get("IN_CHANNELS", _MN_IN_CH))
        _MN_OUT_CH = int(cfg.get("OUT_CHANNELS", _MN_OUT_CH))

        _MN_LAYERS = int(cfg.get("TRANS_NUM_LAYERS", _MN_LAYERS))
        _MN_HEADS = int(cfg.get("TRANS_NUM_HEADS", _MN_HEADS))
        _MN_MLP = float(cfg.get("TRANS_MLP_RATIO", _MN_MLP))
        _MN_DROP = float(cfg.get("TRANS_DROPOUT", _MN_DROP))

        # MN-only training script: cfg.THR
        if "THR" in cfg:
            _MN_THR = float(cfg["THR"])

        _MN_BASE = TRANSUNET_BASE

    model = TransUNet(
        in_ch=_MN_IN_CH,
        out_ch=_MN_OUT_CH,
        base=_MN_BASE,
        img_size=_MN_IMG_SIZE,
        trans_layers=_MN_LAYERS,
        trans_heads=_MN_HEADS,
        trans_mlp_ratio=_MN_MLP,
        trans_dropout=_MN_DROP,
    ).to(TRANS_DEVICE)
    model.eval()

    sd = ck["model_state"] if (isinstance(ck, dict) and "model_state" in ck) else ck
    model.load_state_dict(sd, strict=True)

    MN_MODEL = model
    print(f"[MN-TransUNet] loaded: {MN_TRANSUNET_CKPT_PATH}")
    print(f"[MN-TransUNet] device: {TRANS_DEVICE}")
    print(f"[MN-TransUNet] img_size={_MN_IMG_SIZE} in_ch={_MN_IN_CH} out_ch={_MN_OUT_CH}")
    print(f"[MN-TransUNet] trans: layers={_MN_LAYERS} heads={_MN_HEADS} mlp={_MN_MLP} drop={_MN_DROP}")
    print(f"[MN-TransUNet] thr(MN)={_MN_THR}")


@torch.no_grad()
def mn_transunet_predict_mask(t1_u8: np.ndarray, t2_u8: np.ndarray) -> Optional[np.ndarray]:
    """
    回傳 MN mask（shape: (H,W) uint8 {0,1}），大小是 _MN_IMG_SIZE
    若 MN ckpt 不存在 → 回 None（caller fallback）
    """
    load_mn_transunet_once()
    if MN_MODEL is None:
        return None

    Hm, Wm = _MN_IMG_SIZE
    t1_rs = cv2.resize(t1_u8, (Wm, Hm), interpolation=cv2.INTER_LINEAR)
    t2_rs = cv2.resize(t2_u8, (Wm, Hm), interpolation=cv2.INTER_LINEAR)

    t1_n = norm01_robust(t1_rs)
    t2_n = norm01_robust(t2_rs)

    x = np.stack([t1_n, t2_n], axis=0)  # (2,H,W)
    x = torch.from_numpy(x).unsqueeze(0).to(TRANS_DEVICE)  # (1,2,H,W)

    use_amp = bool(TRANSUNET_MIXED_PRECISION and TRANS_DEVICE.type == "cuda")
    with get_autocast(device_type=TRANS_DEVICE.type, enabled=use_amp):
        logits = MN_MODEL(x)          # (1,1,H,W)
        probs = torch.sigmoid(logits) # (1,1,H,W)
        pred = (probs >= float(_MN_THR)).float()

    mn = pred.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)  # (H,W)
    return mn


@torch.no_grad()
def transunet_predict_multilabel_masks(
    t1_u8: np.ndarray,
    t2_u8: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    input: t1_u8,t2_u8 shape (SIZE,SIZE) uint8
    output: ct, mn, ft masks (each shape (SIZE,SIZE) uint8 in {0,1})

    channel define:
      0 = CT, 1 = MN, 2 = FT
    """
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
        probs = torch.sigmoid(logits) # (1,3,H,W)

        pred = torch.zeros_like(probs)
        pred[:, 0] = (probs[:, 0] >= _THR_CT).float()
        pred[:, 1] = (probs[:, 1] >= _THR_MN).float()
        pred[:, 2] = (probs[:, 2] >= _THR_FT).float()

    pred_small = pred.squeeze(0).detach().cpu().numpy().astype(np.uint8)  # (3,H,W)

    ct_s = pred_small[0]
    mn_s = pred_small[1]
    ft_s = pred_small[2]

    mn_from_mn_ckpt = mn_transunet_predict_mask(t1_u8, t2_u8)
    if mn_from_mn_ckpt is not None:
        mn_s = cv2.resize(mn_from_mn_ckpt, (Wm, Hm), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Segmentation Viewer (TransUNet + CT keep one + FT from T2(CT ROI) dark)")
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

        # ========== 右邊：Tabs + CT/FT/MN ==========
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

            # 載入 GT 時也刪除小點，只留一塊
            if kind == "CT" and CT_KEEP_ONE_COMPONENT:
                mask_bin = keep_one_component_largest(mask_bin, min_area=CT_MIN_AREA)

            # 載入 GT 時也做清理
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

        has_any_gt = (len(self.gt_masks["CT"]) > 0) or (len(self.gt_masks["FT"]) > 0) or (len(self.gt_masks["MN"]) > 0)
        if not has_any_gt:
            ans = QMessageBox.question(
                self,
                "未載入 GT",
                "你目前沒有載入任何 GT mask（CT/FT/MN）。\n仍要執行 Predict 並只顯示 pred 嗎？",
                QMessageBox.Yes | QMessageBox.No
            )
            if ans != QMessageBox.Yes:
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

                    # binarize
                    ct_pred = (ct_pred > 0).astype(np.uint8)
                    ft_model_pred = (ft_model_pred > 0).astype(np.uint8)
                    mn_pred = (mn_pred > 0).astype(np.uint8)

                    # --- CT：刪小點，只留一塊 ---
                    if CT_KEEP_ONE_COMPONENT:
                        ct_pred = keep_one_component_largest(ct_pred, min_area=CT_MIN_AREA)

                    # --- FT：用「T2 的 CT ROI」強化後取暗部 → 取代 model 的 FT ---
                    if FT_FROM_CT_DARK_IN_T2_ENABLE:
                        ft_ct = derive_ft_from_ct_dark_in_t2(t2_img, ct_pred)
                        if np.any(ft_ct):
                            ft_pred = ft_ct
                        else:
                            ft_pred = ft_model_pred if FT_FALLBACK_TO_MODEL_IF_FAIL else np.zeros(size, dtype=np.uint8)
                    else:
                        ft_pred = ft_model_pred

                    # --- MN：score = 0.5 size + 0.25 top + 0.25 right，只留一塊 ---
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
        load_mn_transunet_once()
    except Exception as e:
        print("[TransUNet] load failed:", e)

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
