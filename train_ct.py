# train_transunet_ct_mn_ft_weighted.py
import os
import glob
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

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


def get_grad_scaler(device_type: str, enabled: bool):
    if _HAS_TORCH_AMP and hasattr(_amp, "GradScaler"):
        return _amp.GradScaler(device_type=device_type, enabled=enabled)

    return torch.cuda.amp.GradScaler(enabled=bool(enabled and device_type == "cuda"))


# =========================
# Config
# =========================
@dataclass
class CFG:
    BASE: str = "/home/phlab/VS2025_Project/carpalTunnel"
    OUT_DIR: str = "./runs_transunet_ct_mn_ft"
    EXP_NAME: str = ""
    SEED: int = 1337

    TRAIN_SUBSETS: Tuple[str, ...] = ("0", "1", "2", "3", "4", "5")
    TEST_SUBSETS: Tuple[str, ...] = ("6", "7", "8")
    VAL_SUBSETS: Tuple[str, ...] = ("9",)  # if not exist -> auto split from train

    IMG_SIZE: Tuple[int, int] = (256, 256)  # (H, W)
    IN_CHANNELS: int = 2                    # T1 + T2
    OUT_CHANNELS: int = 3                   # CT, MN, FT (multi-label)

    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 4
    EPOCHS: int = 120
    LR: float = 3e-4
    WEIGHT_DECAY: float = 1e-4
    MIXED_PRECISION: bool = True

    # Dice config
    DICE_SMOOTH: float = 1.0
    DICE_EPS: float = 1e-6

    # thresholds (allow per-class)
    THR_CT: float = 0.5
    THR_MN: float = 0.5
    THR_FT: float = 0.3

    # Visualize
    SAVE_TEST_VIS: bool = True

    # =========================
    # ViT-TransUNet config (Hybrid)
    # =========================
    # patch tokens 來自 CNN 最底層 feature map (H/16, W/16)
    VIT_EMBED_DIM: int = 512          # token dim (D). 建議可調 256/512/768
    VIT_DEPTH: int = 6                # Transformer layers (L)
    VIT_HEADS: int = 8                # Multi-head attention heads
    VIT_MLP_RATIO: float = 4.0        # FFN expansion
    VIT_DROPOUT: float = 0.1

    # =========================
    # Loss weighting (CT/FT heavier)
    # order: (CT, MN, FT)
    # =========================
    CHANNEL_WEIGHTS: Tuple[float, float, float] = (3.0, 1.0, 3.0)

    USE_BCE: bool = True
    DICE_LOSS_WEIGHT: float = 1.0
    BCE_LOSS_WEIGHT: float = 1.0
    BCE_POS_WEIGHTS: Tuple[float, float, float] = (4.0, 1.0, 4.0)

    # =========================
    # Sampler (oversample CT/FT cases)
    # =========================
    USE_SAMPLER: bool = True
    SAMPLE_MUL_CT: float = 2.0
    SAMPLE_MUL_FT: float = 2.0
    SAMPLE_MUL_MN: float = 1.0  # usually leave 1


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Utils: IO / preprocess
# =========================
IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def list_images(folder: str) -> List[str]:
    out = []
    for ext in IMG_EXTS:
        out.extend(glob.glob(os.path.join(folder, ext)))

    def sort_key(p: str):
        s = stem(p)
        if s.isdigit():
            return (0, int(s))
        return (1, s)

    out = sorted(out, key=sort_key)
    return out


def imread_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def resize_hw(img: np.ndarray, hw: Tuple[int, int], is_mask: bool) -> np.ndarray:
    h, w = hw
    if is_mask:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def norm01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max() + 1e-6)
    img = np.clip((img - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return img


def binarize_mask(m: np.ndarray) -> np.ndarray:
    return (m > 0).astype(np.uint8)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def weighted_mean_np(x: np.ndarray, w: Tuple[float, ...]) -> float:
    ww = np.array(w, dtype=np.float64)
    xx = np.array(x, dtype=np.float64)
    s = float(ww.sum())
    if s <= 0:
        return float(xx.mean())
    return float((xx * ww).sum() / (s + 1e-12))


# =========================
# Dataset builder
# =========================
def build_index_for_subset(base: str, subset: str) -> List[Dict[str, str]]:
    sdir = os.path.join(base, subset)
    t1_dir = os.path.join(sdir, "T1")
    t2_dir = os.path.join(sdir, "T2")
    ct_dir = os.path.join(sdir, "CT")
    mn_dir = os.path.join(sdir, "MN")
    ft_dir = os.path.join(sdir, "FT")

    if not os.path.isdir(t1_dir):
        return []

    t1_list = list_images(t1_dir)
    if len(t1_list) == 0:
        return []

    t2_list = list_images(t2_dir) if os.path.isdir(t2_dir) else []
    t2_map = {stem(p): p for p in t2_list}

    ct_map = {stem(p): p for p in list_images(ct_dir)} if os.path.isdir(ct_dir) else {}
    mn_map = {stem(p): p for p in list_images(mn_dir)} if os.path.isdir(mn_dir) else {}
    ft_map = {stem(p): p for p in list_images(ft_dir)} if os.path.isdir(ft_dir) else {}

    items = []
    for p1 in t1_list:
        k = stem(p1)
        item = {
            "subset": subset,
            "key": k,
            "t1": p1,
            "t2": t2_map.get(k, ""),
            "ct": ct_map.get(k, ""),
            "mn": mn_map.get(k, ""),
            "ft": ft_map.get(k, ""),
        }
        if item["ct"] == "" and item["mn"] == "" and item["ft"] == "":
            continue
        items.append(item)

    return items


def build_split(cfg: CFG):
    all_subsets = [str(i) for i in range(10)]
    exist_subsets = [s for s in all_subsets if os.path.isdir(os.path.join(cfg.BASE, s))]

    train_items = []
    for s in cfg.TRAIN_SUBSETS:
        if s in exist_subsets:
            train_items.extend(build_index_for_subset(cfg.BASE, s))

    test_items = []
    for s in cfg.TEST_SUBSETS:
        if s in exist_subsets:
            test_items.extend(build_index_for_subset(cfg.BASE, s))

    val_items = []
    for s in cfg.VAL_SUBSETS:
        if s in exist_subsets:
            val_items.extend(build_index_for_subset(cfg.BASE, s))

    if len(val_items) == 0 and len(train_items) > 0:
        rng = np.random.RandomState(cfg.SEED)
        idx = np.arange(len(train_items))
        rng.shuffle(idx)
        n_val = max(1, int(0.1 * len(train_items)))
        val_idx = set(idx[:n_val].tolist())
        new_train, new_val = [], []
        for i, it in enumerate(train_items):
            (new_val if i in val_idx else new_train).append(it)
        train_items, val_items = new_train, new_val

    return train_items, val_items, test_items


class CarpalTunnelSegDataset(Dataset):
    def __init__(self, items: List[Dict[str, str]], cfg: CFG, is_train: bool):
        self.items = items
        self.cfg = cfg
        self.is_train = is_train

    def __len__(self):
        return len(self.items)

    def _load_inputs(self, item) -> np.ndarray:
        t1 = resize_hw(imread_gray(item["t1"]), self.cfg.IMG_SIZE, is_mask=False)
        t1 = norm01(t1)

        if item["t2"] != "" and os.path.isfile(item["t2"]):
            t2 = resize_hw(imread_gray(item["t2"]), self.cfg.IMG_SIZE, is_mask=False)
            t2 = norm01(t2)
        else:
            t2 = np.zeros_like(t1, dtype=np.float32)

        x = np.stack([t1, t2], axis=0)  # (2,H,W)
        return x.astype(np.float32)

    def _load_multilabel(self, item) -> np.ndarray:
        H, W = self.cfg.IMG_SIZE
        y = np.zeros((3, H, W), dtype=np.uint8)

        if item["ct"] and os.path.isfile(item["ct"]):
            m = resize_hw(imread_gray(item["ct"]), self.cfg.IMG_SIZE, is_mask=True)
            y[0] = binarize_mask(m)

        if item["mn"] and os.path.isfile(item["mn"]):
            m = resize_hw(imread_gray(item["mn"]), self.cfg.IMG_SIZE, is_mask=True)
            y[1] = binarize_mask(m)

        if item["ft"] and os.path.isfile(item["ft"]):
            m = resize_hw(imread_gray(item["ft"]), self.cfg.IMG_SIZE, is_mask=True)
            y[2] = binarize_mask(m)

        return y.astype(np.float32)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        x = self._load_inputs(item)
        y = self._load_multilabel(item)

        if self.is_train:
            # flips
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()
                y = y[:, :, ::-1].copy()
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :].copy()
                y = y[:, ::-1, :].copy()

            # mild brightness/contrast (helps MRI a bit)
            if np.random.rand() < 0.35:
                a = np.random.uniform(0.9, 1.1)   # contrast
                b = np.random.uniform(-0.05, 0.05) # brightness
                x = np.clip(a * x + b, 0.0, 1.0)

        return torch.from_numpy(x), torch.from_numpy(y).float(), {
            "subset": item["subset"],
            "key": item["key"],
            "t1": item["t1"],
        }


# ============================================================
# ViT-TransUNet (Hybrid): CNN skip + ViT encoder + UNet decoder
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
    標準 ViT encoder block:
    - LN → MHA → residual
    - LN → MLP → residual
    """
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


class ViTTransUNet(nn.Module):
    """
    ViT-TransUNet（Hybrid）架構：
    - CNN Encoder：抽多尺度特徵，提供 UNet skip connections
    - Tokenization：取最底層 feature map (H/16, W/16)，用 1x1 conv 投影成 token dim
    - ViT Encoder：對 tokens 做 global self-attention
    - UNet Decoder：逐層 upsample + concat(skip) + conv
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
        self.img_size = img_size
        self.base = base

        assert vit_dim % vit_heads == 0, f"VIT_EMBED_DIM({vit_dim}) must be divisible by VIT_HEADS({vit_heads})."

        # -------------------------
        # CNN Encoder (skip features)
        # -------------------------
        self.d1 = DoubleConv(in_ch, base)          # 256x256
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base * 2)       # 128x128
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base * 2, base * 4)   # 64x64
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base * 4, base * 8)   # 32x32
        self.p4 = nn.MaxPool2d(2)                  # 16x16

        # -------------------------
        # Patch embedding (tokenize)
        # from feature map (N, base*8, H/16, W/16) -> (N, vit_dim, H/16, W/16)
        # -------------------------
        self.patch_proj = nn.Conv2d(base * 8, vit_dim, kernel_size=1, bias=True)

        h16 = img_size[0] // 16
        w16 = img_size[1] // 16
        self.token_len = h16 * w16
        self.vit_dim = vit_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, self.token_len, vit_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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

        # optional bridge conv (helps smooth features)
        self.bridge = DoubleConv(vit_dim, vit_dim)

        # -------------------------
        # UNet Decoder
        # -------------------------
        self.u4 = nn.ConvTranspose2d(vit_dim, base * 8, 2, stride=2)    # 16->32
        self.c4 = DoubleConv(base * 16, base * 8)                      # cat with d4

        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)  # 32->64
        self.c3 = DoubleConv(base * 8, base * 4)                       # cat with d3

        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)  # 64->128
        self.c2 = DoubleConv(base * 4, base * 2)                       # cat with d2

        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)      # 128->256
        self.c1 = DoubleConv(base * 2, base)                           # cat with d1

        self.out = nn.Conv2d(base, out_ch, 1)

    def _to_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (N, D, H, W) -> (N, H*W, D)
        return feat.flatten(2).transpose(1, 2)

    def _to_feat(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # tokens: (N, H*W, D) -> (N, D, H, W)
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN encoder
        d1 = self.d1(x)             # (N, base, 256,256)
        d2 = self.d2(self.p1(d1))   # (N, base*2,128,128)
        d3 = self.d3(self.p2(d2))   # (N, base*4,64,64)
        d4 = self.d4(self.p3(d3))   # (N, base*8,32,32)

        # tokenize from lowest feature map
        f = self.p4(d4)             # (N, base*8,16,16)
        f = self.patch_proj(f)      # (N, vit_dim,16,16)

        n, d, h, w = f.shape
        tokens = self._to_tokens(f)  # (N, 256, vit_dim)

        # add pos embedding
        if tokens.shape[1] == self.token_len:
            tokens = tokens + self.pos_embed
        else:
            # fallback: interpolate pos_embed
            pe = self.pos_embed
            L0 = pe.shape[1]
            s0 = int(np.sqrt(L0))
            pe2 = pe.transpose(1, 2).reshape(1, d, s0, s0)
            pe2 = F.interpolate(pe2, size=(h, w), mode="bilinear", align_corners=False)
            pe2 = pe2.flatten(2).transpose(1, 2)
            tokens = tokens + pe2

        tokens = self.drop(tokens)

        # ViT encoder
        for blk in self.trans_blocks:
            tokens = blk(tokens)

        # reshape back to feature map
        feat = self._to_feat(tokens, h, w)  # (N, vit_dim,16,16)
        feat = self.bridge(feat)

        # UNet decoder + skip
        x = self.u4(feat)                   # (N, base*8,32,32)
        x = torch.cat([x, d4], dim=1)       # (N, base*16,32,32)
        x = self.c4(x)

        x = self.u3(x)                      # (N, base*4,64,64)
        x = torch.cat([x, d3], dim=1)       # (N, base*8,64,64)
        x = self.c3(x)

        x = self.u2(x)                      # (N, base*2,128,128)
        x = torch.cat([x, d2], dim=1)       # (N, base*4,128,128)
        x = self.c2(x)

        x = self.u1(x)                      # (N, base,256,256)
        x = torch.cat([x, d1], dim=1)       # (N, base*2,256,256)
        x = self.c1(x)

        return self.out(x)


# =========================
# Weighted Dice + Weighted BCE
# =========================
class WeightedMultiLabelDiceBCELoss(nn.Module):
    def __init__(
        self,
        channel_weights=(3.0, 1.0, 3.0),
        smooth=1.0,
        eps=1e-6,
        use_bce=True,
        dice_w=1.0,
        bce_w=1.0,
        bce_pos_weights=(4.0, 1.0, 4.0),
    ):
        super().__init__()
        self.smooth = float(smooth)
        self.eps = float(eps)
        self.use_bce = bool(use_bce)
        self.dice_w = float(dice_w)
        self.bce_w = float(bce_w)
        self.register_buffer("cw", torch.tensor(channel_weights, dtype=torch.float32))
        self.register_buffer("posw", torch.tensor(bce_pos_weights, dtype=torch.float32))

    def _weighted_mean(self, v: torch.Tensor) -> torch.Tensor:
        w = self.cw.to(device=v.device, dtype=v.dtype)
        ws = w.sum().clamp_min(1e-12)
        return (v * w).sum() / ws

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)

        inter = (probs * y).sum(dims)  # (C,)
        denom = (probs + y).sum(dims)  # (C,)
        dice = (2 * inter + self.smooth) / (denom + self.smooth + self.eps)
        dice_loss = 1.0 - self._weighted_mean(dice)

        if (not self.use_bce) or (self.bce_w <= 0):
            return self.dice_w * dice_loss

        posw = self.posw.to(device=logits.device, dtype=logits.dtype)
        posw = posw.view(1, -1, *([1] * (logits.ndim - 2)))  # (1,C,1,1)
        bce_map = F.binary_cross_entropy_with_logits(logits, y, pos_weight=posw, reduction="none")
        bce_c = bce_map.mean(dims)  # (C,)
        bce_loss = self._weighted_mean(bce_c)

        return self.dice_w * dice_loss + self.bce_w * bce_loss


@torch.no_grad()
def hard_dice_per_channel_from_logits(logits: torch.Tensor, y: torch.Tensor, thr: Tuple[float, float, float], eps: float = 1e-6) -> np.ndarray:
    probs = torch.sigmoid(logits)
    pred = torch.zeros_like(probs)
    pred[:, 0] = (probs[:, 0] >= thr[0]).float()
    pred[:, 1] = (probs[:, 1] >= thr[1]).float()
    pred[:, 2] = (probs[:, 2] >= thr[2]).float()

    dices = []
    for c in range(3):
        p = pred[:, c]
        g = y[:, c]
        inter = (p * g).sum().item()
        ps = p.sum().item()
        gs = g.sum().item()
        if (ps + gs) == 0:
            d = 1.0
        else:
            d = (2.0 * inter) / (ps + gs + eps)
        dices.append(d)
    return np.array(dices, dtype=np.float32)


# =========================
# Visualization
# =========================
CT_COLOR = (0, 0, 255)
MN_COLOR = (0, 255, 255)
FT_COLOR = (0, 255, 0)


def overlay_multilabel_on_gray(gray01: np.ndarray, y3: np.ndarray, alpha=0.45) -> np.ndarray:
    gray_u8 = (np.clip(gray01, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    color = np.zeros_like(base, dtype=np.uint16)
    ct = y3[0].astype(bool)
    mn = y3[1].astype(bool)
    ft = y3[2].astype(bool)

    if np.any(ct):
        color[ct, 0] += CT_COLOR[0]
        color[ct, 1] += CT_COLOR[1]
        color[ct, 2] += CT_COLOR[2]
    if np.any(mn):
        color[mn, 0] += MN_COLOR[0]
        color[mn, 1] += MN_COLOR[1]
        color[mn, 2] += MN_COLOR[2]
    if np.any(ft):
        color[ft, 0] += FT_COLOR[0]
        color[ft, 1] += FT_COLOR[1]
        color[ft, 2] += FT_COLOR[2]

    color = np.clip(color, 0, 255).astype(np.uint8)
    out = cv2.addWeighted(base, 1.0, color, alpha, 0)
    return out


# =========================
# Sampler weights
# =========================
def build_sampler_weights(train_items: List[Dict[str, str]], cfg: CFG) -> np.ndarray:
    w = np.ones(len(train_items), dtype=np.float64)
    for i, it in enumerate(train_items):
        has_ct = (it.get("ct", "") != "" and os.path.isfile(it["ct"]))
        has_mn = (it.get("mn", "") != "" and os.path.isfile(it["mn"]))
        has_ft = (it.get("ft", "") != "" and os.path.isfile(it["ft"]))
        if has_ct:
            w[i] *= float(cfg.SAMPLE_MUL_CT)
        if has_ft:
            w[i] *= float(cfg.SAMPLE_MUL_FT)
        if has_mn:
            w[i] *= float(cfg.SAMPLE_MUL_MN)
    return w


# =========================
# Train / Eval
# =========================
def run_epoch(model, loader, optimizer, scaler, loss_fn, device, cfg: CFG, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    dice_sum = np.zeros(3, dtype=np.float64)
    n_batches = 0

    amp_enabled = bool(cfg.MIXED_PRECISION and device.type == "cuda")
    autocast_ctx = get_autocast(device_type=device.type, enabled=amp_enabled)

    thr = (cfg.THR_CT, cfg.THR_MN, cfg.THR_FT)

    for x, y, _meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with autocast_ctx:
                logits = model(x)
                loss = loss_fn(logits, y)

        total_loss += float(loss.item())
        dice_sum += hard_dice_per_channel_from_logits(logits, y, thr=thr, eps=cfg.DICE_EPS)
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    avg_dice = (dice_sum / max(1, n_batches)).astype(np.float32)
    mean_dice = float(avg_dice.mean())
    wmean_dice = weighted_mean_np(avg_dice, cfg.CHANNEL_WEIGHTS)
    return avg_loss, avg_dice, mean_dice, wmean_dice


@torch.no_grad()
def test_and_save_vis(model, loader, device, cfg: CFG, out_dir: str):
    model.eval()
    vis_dir = os.path.join(out_dir, "test_vis")
    ensure_dir(vis_dir)

    dice_sum = np.zeros(3, dtype=np.float64)
    n_batches = 0

    amp_enabled = bool(cfg.MIXED_PRECISION and device.type == "cuda")
    autocast_ctx = get_autocast(device_type=device.type, enabled=amp_enabled)

    thr = (cfg.THR_CT, cfg.THR_MN, cfg.THR_FT)

    for x, y, meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(x)

        dice_sum += hard_dice_per_channel_from_logits(logits, y, thr=thr, eps=cfg.DICE_EPS)
        n_batches += 1

        if cfg.SAVE_TEST_VIS:
            probs = torch.sigmoid(logits)
            pred = torch.zeros_like(probs)
            pred[:, 0] = (probs[:, 0] >= cfg.THR_CT).float()
            pred[:, 1] = (probs[:, 1] >= cfg.THR_MN).float()
            pred[:, 2] = (probs[:, 2] >= cfg.THR_FT).float()

            pred = pred.cpu().numpy()
            gt = y.cpu().numpy()
            x_np = x.cpu().numpy()

            for i in range(pred.shape[0]):
                subset = meta["subset"][i]
                key = meta["key"][i]
                t1 = x_np[i, 0]

                ov_gt = overlay_multilabel_on_gray(t1, gt[i], alpha=0.45)
                ov_pr = overlay_multilabel_on_gray(t1, pred[i], alpha=0.45)

                cat = np.concatenate([ov_gt, ov_pr], axis=1)
                fn = f"subset{subset}_{key}_GT_PRED.png"
                cv2.imwrite(os.path.join(vis_dir, fn), cat)

    avg_dice = (dice_sum / max(1, n_batches)).astype(np.float32)
    mean_dice = float(avg_dice.mean())
    wmean_dice = weighted_mean_np(avg_dice, cfg.CHANNEL_WEIGHTS)
    return avg_dice, mean_dice, wmean_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=CFG.BASE)
    parser.add_argument("--out", type=str, default=CFG.OUT_DIR)
    parser.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    parser.add_argument("--bs", type=int, default=CFG.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=CFG.LR)
    parser.add_argument("--workers", type=int, default=CFG.NUM_WORKERS)
    parser.add_argument("--no_amp", action="store_true")

    # thresholds
    parser.add_argument("--thr_ct", type=float, default=CFG.THR_CT)
    parser.add_argument("--thr_mn", type=float, default=CFG.THR_MN)
    parser.add_argument("--thr_ft", type=float, default=CFG.THR_FT)

    # ViT config
    parser.add_argument("--tdim", type=int, default=CFG.VIT_EMBED_DIM)
    parser.add_argument("--tlayers", type=int, default=CFG.VIT_DEPTH)
    parser.add_argument("--theads", type=int, default=CFG.VIT_HEADS)
    parser.add_argument("--tmlp", type=float, default=CFG.VIT_MLP_RATIO)
    parser.add_argument("--tdrop", type=float, default=CFG.VIT_DROPOUT)

    # loss weights
    parser.add_argument("--w_ct", type=float, default=CFG.CHANNEL_WEIGHTS[0])
    parser.add_argument("--w_mn", type=float, default=CFG.CHANNEL_WEIGHTS[1])
    parser.add_argument("--w_ft", type=float, default=CFG.CHANNEL_WEIGHTS[2])
    parser.add_argument("--no_bce", action="store_true")
    parser.add_argument("--dice_w", type=float, default=CFG.DICE_LOSS_WEIGHT)
    parser.add_argument("--bce_w", type=float, default=CFG.BCE_LOSS_WEIGHT)
    parser.add_argument("--pw_ct", type=float, default=CFG.BCE_POS_WEIGHTS[0])
    parser.add_argument("--pw_mn", type=float, default=CFG.BCE_POS_WEIGHTS[1])
    parser.add_argument("--pw_ft", type=float, default=CFG.BCE_POS_WEIGHTS[2])

    # sampler
    parser.add_argument("--no_sampler", action="store_true")
    parser.add_argument("--sm_ct", type=float, default=CFG.SAMPLE_MUL_CT)
    parser.add_argument("--sm_mn", type=float, default=CFG.SAMPLE_MUL_MN)
    parser.add_argument("--sm_ft", type=float, default=CFG.SAMPLE_MUL_FT)

    args = parser.parse_args()

    cfg = CFG()
    cfg.BASE = args.base
    cfg.OUT_DIR = args.out
    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.bs
    cfg.LR = args.lr
    cfg.NUM_WORKERS = args.workers
    cfg.MIXED_PRECISION = (not args.no_amp)
    cfg.EXP_NAME = time.strftime("exp_%Y%m%d_%H%M%S")

    cfg.THR_CT = float(args.thr_ct)
    cfg.THR_MN = float(args.thr_mn)
    cfg.THR_FT = float(args.thr_ft)

    cfg.VIT_EMBED_DIM = int(args.tdim)
    cfg.VIT_DEPTH = int(args.tlayers)
    cfg.VIT_HEADS = int(args.theads)
    cfg.VIT_MLP_RATIO = float(args.tmlp)
    cfg.VIT_DROPOUT = float(args.tdrop)

    cfg.CHANNEL_WEIGHTS = (float(args.w_ct), float(args.w_mn), float(args.w_ft))
    cfg.USE_BCE = (not args.no_bce)
    cfg.DICE_LOSS_WEIGHT = float(args.dice_w)
    cfg.BCE_LOSS_WEIGHT = float(args.bce_w)
    cfg.BCE_POS_WEIGHTS = (float(args.pw_ct), float(args.pw_mn), float(args.pw_ft))

    cfg.USE_SAMPLER = (not args.no_sampler)
    cfg.SAMPLE_MUL_CT = float(args.sm_ct)
    cfg.SAMPLE_MUL_MN = float(args.sm_mn)
    cfg.SAMPLE_MUL_FT = float(args.sm_ft)

    set_seed(cfg.SEED)

    run_dir = os.path.join(cfg.OUT_DIR, cfg.EXP_NAME)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "checkpoints"))

    print(f"[cfg] BASE={cfg.BASE}")
    print(f"[cfg] run_dir={run_dir}")
    print(f"[cfg] train={cfg.TRAIN_SUBSETS} test={cfg.TEST_SUBSETS} val={cfg.VAL_SUBSETS} (val fallback: split 10%)")
    print(f"[cfg] img_size={cfg.IMG_SIZE} in_ch={cfg.IN_CHANNELS} out_ch={cfg.OUT_CHANNELS} amp={cfg.MIXED_PRECISION}")
    print(f"[cfg] thr(CT,MN,FT)=({cfg.THR_CT},{cfg.THR_MN},{cfg.THR_FT})")
    print(f"[cfg] ViT: dim={cfg.VIT_EMBED_DIM} depth={cfg.VIT_DEPTH} heads={cfg.VIT_HEADS} mlp_ratio={cfg.VIT_MLP_RATIO} drop={cfg.VIT_DROPOUT}")
    print(f"[cfg] CHANNEL_WEIGHTS={cfg.CHANNEL_WEIGHTS} USE_BCE={cfg.USE_BCE} dice_w={cfg.DICE_LOSS_WEIGHT} bce_w={cfg.BCE_LOSS_WEIGHT} POSW={cfg.BCE_POS_WEIGHTS}")
    print(f"[cfg] USE_SAMPLER={cfg.USE_SAMPLER} SAMPLE_MUL(CT,MN,FT)=({cfg.SAMPLE_MUL_CT},{cfg.SAMPLE_MUL_MN},{cfg.SAMPLE_MUL_FT})")

    train_items, val_items, test_items = build_split(cfg)
    print(f"[data] train={len(train_items)} val={len(val_items)} test={len(test_items)}")
    if len(train_items) == 0 or len(test_items) == 0:
        print("[error] train/test items are empty. Check folder structure & mask existence.")
        return

    train_ds = CarpalTunnelSegDataset(train_items, cfg, is_train=True)
    val_ds = CarpalTunnelSegDataset(val_items, cfg, is_train=False) if len(val_items) > 0 else None
    test_ds = CarpalTunnelSegDataset(test_items, cfg, is_train=False)

    if cfg.USE_SAMPLER:
        sw = build_sampler_weights(train_items, cfg)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sw, dtype=torch.double),
            num_samples=len(sw),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False)

    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    model = ViTTransUNet(
        in_ch=cfg.IN_CHANNELS,
        out_ch=cfg.OUT_CHANNELS,
        base=32,
        img_size=cfg.IMG_SIZE,
        vit_dim=cfg.VIT_EMBED_DIM,
        vit_depth=cfg.VIT_DEPTH,
        vit_heads=cfg.VIT_HEADS,
        vit_mlp_ratio=cfg.VIT_MLP_RATIO,
        vit_dropout=cfg.VIT_DROPOUT,
    ).to(device)

    loss_fn = WeightedMultiLabelDiceBCELoss(
        channel_weights=cfg.CHANNEL_WEIGHTS,
        smooth=cfg.DICE_SMOOTH,
        eps=cfg.DICE_EPS,
        use_bce=cfg.USE_BCE,
        dice_w=cfg.DICE_LOSS_WEIGHT,
        bce_w=cfg.BCE_LOSS_WEIGHT,
        bce_pos_weights=cfg.BCE_POS_WEIGHTS,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scaler = get_grad_scaler(device_type=device.type, enabled=bool(cfg.MIXED_PRECISION and device.type == "cuda"))

    # simple scheduler (helps small data)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    best_score = -1.0
    best_path = ""

    for ep in range(1, cfg.EPOCHS + 1):
        tr_loss, tr_dice, tr_mean, tr_wmean = run_epoch(model, train_loader, optimizer, scaler, loss_fn, device, cfg, train=True)

        if val_loader is not None:
            va_loss, va_dice, va_mean, va_wmean = run_epoch(model, val_loader, optimizer, scaler, loss_fn, device, cfg, train=False)
            score = va_wmean  # ✅ focus CT/FT
            score_tag = "val"
        else:
            va_loss, va_dice, va_mean, va_wmean = tr_loss, tr_dice, tr_mean, tr_wmean
            score = tr_wmean
            score_tag = "train"

        scheduler.step()

        print(
            f"[ep {ep:03d}] "
            f"train loss={tr_loss:.4f} meanDice={tr_mean:.4f} wMeanDice={tr_wmean:.4f} "
            f"| {score_tag} loss={va_loss:.4f} meanDice={va_mean:.4f} wMeanDice={va_wmean:.4f} "
            f"(CT={va_dice[0]:.4f} MN={va_dice[1]:.4f} FT={va_dice[2]:.4f})"
        )

        if score > best_score:
            best_score = score
            best_path = os.path.join(run_dir, "checkpoints", f"best_{best_score:.4f}.pt")
            torch.save(
                {
                    "cfg": cfg.__dict__,
                    "epoch": ep,
                    "best_score": best_score,
                    "model_state": model.state_dict(),
                    "model_name": "ViTTransUNet",
                },
                best_path,
            )
            print(f"  -> [save] {best_path}")

    print(f"[best] wMeanDice={best_score:.4f} ckpt={best_path}")

    ck = torch.load(best_path, map_location=device)
    model.load_state_dict(ck["model_state"], strict=True)

    test_dice, test_mean, test_wmean = test_and_save_vis(model, test_loader, device, cfg, run_dir)
    print(
        f"[test] meanDice={test_mean:.4f} wMeanDice={test_wmean:.4f} "
        f"(CT={test_dice[0]:.4f} MN={test_dice[1]:.4f} FT={test_dice[2]:.4f})"
    )
    print(f"[out] test overlays: {os.path.join(run_dir, 'test_vis')}")

    last_path = os.path.join(run_dir, "checkpoints", "last.pt")
    torch.save(
        {
            "cfg": cfg.__dict__,
            "epoch": cfg.EPOCHS,
            "best_score": best_score,
            "model_state": model.state_dict(),
            "model_name": "ViTTransUNet",
        },
        last_path,
    )
    print(f"[save] last ckpt: {last_path}")


if __name__ == "__main__":
    main()
