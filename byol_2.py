# ============================================================
# BYOL Finetune SigLIP2 Vision Encoder from Multiple Video Folders
# PATCH-TOKEN LEVEL
#
# - Sample random frames from videos under multiple folders
# - Supports AGIBOT + Mobile ALOHA joint BYOL training
# - Each dataset folder is split into train/val separately
# - Optional source-balanced sampling across datasets
# - Two views of the SAME frame: clean + augmented
# - Teacher ALWAYS sees CLEAN
# - Student ALWAYS sees AUG
# - Use Vision LAST_HIDDEN_STATE patch tokens directly: (B, N, D)
# - Token-level BYOL: projector/predictor operate on each token
# - NO random resized crop in augmentation
# - BYOL online/target networks with EMA teacher
# - Save finetuned model in HuggingFace format every epoch
# ============================================================

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import av
import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, get_worker_info
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


# =========================
# 0) Config
# =========================
@dataclass
class CFG:
    EXPERIMENT_NAME: str = "BYOL_SIGLIP2_PATCHTOKENS_JOINT_AGIBOT_MOBILEALOHA"

    # Put your two dataset folders here.
    # The script recursively finds mp4 / avi / mov / mkv files under each folder.
    VIDEO_DIRS: List[str] = field(default_factory=lambda: [
        "/home/skyler/Desktop/graduation_real_robot_v2/AgiBotWorldChallenge-2025/byol_video",
        "/home/skyler/Desktop/graduation_real_robot/real_robot_data",
    ])

    OUTPUT_DIR: str = "./byol_siglip2_agibot_mobilealoha_joint"
    MODEL_NAME: str = "google/siglip2-base-patch16-512"

    # Split each folder separately.
    VAL_RATIO: float = 0.1

    # If True, each dataset source is sampled approximately equally.
    # If False, videos are sampled uniformly from the merged video list.
    SOURCE_BALANCED: bool = True

    # Sampling
    SAMPLES_PER_EPOCH: int = 10000
    VAL_SAMPLES: int = 500
    MIN_FRAMES: int = 30

    # Training
    EPOCHS: int = 15
    BATCH_SIZE: int = 4
    GRAD_ACCUM: int = 8
    NUM_WORKERS: int = 4
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP: bool = True

    # Finetune control
    FINETUNE_VISION: bool = True
    TRAIN_LAST_N_VIT_BLOCKS: int = 8

    # BYOL MLP sizes
    PROJ_HIDDEN: int = 2048
    PROJ_OUT: int = 256

    # Optim
    LR_VIT: float = 1e-5
    LR_HEAD: float = 1e-4
    WEIGHT_DECAY: float = 1e-2
    GRAD_CLIP: float = 1.0

    # EMA
    EMA_MOMENTUM: float = 0.996

    # Augmentation (photometric + tiny rotation; no crop/resize)
    COLOR_JITTER_PROB: float = 0.8
    COLOR_JITTER_MIN: float = 0.6    # ±40%
    COLOR_JITTER_MAX: float = 1.4
    GRAYSCALE_PROB: float = 0.2
    BLUR_PROB: float = 0.8
    BLUR_RADIUS_MIN: float = 0.5
    BLUR_RADIUS_MAX: float = 2.0
    ROT_PROB: float = 0.2            # random ±1° rotation
    ROT_MAX_DEG: float = 1.0

    # Debug
    PRINT_SHAPES_ONCE: bool = True
    PRINT_TRAINABLE_PARAM_SUMMARY: bool = True


cfg = CFG()
print(f"★ Running Experiment: {cfg.EXPERIMENT_NAME} ★")
print(f"[Info] Device = {cfg.DEVICE}")


# =========================
# 1) Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_videos(input_dir: Path, exts=(".mp4", ".avi", ".mov", ".mkv")) -> List[Path]:
    return sorted([
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ])


def get_nframes_cv2(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def get_nframes_pyav_or_cv2(path: Path) -> int:
    """
    PyAV stream.frames can be 0 for some videos.
    Fall back to OpenCV frame count when needed.
    """
    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        n = int(stream.frames or 0)
        container.close()
        if n > 0:
            return n
    except Exception:
        pass

    return get_nframes_cv2(path)


def read_frame_as_pil_pyav(video_path: Path, frame_idx: int) -> Optional[Image.Image]:
    """
    Decode a specific frame index using PyAV and return PIL.Image.
    """
    try:
        container = av.open(str(video_path))
    except Exception as e:
        print(f"❌ Cannot open video {video_path}: {e}")
        return None

    try:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_idx:
                img = frame.to_image().convert("RGB")
                container.close()
                return img
    except Exception as e:
        print(f"❌ Cannot decode frame {frame_idx} from {video_path}: {e}")
    finally:
        try:
            container.close()
        except Exception:
            pass

    return None


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collect_video_splits(
    video_dirs: List[str],
    seed: int = 42,
    val_ratio: float = 0.1,
) -> Tuple[List[List[Path]], List[List[Path]], List[str]]:
    """
    For each dataset folder:
    - recursively find videos
    - split train/val inside that folder
    - return train groups and val groups

    Return:
        train_groups: List[List[Path]]
        val_groups:   List[List[Path]]
        source_names: List[str]
    """
    rng = random.Random(seed)

    train_groups: List[List[Path]] = []
    val_groups: List[List[Path]] = []
    source_names: List[str] = []

    print("\n[Info] Collecting videos from multiple folders:")

    for root in video_dirs:
        root_path = Path(root)

        if not root_path.exists():
            raise FileNotFoundError(f"Video folder does not exist: {root}")

        files = find_videos(root_path)

        if len(files) == 0:
            raise FileNotFoundError(f"No video files found under: {root}")

        rng.shuffle(files)

        if len(files) == 1:
            train_files = files
            val_files = files
        else:
            split_idx = int(len(files) * (1.0 - val_ratio))
            split_idx = max(1, min(split_idx, len(files) - 1))
            train_files = files[:split_idx]
            val_files = files[split_idx:]

        source_name = root_path.name

        train_groups.append(train_files)
        val_groups.append(val_files)
        source_names.append(source_name)

        print(f"  - Source: {source_name}")
        print(f"    path  = {root}")
        print(f"    total = {len(files)}, train = {len(train_files)}, val = {len(val_files)}")

    print(f"\n[Info] Total train videos = {sum(len(g) for g in train_groups)}")
    print(f"[Info] Total val videos   = {sum(len(g) for g in val_groups)}")
    print(f"[Info] Source balanced sampling = {cfg.SOURCE_BALANCED}\n")

    return train_groups, val_groups, source_names


# =========================
# 2) Augmentations
# =========================
# #region agent log
import json
import time
_DEBUG_LOG_PATH = "/home/mark/Desktop/swiftplan_code-main/.cursor/debug-9bd7c0.log"
_DEBUG_AUG_COUNT = 0


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "aug-pre"):
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "9bd7c0",
                "runId": run_id,
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
# #endregion


def _to_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L").convert("RGB")


def augment_view(img: Image.Image, rng: np.random.RandomState) -> Image.Image:
    """
    No crop / no random resize.
    HF processor may still deterministically resize to model input size.

    - color jitter
    - grayscale
    - gaussian blur
    - small random rotation (±ROT_MAX_DEG)
    """
    global _DEBUG_AUG_COUNT
    applied = {"color": False, "gray": False, "blur": False, "rot_deg": None}

    if rng.rand() < cfg.COLOR_JITTER_PROB:
        lo, hi = cfg.COLOR_JITTER_MIN, cfg.COLOR_JITTER_MAX
        img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(lo, hi)))
        img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(lo, hi)))
        img = ImageEnhance.Color(img).enhance(float(rng.uniform(lo, hi)))
        applied["color"] = True

    if rng.rand() < cfg.GRAYSCALE_PROB:
        img = _to_grayscale(img)
        applied["gray"] = True

    if rng.rand() < cfg.BLUR_PROB:
        radius = float(rng.uniform(cfg.BLUR_RADIUS_MIN, cfg.BLUR_RADIUS_MAX))
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        applied["blur"] = True

    if rng.rand() < cfg.ROT_PROB:
        angle = float(rng.uniform(-cfg.ROT_MAX_DEG, cfg.ROT_MAX_DEG))
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
        applied["rot_deg"] = angle

    # #region agent log
    if _DEBUG_AUG_COUNT < 8:
        _DEBUG_AUG_COUNT += 1
        _agent_log(
            "H1",
            "byol.py:augment_view",
            "aug applied",
            {
                "n": _DEBUG_AUG_COUNT,
                "applied": applied,
                "cfg": {
                    "COLOR_JITTER_PROB": cfg.COLOR_JITTER_PROB,
                    "COLOR_JITTER_MIN": cfg.COLOR_JITTER_MIN,
                    "COLOR_JITTER_MAX": cfg.COLOR_JITTER_MAX,
                    "GRAYSCALE_PROB": cfg.GRAYSCALE_PROB,
                    "BLUR_PROB": cfg.BLUR_PROB,
                    "BLUR_RADIUS_MIN": cfg.BLUR_RADIUS_MIN,
                    "BLUR_RADIUS_MAX": cfg.BLUR_RADIUS_MAX,
                    "ROT_PROB": cfg.ROT_PROB,
                    "ROT_MAX_DEG": cfg.ROT_MAX_DEG,
                },
            },
        )
    # #endregion

    return img


# =========================
# 3) Dataset
# =========================
class VideoFrameBYOLDatasetPyAV(Dataset):
    """
    Samples one random frame and returns:
        clean image, augmented image

    Supports:
    - merged sampling
    - source-balanced sampling
    """
    def __init__(
        self,
        video_groups: List[List[Path]],
        source_names: List[str],
        samples_per_epoch: int,
        seed: int = 42,
        min_frames: int = 30,
        source_balanced: bool = True,
    ):
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.min_frames = min_frames
        self.source_balanced = source_balanced

        assert len(video_groups) == len(source_names)

        self.groups = []
        self.flat_files: List[Path] = []
        self.flat_nframes: List[int] = []

        print("[Info] Filtering valid videos:")

        for source_name, files in zip(source_names, video_groups):
            valid_files = []
            valid_nframes = []

            for p in files:
                n = get_nframes_pyav_or_cv2(p)
                if n >= min_frames:
                    valid_files.append(p)
                    valid_nframes.append(n)

            if len(valid_files) > 0:
                self.groups.append({
                    "name": source_name,
                    "files": valid_files,
                    "nframes": valid_nframes,
                })

                self.flat_files.extend(valid_files)
                self.flat_nframes.extend(valid_nframes)

            print(f"  - {source_name}: valid {len(valid_files)} / raw {len(files)}")

        if len(self.flat_files) == 0:
            raise RuntimeError("No valid videos available. Check paths, codecs, or MIN_FRAMES.")

        if self.source_balanced:
            print("[Info] Dataset sampling mode: source-balanced")
        else:
            print("[Info] Dataset sampling mode: merged-uniform")

        print(f"[Info] Total valid videos in this split: {len(self.flat_files)}")

        self._worker_rng = None
        self._worker_id = None

    def __len__(self):
        return self.samples_per_epoch

    def _get_rng(self) -> np.random.RandomState:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if self._worker_rng is None or self._worker_id != worker_id:
            self._worker_id = worker_id
            self._worker_rng = np.random.RandomState(self.seed + 1009 * worker_id)

        return self._worker_rng

    def _sample_video(self, rng: np.random.RandomState) -> Tuple[Path, int]:
        if self.source_balanced and len(self.groups) > 1:
            group_idx = int(rng.randint(0, len(self.groups)))
            group = self.groups[group_idx]
            vid_i = int(rng.randint(0, len(group["files"])))
            return group["files"][vid_i], group["nframes"][vid_i]

        vid_i = int(rng.randint(0, len(self.flat_files)))
        return self.flat_files[vid_i], self.flat_nframes[vid_i]

    def __getitem__(self, idx):
        rng = self._get_rng()

        for _ in range(20):
            path, n = self._sample_video(rng)

            if n <= 0:
                continue

            frame_idx = int(rng.randint(0, n))
            x = read_frame_as_pil_pyav(path, frame_idx)

            if x is None:
                continue

            v_clean = x
            v_aug = augment_view(x, rng)
            return v_clean, v_aug

        raise RuntimeError("Failed to sample a readable frame after 20 attempts.")


def collate_fn(batch):
    v_clean, v_aug = zip(*batch)
    return list(v_clean), list(v_aug)


# =========================
# 4) Encode helper
# =========================
def encode_vision_last_hidden_tokens(
    model: AutoModel,
    processor: AutoProcessor,
    images: List[Image.Image],
    device: str,
) -> torch.Tensor:
    """
    Return patch tokens:
        (B, N, D)
    """
    inputs = processor(images=images, return_tensors="pt").to(device)

    if not hasattr(model, "vision_model"):
        raise ValueError("Model has no vision_model attribute. Please check SigLIP/SigLIP2 loading.")

    vision_out = model.vision_model(
        pixel_values=inputs["pixel_values"],
        output_hidden_states=False,
        return_dict=True,
    )

    return vision_out.last_hidden_state


# =========================
# 5) Token-level BYOL modules
# =========================
class TokenBYOLMLP(nn.Module):
    """
    Supports:
        (B, N, D) -> (B, N, out_dim)
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def byol_loss_tokens(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Token-level BYOL loss.
    p, z: (B, N, P)
    """
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, m: float):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(m).add_(sp.data, alpha=(1.0 - m))


@torch.no_grad()
def ema_update_vision_only(teacher: AutoModel, student: AutoModel, m: float):
    if hasattr(teacher, "vision_model") and hasattr(student, "vision_model"):
        ema_update(teacher.vision_model, student.vision_model, m)
    else:
        ema_update(teacher, student, m)


# =========================
# 6) Finetune scope
# =========================
def set_vision_trainable(siglip2: AutoModel, train_last_n_blocks: int):
    """
    train_last_n_blocks = 0:
        train all vision tower params

    train_last_n_blocks > 0:
        train last N Transformer blocks + all LayerNorms
    """
    for p in siglip2.parameters():
        p.requires_grad = False

    if not hasattr(siglip2, "vision_model"):
        print("[Warn] No vision_model found. Unfreezing the whole model.")
        for p in siglip2.parameters():
            p.requires_grad = True
        return

    vm = siglip2.vision_model

    layers = None
    if hasattr(vm, "encoder") and hasattr(vm.encoder, "layers"):
        layers = vm.encoder.layers
    elif hasattr(vm, "encoder") and hasattr(vm.encoder, "layer"):
        layers = vm.encoder.layer

    if layers is None:
        print("[Warn] Cannot find encoder layers. Unfreezing whole vision_model.")
        for p in vm.parameters():
            p.requires_grad = True
        return

    n = len(layers)
    start = 0 if train_last_n_blocks <= 0 else max(0, n - train_last_n_blocks)

    print(f"[Info] Unfreezing vision blocks: {start}..{n - 1} / total {n}")

    for i in range(start, n):
        for p in layers[i].parameters():
            p.requires_grad = True

    for mod in vm.modules():
        if isinstance(mod, nn.LayerNorm):
            for p in mod.parameters():
                p.requires_grad = True


def get_trainable_vision_params(siglip2: AutoModel):
    if hasattr(siglip2, "vision_model"):
        return [p for p in siglip2.vision_model.parameters() if p.requires_grad]
    return [p for p in siglip2.parameters() if p.requires_grad]


# =========================
# 7) Main
# =========================
def main():
    set_seed(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    train_groups, val_groups, source_names = collect_video_splits(
        cfg.VIDEO_DIRS,
        seed=cfg.SEED,
        val_ratio=cfg.VAL_RATIO,
    )

    train_ds = VideoFrameBYOLDatasetPyAV(
        video_groups=train_groups,
        source_names=source_names,
        samples_per_epoch=cfg.SAMPLES_PER_EPOCH,
        seed=cfg.SEED,
        min_frames=cfg.MIN_FRAMES,
        source_balanced=cfg.SOURCE_BALANCED,
    )

    val_ds = VideoFrameBYOLDatasetPyAV(
        video_groups=val_groups,
        source_names=source_names,
        samples_per_epoch=cfg.VAL_SAMPLES,
        seed=cfg.SEED + 1,
        min_frames=cfg.MIN_FRAMES,
        source_balanced=cfg.SOURCE_BALANCED,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    # ---- Load student ----
    print(f"[Info] Loading model: {cfg.MODEL_NAME}")
    student = AutoModel.from_pretrained(cfg.MODEL_NAME).to(cfg.DEVICE)
    processor = AutoProcessor.from_pretrained(cfg.MODEL_NAME)

    # ---- Freeze / unfreeze vision ----
    if cfg.FINETUNE_VISION:
        set_vision_trainable(student, cfg.TRAIN_LAST_N_VIT_BLOCKS)
    else:
        for p in student.parameters():
            p.requires_grad = False

    if cfg.PRINT_TRAINABLE_PARAM_SUMMARY:
        print(f"[Info] Trainable param count, student total = {count_trainable_params(student):,}")

    # ---- Build teacher ----
    teacher = AutoModel.from_pretrained(cfg.MODEL_NAME).to(cfg.DEVICE)
    teacher.load_state_dict(student.state_dict(), strict=True)
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad = False

    # ---- Infer token dimension ----
    with torch.no_grad():
        v_clean_pil, _ = next(iter(train_dl))
        tok = encode_vision_last_hidden_tokens(
            teacher,
            processor,
            v_clean_pil,
            cfg.DEVICE,
        )

        feat_dim = tok.shape[-1]
        n_tokens = tok.shape[1]

        if cfg.PRINT_SHAPES_ONCE:
            print(f"[Info] Tokens shape = {tuple(tok.shape)} => N = {n_tokens}, D = {feat_dim}")

    # ---- Projector and predictor ----
    proj_s = TokenBYOLMLP(
        in_dim=feat_dim,
        hidden_dim=cfg.PROJ_HIDDEN,
        out_dim=cfg.PROJ_OUT,
    ).to(cfg.DEVICE)

    pred_s = TokenBYOLMLP(
        in_dim=cfg.PROJ_OUT,
        hidden_dim=cfg.PROJ_HIDDEN,
        out_dim=cfg.PROJ_OUT,
    ).to(cfg.DEVICE)

    proj_t = TokenBYOLMLP(
        in_dim=feat_dim,
        hidden_dim=cfg.PROJ_HIDDEN,
        out_dim=cfg.PROJ_OUT,
    ).to(cfg.DEVICE)

    proj_t.load_state_dict(proj_s.state_dict(), strict=True)
    proj_t.eval()

    for p in proj_t.parameters():
        p.requires_grad = False

    total_student = sum(p.numel() for p in student.parameters())
    trainable_vit = sum(p.numel() for p in student.parameters() if p.requires_grad)
    proj_s_params = sum(p.numel() for p in proj_s.parameters())
    pred_s_params = sum(p.numel() for p in pred_s.parameters())
    byol_head_params = proj_s_params + pred_s_params
    total_trainable = trainable_vit + byol_head_params

    print(f"\n{'=' * 60}")
    print(f"[Params] SigLIP2 total           : {total_student:,} ({total_student / 1e6:.1f}M)")
    print(f"[Params] ViT trainable           : {trainable_vit:,} ({trainable_vit / 1e6:.1f}M)")
    print(f"[Params] BYOL Projector          : {proj_s_params:,} ({proj_s_params / 1e6:.1f}M)")
    print(f"[Params] BYOL Predictor          : {pred_s_params:,} ({pred_s_params / 1e6:.1f}M)")
    print(f"[Params] Total trainable Stage 1 : {total_trainable:,} ({total_trainable / 1e6:.1f}M)")
    print(f"{'=' * 60}\n")

    # ---- Optimizer ----
    vit_params = get_trainable_vision_params(student)

    optim_groups = []

    if len(vit_params) > 0:
        optim_groups.append({
            "params": vit_params,
            "lr": cfg.LR_VIT,
        })

    optim_groups += [
        {
            "params": proj_s.parameters(),
            "lr": cfg.LR_HEAD,
        },
        {
            "params": pred_s.parameters(),
            "lr": cfg.LR_HEAD,
        },
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        enabled=(cfg.USE_AMP and cfg.DEVICE == "cuda")
    )

    best_val = float("inf")

    print("[Info] Start BYOL training.")
    print(f"       FINETUNE_VISION        = {cfg.FINETUNE_VISION}")
    print(f"       TRAIN_LAST_N_VIT_BLOCKS = {cfg.TRAIN_LAST_N_VIT_BLOCKS}")
    print(f"       SOURCE_BALANCED        = {cfg.SOURCE_BALANCED}")
    print(f"       Color jitter           = prob={cfg.COLOR_JITTER_PROB}, range=[{cfg.COLOR_JITTER_MIN}, {cfg.COLOR_JITTER_MAX}]")
    print(f"       Grayscale              = prob={cfg.GRAYSCALE_PROB}")
    print(f"       Gaussian blur          = prob={cfg.BLUR_PROB}, radius=[{cfg.BLUR_RADIUS_MIN}, {cfg.BLUR_RADIUS_MAX}]")
    print(f"       Rotation               = prob={cfg.ROT_PROB}, max_angle=±{cfg.ROT_MAX_DEG}°")
    print("       Teacher target          = clean image")
    print("       Student input           = augmented image")
    print("       Loss                    = token-level BYOL loss\n")
    # #region agent log
    _agent_log("H2", "byol.py:main", "aug cfg at train start", {
        "COLOR_JITTER_PROB": cfg.COLOR_JITTER_PROB,
        "COLOR_JITTER_MIN": cfg.COLOR_JITTER_MIN,
        "COLOR_JITTER_MAX": cfg.COLOR_JITTER_MAX,
        "GRAYSCALE_PROB": cfg.GRAYSCALE_PROB,
        "BLUR_PROB": cfg.BLUR_PROB,
        "BLUR_RADIUS_MIN": cfg.BLUR_RADIUS_MIN,
        "BLUR_RADIUS_MAX": cfg.BLUR_RADIUS_MAX,
        "ROT_PROB": cfg.ROT_PROB,
        "ROT_MAX_DEG": cfg.ROT_MAX_DEG,
    })
    # #endregion

    # =========================
    # Training loop
    # =========================
    for epoch in range(1, cfg.EPOCHS + 1):
        student.train()
        proj_s.train()
        pred_s.train()

        optimizer.zero_grad(set_to_none=True)

        tr_loss = 0.0
        steps = 0

        pbar = tqdm(train_dl, desc=f"Ep {epoch}/{cfg.EPOCHS} [Train]")

        for it, (v_clean_pil, v_aug_pil) in enumerate(pbar):
            # ---- Teacher forward: clean only ----
            with torch.no_grad():
                t_clean = encode_vision_last_hidden_tokens(
                    teacher,
                    processor,
                    v_clean_pil,
                    cfg.DEVICE,
                )

                z_t = proj_t(t_clean)

            # ---- Student forward: augmented only ----
            with torch.amp.autocast(
                device_type="cuda",
                enabled=(cfg.USE_AMP and cfg.DEVICE == "cuda"),
            ):
                s_aug = encode_vision_last_hidden_tokens(
                    student,
                    processor,
                    v_aug_pil,
                    cfg.DEVICE,
                )

                z = proj_s(s_aug)
                p = pred_s(z)

                loss = byol_loss_tokens(p, z_t.detach())
                loss = loss / float(cfg.GRAD_ACCUM)

            scaler.scale(loss).backward()

            do_step = ((it + 1) % cfg.GRAD_ACCUM == 0) or ((it + 1) == len(train_dl))

            if do_step:
                if cfg.GRAD_CLIP and cfg.GRAD_CLIP > 0:
                    scaler.unscale_(optimizer)

                    params_for_clip = (
                        vit_params
                        + list(proj_s.parameters())
                        + list(pred_s.parameters())
                    )

                    torch.nn.utils.clip_grad_norm_(
                        params_for_clip,
                        cfg.GRAD_CLIP,
                    )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # EMA update teacher vision tower and teacher projector.
                ema_update_vision_only(
                    teacher,
                    student,
                    cfg.EMA_MOMENTUM,
                )

                ema_update(
                    proj_t,
                    proj_s,
                    cfg.EMA_MOMENTUM,
                )

            tr_loss += float(loss.item()) * float(cfg.GRAD_ACCUM)
            steps += 1

            pbar.set_postfix({
                "L": f"{tr_loss / max(steps, 1):.3f}"
            })

        # =========================
        # Validation
        # =========================
        student.eval()
        proj_s.eval()
        pred_s.eval()

        va_loss = 0.0
        vsteps = 0

        with torch.no_grad():
            for v_clean_pil, v_aug_pil in val_dl:
                t_clean = encode_vision_last_hidden_tokens(
                    teacher,
                    processor,
                    v_clean_pil,
                    cfg.DEVICE,
                )

                z_t = proj_t(t_clean)

                s_aug = encode_vision_last_hidden_tokens(
                    student,
                    processor,
                    v_aug_pil,
                    cfg.DEVICE,
                )

                z = proj_s(s_aug)
                p = pred_s(z)

                loss = byol_loss_tokens(p, z_t)

                va_loss += float(loss.item())
                vsteps += 1

        avg_tr = tr_loss / max(steps, 1)
        avg_va = va_loss / max(vsteps, 1)

        print(f"\nEpoch {epoch} Result:")
        print(f"  Train Loss = {avg_tr:.6f}")
        print(f"  Val   Loss = {avg_va:.6f}")

        # =========================
        # Save checkpoint every epoch
        # =========================
        epoch_dir = os.path.join(cfg.OUTPUT_DIR, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        save_dir = os.path.join(
            epoch_dir,
            "finetuned_byol_vit_patchtokens_tokenlevel",
        )

        student.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)

        torch.save(
            {
                "proj_s": proj_s.state_dict(),
                "pred_s": pred_s.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "train_loss": avg_tr,
                "val_loss": avg_va,
                "best_val_loss_so_far": best_val,
            },
            os.path.join(epoch_dir, "byol_heads.pt"),
        )

        print(f"  [CKPT] Saved epoch checkpoint -> {epoch_dir}")

        # =========================
        # Save best
        # =========================
        if avg_va < best_val:
            best_val = avg_va

            best_dir = os.path.join(cfg.OUTPUT_DIR, "best")
            os.makedirs(best_dir, exist_ok=True)

            best_save_dir = os.path.join(
                best_dir,
                "finetuned_byol_vit_patchtokens_tokenlevel",
            )

            student.save_pretrained(best_save_dir)
            processor.save_pretrained(best_save_dir)

            torch.save(
                {
                    "proj_s": proj_s.state_dict(),
                    "pred_s": pred_s.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                    "train_loss": avg_tr,
                    "val_loss": avg_va,
                    "best_val_loss": best_val,
                },
                os.path.join(best_dir, "byol_heads.pt"),
            )

            print(f"  ★ New Best! Saved best model -> {best_dir}")

    print("\n[Done] BYOL Finetune Finished.")
    print(f"Best Val Loss = {best_val:.6f}")
    print("\n[Next] Downstream Stage 2 should use:")
    print(
        f"  MODEL_NAME = '{os.path.join(cfg.OUTPUT_DIR, 'best', 'finetuned_byol_vit_patchtokens_tokenlevel')}'"
    )


if __name__ == "__main__":
    main()
