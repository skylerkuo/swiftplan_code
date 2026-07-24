# ============================================================
# BYOL Finetune SigLIP2 Vision Encoder from Images (PATCH-TOKEN LEVEL)
# - Load images directly from a folder (recursively)
# - Two views of the SAME image: clean + augmented
# - Teacher ALWAYS sees CLEAN
# - Student ALWAYS sees AUG
# - Use Vision LAST_HIDDEN_STATE (patch tokens) directly (B, N, D)
# - Token-level BYOL: projector/predictor operate on each token (B, N, *)
# - NO random resized crop (no zoom/crop/resize) in augmentation (PIL aug only)
# - BYOL online/target networks with EMA teacher
# - Save finetuned model in HuggingFace format (every epoch)
# ============================================================

import os
import random
from dataclasses import dataclass
from typing import List, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

from transformers import AutoModel, AutoProcessor


# =========================
# 0) Config
# =========================
@dataclass
class CFG:
    EXPERIMENT_NAME: str = "BYOL_SIGLIP2_PATCHTOKENS_TOKENLEVEL"

    IMAGE_DIR: str = "/home/skyler/Desktop/isaac_python/captured_images"   # ← point to your image folder
    OUTPUT_DIR: str = "./byol_siglip2_images_ckpt"
    MODEL_NAME: str = "google/siglip2-base-patch16-512"

    # image extensions to search for
    IMAGE_EXTS: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    # sampling
    SAMPLES_PER_EPOCH: int = 630
    VAL_SAMPLES: int = 70

    # train
    EPOCHS: int = 10
    BATCH_SIZE: int = 4
    GRAD_ACCUM: int = 8          # effective batch = BATCH_SIZE * GRAD_ACCUM
    NUM_WORKERS: int = 4
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP: bool = True

    # finetune control (vision tower only)
    FINETUNE_VISION: bool = True
    TRAIN_LAST_N_VIT_BLOCKS: int = 2   # 0 = all blocks

    # BYOL MLP sizes (token-level)
    PROJ_HIDDEN: int = 2048
    PROJ_OUT: int = 256

    # optim
    LR_VIT: float = 1e-6
    LR_HEAD: float = 5e-5
    WEIGHT_DECAY: float = 1e-2
    GRAD_CLIP: float = 1.0

    # EMA
    EMA_MOMENTUM: float = 0.996

    # augmentation
    ROT_PROB: float  = 0.3   # probability of applying rotation
    ROT_MAX_DEG: float = 1.0  # max rotation angle in degrees (±)

    # debug
    PRINT_SHAPES_ONCE: bool = True
    PRINT_TRAINABLE_PARAM_SUMMARY: bool = True


cfg = CFG()
print(f"★ Running Experiment: {cfg.EXPERIMENT_NAME} ★")
print(f"[Info] Device = {cfg.DEVICE}")


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_images(root: str, exts: tuple) -> List[str]:
    """Recursively find all image files under root."""
    found = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(exts):
                found.append(os.path.join(dirpath, fname))
    found.sort()
    if not found:
        raise FileNotFoundError(f"No images found under: {root}")
    return found


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================
# Augmentations (PIL) - NO CROP/RESIZE
# =========================
def _to_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L").convert("RGB")


def augment_view(img: Image.Image, rng: np.random.RandomState) -> Image.Image:
    """
    NO random resized crop in this augmentation.
    Augmentations applied:
    - color jitter         (prob 0.6)  brightness / contrast / saturation
    - random grayscale     (prob 0.05)
    - gaussian blur        (prob 0.15)
    - small rotation       (prob ROT_PROB, angle uniformly sampled in ±ROT_MAX_DEG)
      expand=False keeps the canvas size the same; corners are filled with black.
    """
    if rng.rand() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.85, 1.15)))
        img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.85, 1.15)))
        img = ImageEnhance.Color(img).enhance(float(rng.uniform(0.85, 1.15)))

    if rng.rand() < 0.05:
        img = _to_grayscale(img)

    if rng.rand() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.1, 1.0))))

    if rng.rand() < cfg.ROT_PROB:
        angle = float(rng.uniform(-cfg.ROT_MAX_DEG, cfg.ROT_MAX_DEG))
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))

    return img


# =========================
# Dataset: sample 1 image -> (clean, aug)
# =========================
class ImageFolderBYOLDataset(Dataset):
    """
    Replaces VideoFrameBYOLDatasetPyAV.
    Samples images randomly from a flat list with replacement,
    so SAMPLES_PER_EPOCH controls epoch length independently of dataset size.
    """

    def __init__(self, file_list: List[str], samples_per_epoch: int, seed: int = 42):
        self.file_list = file_list
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.RandomState(seed)

        if not self.file_list:
            raise RuntimeError("ImageFolderBYOLDataset: empty file list.")

        print(f"[Dataset] {len(self.file_list)} images available, "
              f"sampling {self.samples_per_epoch} per epoch.")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        # Sample a random image (with replacement)
        path = self.file_list[int(self.rng.randint(0, len(self.file_list)))]

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[Warn] Failed to open {path}: {e}. Retrying with next index.")
            return self.__getitem__(idx + 1)

        v_clean = img
        v_aug   = augment_view(img, self.rng)
        return v_clean, v_aug


def collate_fn(batch):
    v_clean, v_aug = zip(*batch)
    return list(v_clean), list(v_aug)


# =========================
# Encode helper: Vision last_hidden_state TOKENS
# =========================
def encode_vision_last_hidden_tokens(
    model: AutoModel,
    processor: AutoProcessor,
    images: List[Image.Image],
    device: str,
) -> torch.Tensor:
    """
    Returns (B, N, D) patch tokens from vision_model.last_hidden_state.
    """
    inputs = processor(images=images, return_tensors="pt").to(device)

    if not hasattr(model, "vision_model"):
        raise ValueError("Model has no vision_model attribute. "
                         "Make sure you are loading a SigLIP / SigLIP2 model.")

    vision_out = model.vision_model(
        pixel_values=inputs["pixel_values"],
        output_hidden_states=False,
        return_dict=True,
    )
    return vision_out.last_hidden_state  # (B, N, D)


# =========================
# Token-level BYOL MLP (LayerNorm)
# =========================
class TokenBYOLMLP(nn.Module):
    """
    (B, N, D) -> (B, N, out_dim)
    Uses LayerNorm instead of BN to handle token-level inputs.
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
    """Token-level BYOL loss. p, z: (B, N, P)"""
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, m: float):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(m).add_(sp.data, alpha=(1.0 - m))


# =========================
# Finetune scope helper (VISION ONLY)
# =========================
def set_vision_trainable(siglip2: AutoModel, train_last_n_blocks: int):
    for p in siglip2.parameters():
        p.requires_grad = False

    if not hasattr(siglip2, "vision_model"):
        print("[Warn] No vision_model found, unfreezing entire model.")
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
        print("[Warn] Cannot find encoder layers, unfreezing entire vision_model.")
        for p in vm.parameters():
            p.requires_grad = True
        return

    n = len(layers)
    start = 0 if train_last_n_blocks <= 0 else max(0, n - train_last_n_blocks)
    print(f"[Info] Unfreezing vision blocks: {start}..{n-1} (total {n})")

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


@torch.no_grad()
def ema_update_vision_only(teacher: AutoModel, student: AutoModel, m: float):
    if hasattr(teacher, "vision_model") and hasattr(student, "vision_model"):
        ema_update(teacher.vision_model, student.vision_model, m)
    else:
        ema_update(teacher, student, m)


# =========================
# Main
# =========================
def main():
    set_seed(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ---- Discover images ----
    all_images = find_images(cfg.IMAGE_DIR, cfg.IMAGE_EXTS)
    print(f"[Info] Total images found : {len(all_images)}")

    # ---- Train / val split (by file, not by sample) ----
    random.shuffle(all_images)
    split_idx   = int(len(all_images) * 0.9)
    train_files = all_images[:split_idx]
    val_files   = all_images[split_idx:] or train_files   # fallback if tiny dataset

    print(f"[Info] Train images : {len(train_files)} | Val images : {len(val_files)}")

    train_ds = ImageFolderBYOLDataset(train_files, cfg.SAMPLES_PER_EPOCH, seed=cfg.SEED)
    val_ds   = ImageFolderBYOLDataset(val_files,   cfg.VAL_SAMPLES,        seed=cfg.SEED + 1)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ---- Load student model ----
    print(f"[Info] Loading model: {cfg.MODEL_NAME}")
    student   = AutoModel.from_pretrained(cfg.MODEL_NAME).to(cfg.DEVICE)
    processor = AutoProcessor.from_pretrained(cfg.MODEL_NAME)

    if cfg.FINETUNE_VISION:
        set_vision_trainable(student, cfg.TRAIN_LAST_N_VIT_BLOCKS)
    else:
        for p in student.parameters():
            p.requires_grad = False

    if cfg.PRINT_TRAINABLE_PARAM_SUMMARY:
        print(f"[Info] Trainable params (student) = {count_trainable_params(student):,}")

    # ---- Build teacher (EMA copy, frozen) ----
    teacher = AutoModel.from_pretrained(cfg.MODEL_NAME).to(cfg.DEVICE)
    teacher.load_state_dict(student.state_dict(), strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ---- Infer token dim from one batch ----
    with torch.no_grad():
        v_clean_pil, _ = next(iter(train_dl))
        tok = encode_vision_last_hidden_tokens(teacher, processor, v_clean_pil, cfg.DEVICE)
        feat_dim = tok.shape[-1]
        n_tokens = tok.shape[1]
        if cfg.PRINT_SHAPES_ONCE:
            print(f"[Info] Token shape = {tuple(tok.shape)}  => N={n_tokens}, D={feat_dim}")

    # ---- Projector & predictor (token-level) ----
    proj_s = TokenBYOLMLP(feat_dim, cfg.PROJ_HIDDEN, cfg.PROJ_OUT).to(cfg.DEVICE)
    pred_s = TokenBYOLMLP(cfg.PROJ_OUT, cfg.PROJ_HIDDEN, cfg.PROJ_OUT).to(cfg.DEVICE)

    proj_t = TokenBYOLMLP(feat_dim, cfg.PROJ_HIDDEN, cfg.PROJ_OUT).to(cfg.DEVICE)
    proj_t.load_state_dict(proj_s.state_dict(), strict=True)
    proj_t.eval()
    for p in proj_t.parameters():
        p.requires_grad = False

    # ---- Print param summary ----
    total_student  = sum(p.numel() for p in student.parameters())
    trainable_vit  = sum(p.numel() for p in student.parameters() if p.requires_grad)
    proj_s_params  = sum(p.numel() for p in proj_s.parameters())
    pred_s_params  = sum(p.numel() for p in pred_s.parameters())
    total_trainable = trainable_vit + proj_s_params + pred_s_params

    print(f"\n{'='*50}")
    print(f"[Params] SigLIP2 total                       : {total_student:,} ({total_student/1e6:.1f}M)")
    print(f"[Params] ViT trainable (last {cfg.TRAIN_LAST_N_VIT_BLOCKS} blocks)      : {trainable_vit:,} ({trainable_vit/1e6:.1f}M)")
    print(f"[Params] BYOL Projector                      : {proj_s_params:,} ({proj_s_params/1e6:.1f}M)")
    print(f"[Params] BYOL Predictor                      : {pred_s_params:,} ({pred_s_params/1e6:.1f}M)")
    print(f"[Params] Total trainable                     : {total_trainable:,} ({total_trainable/1e6:.1f}M)")
    print(f"{'='*50}\n")

    # ---- Optimizer ----
    vit_params    = get_trainable_vision_params(student)
    optim_groups  = []
    if vit_params:
        optim_groups.append({"params": vit_params, "lr": cfg.LR_VIT})
    optim_groups += [
        {"params": proj_s.parameters(), "lr": cfg.LR_HEAD},
        {"params": pred_s.parameters(), "lr": cfg.LR_HEAD},
    ]
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=cfg.WEIGHT_DECAY)
    scaler    = torch.amp.GradScaler(enabled=(cfg.USE_AMP and cfg.DEVICE == "cuda"))

    best_val = float("inf")

    print("[Info] Start BYOL training (TOKEN-LEVEL, TEACHER=CLEAN ONLY).")
    print(f"       FINETUNE_VISION={cfg.FINETUNE_VISION}, TRAIN_LAST_N_VIT_BLOCKS={cfg.TRAIN_LAST_N_VIT_BLOCKS}")
    print(f"       Rotation aug   : prob={cfg.ROT_PROB}, max_angle=±{cfg.ROT_MAX_DEG}°")
    print("       Teacher target : z_t(clean)")
    print("       Student input  : aug -> proj_s -> pred_s")
    print("       Loss           : BYOL(p(aug), z_t(clean))  (mean over B*N)")

    for epoch in range(1, cfg.EPOCHS + 1):
        student.train()
        proj_s.train()
        pred_s.train()

        optimizer.zero_grad(set_to_none=True)
        tr_loss = 0.0
        steps   = 0

        pbar = tqdm(train_dl, desc=f"Ep {epoch}/{cfg.EPOCHS} [Train]")
        for it, (v_clean_pil, v_aug_pil) in enumerate(pbar):

            # ---- Teacher forward: CLEAN (no grad) ----
            with torch.no_grad():
                t_clean = encode_vision_last_hidden_tokens(teacher, processor, v_clean_pil, cfg.DEVICE)
                z_t     = proj_t(t_clean)   # (B, N, P)

            # ---- Student forward: AUG ----
            with torch.amp.autocast(device_type="cuda", enabled=(cfg.USE_AMP and cfg.DEVICE == "cuda")):
                s_aug = encode_vision_last_hidden_tokens(student, processor, v_aug_pil, cfg.DEVICE)
                z     = proj_s(s_aug)        # (B, N, P)
                p     = pred_s(z)            # (B, N, P)
                loss  = byol_loss_tokens(p, z_t.detach()) / float(cfg.GRAD_ACCUM)

            scaler.scale(loss).backward()

            if (it + 1) % cfg.GRAD_ACCUM == 0:
                if cfg.GRAD_CLIP > 0:
                    scaler.unscale_(optimizer)
                    params_for_clip = vit_params + list(proj_s.parameters()) + list(pred_s.parameters())
                    torch.nn.utils.clip_grad_norm_(params_for_clip, cfg.GRAD_CLIP)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                ema_update_vision_only(teacher, student, cfg.EMA_MOMENTUM)
                ema_update(proj_t, proj_s, cfg.EMA_MOMENTUM)

            tr_loss += float(loss.item()) * float(cfg.GRAD_ACCUM)
            steps   += 1
            pbar.set_postfix({"L": f"{tr_loss/max(steps,1):.3f}"})

        # ---- Validation ----
        student.eval()
        proj_s.eval()
        pred_s.eval()

        va_loss = 0.0
        vsteps  = 0
        with torch.no_grad():
            for v_clean_pil, v_aug_pil in val_dl:
                t_clean = encode_vision_last_hidden_tokens(teacher, processor, v_clean_pil, cfg.DEVICE)
                z_t     = proj_t(t_clean)

                s_aug = encode_vision_last_hidden_tokens(student, processor, v_aug_pil, cfg.DEVICE)
                z     = proj_s(s_aug)
                p     = pred_s(z)

                va_loss += float(byol_loss_tokens(p, z_t).item())
                vsteps  += 1

        avg_tr = tr_loss / max(steps, 1)
        avg_va = va_loss / max(vsteps, 1)
        print(f"\nEpoch {epoch} Result:")
        print(f"  Train Loss = {avg_tr:.6f}")
        print(f"  Val   Loss = {avg_va:.6f}")

        # ---- Save checkpoint every epoch ----
        epoch_dir = os.path.join(cfg.OUTPUT_DIR, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        save_dir  = os.path.join(epoch_dir, "finetuned_byol_vit_patchtokens_tokenlevel")

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
            },
            os.path.join(epoch_dir, "byol_heads.pt"),
        )
        print(f"  [CKPT] Saved -> {epoch_dir}")

        # ---- Save best ----
        if avg_va < best_val:
            best_val = avg_va
            best_dir      = os.path.join(cfg.OUTPUT_DIR, "best")
            best_save_dir = os.path.join(best_dir, "finetuned_byol_vit_patchtokens_tokenlevel")
            os.makedirs(best_dir, exist_ok=True)

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
            print(f"  ★ New Best! Saved -> {best_dir}")

    print("\n[Done] BYOL Finetune Finished.")
    print(f"Best Val Loss = {best_val:.6f}")
    print(f"\n[Next] Load finetuned model with:")
    print(f"  MODEL_NAME = '{os.path.join(cfg.OUTPUT_DIR, 'best', 'finetuned_byol_vit_patchtokens_tokenlevel')}'")


if __name__ == "__main__":
    main()
