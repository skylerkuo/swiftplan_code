import os
import json
import math
import random
from typing import List, Dict, Any, Tuple, DefaultDict
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup


# ============================================================
# 0. Configuration
# ============================================================

EXPERIMENT_NAME = "PATCH_CE_FUSION_MLP_TEXT_GUIDED"
print(f"★ Running Experiment: {EXPERIMENT_NAME} ★")

JSONL_PATH = "data_gradu.jsonl"
TEST_JSONL_PATH = "/home/skyler/Desktop/isaac_python/data_gradu_test.jsonl"

IMAGE_ROOT = "."
MODEL_NAME = "google/siglip2-base-patch16-512"
# MODEL_NAME  = "/home/skyler/Desktop/isaac_python/byol_siglip2_images_ckpt/best/finetuned_byol_vit_patchtokens_tokenlevel"
EMB_SAVE_PATH = "./offline_embeds/libero_siglip2_embeddings.pt"
TEST_EMB_SAVE_PATH = "./offline_embeds/libero_siglip2_fusion_mlp_test.pt"

OUTPUT_DIR = "./patch_ce_ckpt_fusion_mlp_fixed_split"
SPLIT_PATH = "./stage1_supcon_ckpt/train_val_split.pt"

# test wrong samples output
TEST_WRONG_JSONL_PATH = "./patch_ce_ckpt_fusion_mlp_fixed_split/test_wrong_samples.jsonl"

# offline embedding
BATCH_SIZE_EMB = 16
MAX_TEXT_LEN = 64

# train
BATCH_SIZE_TRAIN = 16
NUM_EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.05
SEED = 40

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# dataloader
NUM_WORKERS = 0
PIN_MEMORY = True

# clip-style logit scale init
INIT_TEMPERATURE = 0.07

# ===== SupCon aux settings (train-pool positives) =====
SUPCON_TEMPERATURE = 0.07
SUPCON_LAMBDA_START = 1.0
SUPCON_LAMBDA_END = 0.1

# 從訓練集中抓同類別 2 個當 positive
SUPCON_K_POS = 2
SUPCON_N_VIEWS = 1 + SUPCON_K_POS

# best tie-break 浮點容忍
TIE_EPS = 1e-8


# ============================================================
# 1. Utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_stage1_split(path: str, n: int) -> Tuple[List[int], List[int], Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 split 檔：{path}\n"
            f"請先跑兩階段的 Stage1 產生 train_val_split.pt"
        )
    split = torch.load(path, map_location="cpu")
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]

    if len(train_idx) + len(val_idx) != n:
        raise ValueError(f"Split size mismatch: {len(train_idx)}+{len(val_idx)} != {n}")
    if len(set(train_idx) & set(val_idx)) != 0:
        raise ValueError("Split has overlap between train and val!")
    if max(train_idx + val_idx) >= n:
        raise ValueError("Split contains index out of range!")
    return train_idx, val_idx, split


def exp_lambda(epoch: int, num_epochs: int, start: float, end: float) -> float:
    if num_epochs <= 1:
        return end
    t = (epoch - 1) / (num_epochs - 1)
    return start * ((end / start) ** t)


def is_better(val1: float, val3: float, best1: float, best3: float, eps: float = 1e-8) -> bool:
    """
    best 選擇規則
    1) Val@1 更高
    2) Val@1 幾乎相同（eps 內）→ Val@3 更高者勝
    """
    if val1 > best1 + eps:
        return True
    if abs(val1 - best1) <= eps and val3 > best3 + eps:
        return True
    return False


# ============================================================
# 2. Offline embeddings (train)
# ============================================================

def build_or_load_embeddings():
    if os.path.exists(EMB_SAVE_PATH):
        print(f"[Info] Loading embeddings from {EMB_SAVE_PATH}")
        return torch.load(EMB_SAVE_PATH, map_location="cpu")

    print("[Info] Calculating Patch embeddings (TRAIN)...")
    records = load_jsonl(JSONL_PATH)

    IMG_KEY = "image_path"
    TEXT_KEY = "instruction"
    ACTION_KEY = "action_label"

    actions = sorted({
        r.get("annotation", {}).get(ACTION_KEY)
        for r in records
        if r.get("annotation", {}).get(ACTION_KEY) is not None
    })
    action2id = {a: i for i, a in enumerate(actions)}
    print(f"[Info] num_actions = {len(actions)}")

    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    all_patch, all_txt, all_labels = [], [], []
    kept, skipped = 0, 0

    for i in tqdm(range(0, len(records), BATCH_SIZE_EMB), desc="Embedding(TRAIN)"):
        batch = records[i:i + BATCH_SIZE_EMB]
        images, texts, labels = [], [], []

        for r in batch:
            img_path = r.get(IMG_KEY)
            if not img_path:
                skipped += 1
                continue
            if not os.path.isabs(img_path):
                img_path = os.path.join(IMAGE_ROOT, img_path)
            if not os.path.exists(img_path):
                skipped += 1
                continue

            action = r.get("annotation", {}).get(ACTION_KEY)
            if action not in action2id:
                skipped += 1
                continue

            images.append(Image.open(img_path).convert("RGB"))
            texts.append(r.get(TEXT_KEY, ""))
            labels.append(action2id[action])

        if len(images) == 0:
            continue

        inputs = processor(
            images=images,
            text=texts,
            padding="max_length",
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)
            patch = out.vision_model_output.last_hidden_state
            txt = out.text_embeds

        all_patch.append(patch.cpu())
        all_txt.append(txt.cpu())
        all_labels.append(torch.tensor(labels, dtype=torch.long))
        kept += len(labels)

    patch_embs = torch.cat(all_patch, dim=0)
    txt_embs = torch.cat(all_txt, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"[Info] kept={kept}, skipped={skipped}")
    print(f"[Info] patch_embs={tuple(patch_embs.shape)} txt_embs={tuple(txt_embs.shape)} labels={tuple(labels.shape)}")

    with torch.no_grad():
        action_inputs = processor(
            text=actions,
            padding="max_length",
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )
        action_inputs = {k: v.to(DEVICE) for k, v in action_inputs.items()}

        if hasattr(model, "get_text_features"):
            action_text_embs = model.get_text_features(**action_inputs).cpu()
        else:
            out2 = model(**action_inputs)
            action_text_embs = out2.text_embeds.cpu()

    data = {
        "patch_embs": patch_embs,
        "txt_embs": txt_embs,
        "labels": labels,
        "action_text_embs": action_text_embs,
        "id2action": actions,
    }

    os.makedirs(os.path.dirname(EMB_SAVE_PATH), exist_ok=True)
    torch.save(data, EMB_SAVE_PATH)
    print(f"[OK] Saved embeddings to {EMB_SAVE_PATH}")
    return data


# ============================================================
# 2.1 Offline embeddings (test) - uses train id2action mapping
# ============================================================

def build_or_load_test_embeddings(id2action: List[str]):
    """
    測試集 embeddings
    - label mapping 必須跟 train 的 id2action 對齊
    - unknown label 會被跳過
    """
    if os.path.exists(TEST_EMB_SAVE_PATH):
        print(f"[Info] Loading TEST embeddings from {TEST_EMB_SAVE_PATH}")
        data = torch.load(TEST_EMB_SAVE_PATH, map_location="cpu")
        if "meta" in data:
            return data
        print("[Warn] Existing TEST cache has no meta field, rebuilding it...")
        try:
            os.remove(TEST_EMB_SAVE_PATH)
        except OSError:
            pass

    print("[Info] Calculating Patch embeddings (TEST)...")
    records = load_jsonl(TEST_JSONL_PATH)

    IMG_KEY = "image_path"
    TEXT_KEY = "instruction"
    ACTION_KEY = "action_label"

    action2id = {a: i for i, a in enumerate(id2action)}
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    all_patch, all_txt, all_labels = [], [], []
    all_meta, skipped_meta = [], []
    kept, skipped, skipped_unknown = 0, 0, 0

    for jsonl_idx, r in enumerate(tqdm(records, desc="Embedding(TEST)")):
        img_path = r.get(IMG_KEY)
        action = r.get("annotation", {}).get(ACTION_KEY)
        text = r.get(TEXT_KEY, "")

        if not img_path:
            skipped += 1
            skipped_meta.append({
                "jsonl_idx": jsonl_idx,
                "reason": "missing_image_path",
                "image_path": None,
                "action": action,
                "instruction": text,
            })
            continue

        if not os.path.isabs(img_path):
            img_path = os.path.join(IMAGE_ROOT, img_path)

        if not os.path.exists(img_path):
            skipped += 1
            skipped_meta.append({
                "jsonl_idx": jsonl_idx,
                "reason": "image_not_found",
                "image_path": img_path,
                "action": action,
                "instruction": text,
            })
            continue

        if action not in action2id:
            skipped_unknown += 1
            skipped_meta.append({
                "jsonl_idx": jsonl_idx,
                "reason": "unknown_label_not_in_train_mapping",
                "image_path": img_path,
                "action": action,
                "instruction": text,
            })
            continue

        image = Image.open(img_path).convert("RGB")
        inputs = processor(
            images=[image],
            text=[text],
            padding="max_length",
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs)
            patch = out.vision_model_output.last_hidden_state
            txt = out.text_embeds

        all_patch.append(patch.cpu())
        all_txt.append(txt.cpu())
        all_labels.append(torch.tensor([action2id[action]], dtype=torch.long))
        all_meta.append({
            "jsonl_idx": jsonl_idx,
            "image_path": img_path,
            "action": action,
            "instruction": text,
        })
        kept += 1

    if kept == 0:
        raise RuntimeError("[Error] TEST kept=0，測試集沒有任何可用樣本（路徑或 label 對不上 train mapping）。")

    patch_embs = torch.cat(all_patch, dim=0)
    txt_embs = torch.cat(all_txt, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"[Info][TEST] kept={kept}, skipped_missing={skipped}, skipped_unknown_label={skipped_unknown}")
    print(f"[Info][TEST] patch_embs={tuple(patch_embs.shape)} txt_embs={tuple(txt_embs.shape)} labels={tuple(labels.shape)}")

    data = {
        "patch_embs": patch_embs,
        "txt_embs": txt_embs,
        "labels": labels,
        "meta": all_meta,
        "skipped_meta": skipped_meta,
    }

    os.makedirs(os.path.dirname(TEST_EMB_SAVE_PATH), exist_ok=True)
    torch.save(data, TEST_EMB_SAVE_PATH)
    print(f"[OK] Saved TEST embeddings to {TEST_EMB_SAVE_PATH}")
    return data


# ============================================================
# 3. Dataset
# ============================================================

class PatchDataset(Dataset):
    def __init__(self, patch_embs, txt_embs, labels, indices: List[int]):
        self.patch_embs = patch_embs
        self.txt_embs = txt_embs
        self.labels = labels.long()
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return {
            "patch_emb": self.patch_embs[idx].float(),
            "txt_emb": self.txt_embs[idx].float(),
            "label": self.labels[idx],
            "sample_idx": idx,
        }


class PatchDatasetWithKPositives(Dataset):
    def __init__(self, patch_embs, txt_embs, labels, indices: List[int], k_pos: int = 3):
        assert k_pos >= 1
        self.patch_embs = patch_embs
        self.txt_embs = txt_embs
        self.labels = labels.long()
        self.indices = list(indices)
        self.k_pos = k_pos

        self.label2pos: DefaultDict[int, List[int]] = defaultdict(list)
        for pos, idx in enumerate(self.indices):
            y = int(self.labels[idx].item())
            self.label2pos[y].append(pos)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx_a = self.indices[i]
        y = int(self.labels[idx_a].item())

        candidates = self.label2pos[y]
        pos_list = []

        if len(candidates) <= 1:
            pos_list = [i] * self.k_pos
        else:
            for _ in range(self.k_pos):
                j = i
                tries = 0
                while j == i and tries < 10:
                    j = random.choice(candidates)
                    tries += 1
                if j == i:
                    j = i
                pos_list.append(j)

        patch_pos, txt_pos = [], []
        for j in pos_list:
            idx_p = self.indices[j]
            patch_pos.append(self.patch_embs[idx_p].float())
            txt_pos.append(self.txt_embs[idx_p].float())

        return {
            "patch_a": self.patch_embs[idx_a].float(),
            "txt_a": self.txt_embs[idx_a].float(),
            "patch_pos": torch.stack(patch_pos, dim=0),
            "txt_pos": torch.stack(txt_pos, dim=0),
            "label": self.labels[idx_a],
        }


# ============================================================
# 4. SupConLoss
# ============================================================

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


# ============================================================
# 5. Model (Fusion MLP)
# ============================================================

class PatchTextualCEModel_FusionMLP(nn.Module):
    def __init__(self, emb_dim, action_text_embs, num_heads=8, dropout=0.1):
        super().__init__()

        self.register_buffer("action_embs", action_text_embs.clone())

        self.cross_attn = nn.MultiheadAttention(
            emb_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm_img = nn.LayerNorm(emb_dim)

        self.img_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.txt_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gate = nn.Sequential(
            nn.LayerNorm(2 * emb_dim),
            nn.Linear(2 * emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Sigmoid(),
        )

        self.fusion_refine = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
        )

        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / INIT_TEMPERATURE)
        )

    def get_logit_scale(self):
        return self.logit_scale.exp().clamp(max=100)

    def encode_feat(self, patch_emb, txt_emb):
        txt_feat = self.txt_proj(txt_emb)

        query = txt_feat.unsqueeze(1)
        attn_out, _ = self.cross_attn(query, patch_emb, patch_emb)
        img_feat = self.norm_img(query + attn_out).squeeze(1)
        img_feat = self.img_proj(img_feat)

        gate_input = torch.cat([img_feat, txt_feat], dim=-1)
        g = self.gate(gate_input)

        fused = g * img_feat + (1 - g) * txt_feat
        # fused = 0.5 * img_feat + 0.5 * txt_feat
        fused = self.fusion_refine(fused)

        return fused

    def forward(self, patch_emb, txt_emb):
        feat = self.encode_feat(patch_emb, txt_emb)

        q = F.normalize(feat, dim=-1)
        a = F.normalize(self.action_embs, dim=-1)

        logits = self.get_logit_scale() * (q @ a.t())
        return logits


# ============================================================
# 6. Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total, top1, top3 = 0, 0, 0
    for batch in dataloader:
        patch = batch["patch_emb"].to(DEVICE)
        txt = batch["txt_emb"].to(DEVICE)
        y = batch["label"].to(DEVICE)

        logits = model(patch, txt)

        top1 += (logits.argmax(dim=-1) == y).sum().item()
        k = min(3, logits.size(-1))
        topk = logits.topk(k, dim=-1).indices
        top3 += (topk == y.unsqueeze(-1)).any(dim=-1).sum().item()
        total += y.size(0)

    return top1 / max(total, 1), top3 / max(total, 1)


@torch.no_grad()
def evaluate_and_dump_wrong_samples(model, dataloader, id2action: List[str], meta: List[Dict[str, Any]], output_jsonl: str):
    """
    將 test 中預測錯誤的樣本全部輸出成 JSONL
    """
    model.eval()
    total, top1, top3 = 0, 0, 0
    wrong_records = []

    for batch in dataloader:
        patch = batch["patch_emb"].to(DEVICE)
        txt = batch["txt_emb"].to(DEVICE)
        y = batch["label"].to(DEVICE)
        sample_idx = batch["sample_idx"]

        logits = model(patch, txt)
        pred = logits.argmax(dim=-1)

        k = min(3, logits.size(-1))
        topk = logits.topk(k, dim=-1).indices

        top1 += (pred == y).sum().item()
        top3 += (topk == y.unsqueeze(-1)).any(dim=-1).sum().item()
        total += y.size(0)

        for i in range(y.size(0)):
            true_id = int(y[i].item())
            pred_id = int(pred[i].item())
            idx = int(sample_idx[i].item()) if torch.is_tensor(sample_idx[i]) else int(sample_idx[i])

            this_meta = meta[idx] if meta is not None and idx < len(meta) else {}

            topk_ids = topk[i].tolist()
            topk_labels = [id2action[j] if 0 <= j < len(id2action) else None for j in topk_ids]

            if pred_id != true_id:
                wrong_records.append({
                    "sample_idx": idx,
                    "jsonl_idx": this_meta.get("jsonl_idx", idx),
                    "image_path": this_meta.get("image_path", None),
                    "instruction": this_meta.get("instruction", None),
                    "true_label_id": true_id,
                    "true_label": id2action[true_id] if 0 <= true_id < len(id2action) else None,
                    "pred_label_id": pred_id,
                    "pred_label": id2action[pred_id] if 0 <= pred_id < len(id2action) else None,
                    "top3_pred_ids": topk_ids,
                    "top3_pred_labels": topk_labels,
                    "top1_correct": False,
                    "top3_correct": true_id in topk_ids,
                })

    acc1 = top1 / max(total, 1)
    acc3 = top3 / max(total, 1)

    save_jsonl(wrong_records, output_jsonl)
    print(f"[TEST] wrong samples saved to: {output_jsonl}")
    print(f"[TEST] wrong sample count: {len(wrong_records)}")

    return acc1, acc3, wrong_records


# ============================================================
# 7. Train + pick best by (Val@1, tie-break Val@3) + Test
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = build_or_load_embeddings()

    patch_embs = data["patch_embs"]
    txt_embs = data["txt_embs"]
    labels = data["labels"]

    n = len(labels)
    emb_dim = patch_embs.shape[-1]
    num_actions = data["action_text_embs"].shape[0]
    print(f"[Info] N={n} emb_dim={emb_dim} num_actions={num_actions}")

    train_idx, val_idx, split_meta = load_stage1_split(SPLIT_PATH, n=n)
    print(f"[Info] Using fixed split from: {SPLIT_PATH}")
    print(f"[Info] split meta: seed={split_meta.get('seed')} val_split={split_meta.get('val_split')}")
    print(f"[Info] train={len(train_idx)} val={len(val_idx)} overlap={len(set(train_idx)&set(val_idx))}")
    print(f"[Info] SupCon views = {SUPCON_N_VIEWS} (anchor + {SUPCON_K_POS} positives from train pool)")

    train_set = PatchDatasetWithKPositives(patch_embs, txt_embs, labels, train_idx, k_pos=SUPCON_K_POS)
    val_set = PatchDataset(patch_embs, txt_embs, labels, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    model = PatchTextualCEModel_FusionMLP(
        emb_dim=emb_dim,
        action_text_embs=data["action_text_embs"].to(DEVICE),
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Info] Total params     : {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"[Info] Trainable params : {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    supcon_criterion = SupConLoss(temperature=SUPCON_TEMPERATURE).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    num_steps = NUM_EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(num_steps * WARMUP_RATIO), num_steps
    )

    best_val1 = -1.0
    best_val3 = -1.0
    best_epoch = -1
    best_path = os.path.join(OUTPUT_DIR, "ce_supcon_aux_trainpoolpos_k3_best.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        lam = exp_lambda(epoch, NUM_EPOCHS, SUPCON_LAMBDA_START, SUPCON_LAMBDA_END)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} (λ_supcon={lam:.3f})")
        for batch in pbar:
            patch_a = batch["patch_a"].to(DEVICE)
            txt_a = batch["txt_a"].to(DEVICE)
            patch_pos = batch["patch_pos"].to(DEVICE)
            txt_pos = batch["txt_pos"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            logits = model(patch_a, txt_a)
            loss_ce = F.cross_entropy(logits, y)

            feat_a = F.normalize(model.encode_feat(patch_a, txt_a), dim=-1)

            B, K, Np, D = patch_pos.shape
            patch_pos_flat = patch_pos.view(B * K, Np, D)
            txt_pos_flat = txt_pos.view(B * K, D)

            feat_pos_flat = F.normalize(model.encode_feat(patch_pos_flat, txt_pos_flat), dim=-1)
            feat_pos = feat_pos_flat.view(B, K, -1)

            feats = torch.cat([feat_a.unsqueeze(1), feat_pos], dim=1)
            loss_supcon = supcon_criterion(feats, labels=y)

            loss = loss_ce + lam * loss_supcon
            # loss = loss_ce

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{loss_ce.item():.4f}",
                supcon=f"{loss_supcon.item():.4f}",
                lam=f"{lam:.3f}",
                scale=f"{model.get_logit_scale().item():.2f}",
            )

        val_top1, val_top3 = evaluate(model, val_loader)

        is_best = is_better(val_top1, val_top3, best_val1, best_val3, eps=TIE_EPS)
        if is_best:
            best_val1 = val_top1
            best_val3 = val_top3
            best_epoch = epoch
            torch.save(
                {
                    "experiment": EXPERIMENT_NAME,
                    "model_state": model.state_dict(),
                    "emb_dim": emb_dim,
                    "num_actions": num_actions,
                    "id2action": data["id2action"],
                    "split_path": SPLIT_PATH,
                    "split_meta": split_meta,
                    "supcon_temperature": SUPCON_TEMPERATURE,
                    "supcon_lambda_start": SUPCON_LAMBDA_START,
                    "supcon_lambda_end": SUPCON_LAMBDA_END,
                    "supcon_lambda_schedule": "exponential_decay",
                    "supcon_trainpool_positives": True,
                    "supcon_k_pos": SUPCON_K_POS,
                    "supcon_n_views": SUPCON_N_VIEWS,
                    "best_rule": "max(Val@1), tie-break max(Val@3)",
                },
                best_path
            )

        tag = " ★ BEST" if is_best else ""
        print(f"\nEpoch {epoch} | λ_supcon={lam:.3f} | Val@1={val_top1:.4f} | Val@3={val_top3:.4f}{tag}")

    print("\n" + "=" * 50)
    print("Training finished. Best validation (Val@1 tie-break Val@3):")
    print(f"Best Epoch : {best_epoch}")
    print(f"Best Val@1 : {best_val1:.4f}")
    print(f"Best Val@3 : {best_val3:.4f}")
    print(f"Saved      : {best_path}")
    print("=" * 50)

    # ========================================================
    # TEST (只在訓練結束後跑一次)
    # ========================================================
    test_data = build_or_load_test_embeddings(id2action=data["id2action"])
    test_patch = test_data["patch_embs"]
    test_txt = test_data["txt_embs"]
    test_labels = test_data["labels"]

    test_idx = list(range(len(test_labels)))
    test_set = PatchDataset(test_patch, test_txt, test_labels, test_idx)
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(DEVICE)

    test_top1, test_top3, wrong_records = evaluate_and_dump_wrong_samples(
        model=model,
        dataloader=test_loader,
        id2action=data["id2action"],
        meta=test_data.get("meta", None),
        output_jsonl=TEST_WRONG_JSONL_PATH,
    )

    print("\n" + "=" * 50)
    print(f"[TEST] Using best epoch={best_epoch} (Val@1={best_val1:.4f}, Val@3={best_val3:.4f})")
    print(f"[TEST] Test@1={test_top1:.4f} | Test@3={test_top3:.4f}")
    print(f"[TEST] Wrong samples dumped: {len(wrong_records)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
