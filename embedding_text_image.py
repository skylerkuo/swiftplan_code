import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/save offline embeddings and train/val split only. No training."
    )
    parser.add_argument(
        "--jsonl-path",
        default="data_gradu.jsonl",
        help="Path to the JSONL dataset.",
    )
    parser.add_argument(
        "--image-root",
        default=".",
        help="Root directory used when image_path is relative.",
    )
    parser.add_argument(
        "--model-name",
        # If you want BYOL-SigLIP2 embeddings, change this to your local BYOL checkpoint.
        # default="/home/skyler/Desktop/isaac_python/byol_siglip2_images_ckpt/best/finetuned_byol_vit_patchtokens_tokenlevel",
        default="google/siglip2-base-patch16-512",
        help="HuggingFace model name or local checkpoint path.",
    )
    parser.add_argument(
        "--save-path",
        default="./offline_embeds/libero_siglip2_embeddings.pt",
        help="Output embedding .pt file path.",
    )
    parser.add_argument(
        "--split-save-path",
        default="./stage1_supcon_ckpt/train_val_split.pt",
        help="Output train/val split .pt file path.",
    )
    parser.add_argument("--img-key", default="image_path")
    parser.add_argument("--text-key", default="instruction")
    parser.add_argument("--action-key", default="action_label")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite embedding file if it already exists.",
    )
    parser.add_argument(
        "--overwrite-split",
        action="store_true",
        help="Overwrite split file if it already exists.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_action(record: Dict[str, Any], action_key: str) -> Optional[str]:
    annotation = record.get("annotation", {})
    if isinstance(annotation, dict) and annotation.get(action_key) is not None:
        return str(annotation[action_key])
    if record.get(action_key) is not None:
        return str(record[action_key])
    return None


def get_text(record: Dict[str, Any], text_key: str) -> str:
    value = record.get(text_key, "")
    if value is None:
        return ""
    return str(value)


def resolve_image_path(image_root: str, image_path: str) -> str:
    if os.path.isabs(image_path):
        return image_path
    return os.path.join(image_root, image_path)


def load_rgb_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def validate_embedding_data(data: Dict[str, Any]) -> None:
    required_keys = ["patch_embs", "txt_embs", "labels", "id2action"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Embedding file missing key: {key}")

    n = len(data["labels"])
    if data["patch_embs"].shape[0] != n:
        raise ValueError("patch_embs and labels have different sample counts.")
    if data["txt_embs"].shape[0] != n:
        raise ValueError("txt_embs and labels have different sample counts.")


@torch.no_grad()
def encode_action_texts(
    model: torch.nn.Module,
    processor: Any,
    actions: List[str],
    max_text_len: int,
    device: str,
) -> torch.Tensor:
    action_inputs = processor(
        text=actions,
        padding="max_length",
        max_length=max_text_len,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    if hasattr(model, "get_text_features"):
        action_text_embs = model.get_text_features(**action_inputs)
    else:
        action_text_embs = model(**action_inputs).text_embeds

    return action_text_embs.cpu()


@torch.no_grad()
def build_embeddings(args: argparse.Namespace) -> Dict[str, Any]:
    records = load_jsonl(args.jsonl_path)
    print(f"[Info] Loaded records: {len(records)}")

    actions = sorted(
        {
            action
            for record in records
            if (action := get_action(record, args.action_key)) is not None
        }
    )
    action2id = {action: idx for idx, action in enumerate(actions)}
    print(f"[Info] num_actions = {len(actions)}")

    model = AutoModel.from_pretrained(args.model_name).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_name)
    model.eval()

    all_patch_embs: List[torch.Tensor] = []
    all_txt_embs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    kept_records: List[Dict[str, Any]] = []

    kept = 0
    skipped = 0

    for start in tqdm(range(0, len(records), args.batch_size), desc="Embedding"):
        batch_records = records[start : start + args.batch_size]

        images: List[Image.Image] = []
        texts: List[str] = []
        labels: List[int] = []
        batch_meta: List[Dict[str, Any]] = []

        for global_idx, record in enumerate(batch_records, start=start):
            image_path = record.get(args.img_key)
            action = get_action(record, args.action_key)

            if not image_path or action not in action2id:
                skipped += 1
                continue

            full_image_path = resolve_image_path(args.image_root, str(image_path))
            if not os.path.exists(full_image_path):
                skipped += 1
                continue

            try:
                images.append(load_rgb_image(full_image_path))
            except Exception as exc:
                print(f"[Warn] Failed to load image: {full_image_path} ({exc})")
                skipped += 1
                continue

            text = get_text(record, args.text_key)
            texts.append(text)
            labels.append(action2id[action])
            batch_meta.append(
                {
                    "record_index": global_idx,
                    "image_path": full_image_path,
                    "text": text,
                    "action": action,
                    "label": action2id[action],
                }
            )

        if not images:
            continue

        inputs = processor(
            images=images,
            text=texts,
            padding="max_length",
            max_length=args.max_text_len,
            truncation=True,
            return_tensors="pt",
        ).to(args.device)

        outputs = model(**inputs)
        patch_embs = outputs.vision_model_output.last_hidden_state
        txt_embs = outputs.text_embeds

        all_patch_embs.append(patch_embs.cpu())
        all_txt_embs.append(txt_embs.cpu())
        all_labels.append(torch.tensor(labels, dtype=torch.long))
        kept_records.extend(batch_meta)
        kept += len(labels)

    if kept == 0:
        raise RuntimeError("No valid samples were embedded. Please check paths and JSONL keys.")

    patch_embs = torch.cat(all_patch_embs, dim=0)
    txt_embs = torch.cat(all_txt_embs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    action_text_embs = encode_action_texts(
        model=model,
        processor=processor,
        actions=actions,
        max_text_len=args.max_text_len,
        device=args.device,
    )

    print(f"[Info] kept={kept}, skipped={skipped}")
    print(f"[Info] patch_embs       = {tuple(patch_embs.shape)}")
    print(f"[Info] txt_embs         = {tuple(txt_embs.shape)}")
    print(f"[Info] labels           = {tuple(labels.shape)}")
    print(f"[Info] action_text_embs = {tuple(action_text_embs.shape)}")

    data = {
        "patch_embs": patch_embs,
        "txt_embs": txt_embs,
        "labels": labels,
        "action_text_embs": action_text_embs,
        "id2action": actions,
        "action2id": action2id,
        "records": kept_records,
        "config": {
            "jsonl_path": args.jsonl_path,
            "image_root": args.image_root,
            "model_name": args.model_name,
            "img_key": args.img_key,
            "text_key": args.text_key,
            "action_key": args.action_key,
            "max_text_len": args.max_text_len,
        },
    }
    validate_embedding_data(data)
    return data


def load_or_build_embeddings(args: argparse.Namespace) -> Dict[str, Any]:
    if os.path.exists(args.save_path) and not args.overwrite:
        print(f"[Info] Embedding file exists. Loading: {args.save_path}")
        data = torch.load(args.save_path, map_location="cpu")
        validate_embedding_data(data)
        print(f"[Info] N={len(data['labels'])}")
        print(f"[Info] patch_embs = {tuple(data['patch_embs'].shape)}")
        print(f"[Info] txt_embs   = {tuple(data['txt_embs'].shape)}")
        print(f"[Info] labels     = {tuple(data['labels'].shape)}")
        return data

    data = build_embeddings(args)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(data, args.save_path)
    print(f"[OK] Saved embeddings to: {args.save_path}")
    return data


def make_or_load_split(
    n: int,
    val_split: float,
    seed: int,
    save_path: str,
    overwrite_split: bool = False,
) -> Dict[str, Any]:
    if os.path.exists(save_path) and not overwrite_split:
        split = torch.load(save_path, map_location="cpu")
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]

        if len(train_idx) + len(val_idx) != n:
            raise ValueError(
                f"Split size mismatch: {len(train_idx)} + {len(val_idx)} != {n}. "
                "Use --overwrite-split if this split came from another embedding file."
            )
        if len(set(train_idx) & set(val_idx)) != 0:
            raise ValueError("Split has overlap between train and val.")

        print(f"[Info] Split file exists. Loading: {save_path}")
        print(f"[Info] train={len(train_idx)} val={len(val_idx)} total={n}")
        return split

    val_size = int(n * val_split)
    train_size = n - val_size

    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()

    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    split = {
        "seed": seed,
        "val_split": val_split,
        "n": n,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "split_method": "torch_randperm_global_like_random_split_default",
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(split, save_path)

    print(f"[OK] Saved split to: {save_path}")
    print(f"[Info] seed={seed} val_split={val_split}")
    print(f"[Info] train={len(train_idx)} val={len(val_idx)} total={n}")
    return split


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data = load_or_build_embeddings(args)
    n = len(data["labels"])

    make_or_load_split(
        n=n,
        val_split=args.val_split,
        seed=args.seed,
        save_path=args.split_save_path,
        overwrite_split=args.overwrite_split,
    )

    print("\nDone. You can now run swiftpaln_train.py.")
    print(f"Embedding file: {args.save_path}")
    print(f"Split file    : {args.split_save_path}")


if __name__ == "__main__":
    main()
