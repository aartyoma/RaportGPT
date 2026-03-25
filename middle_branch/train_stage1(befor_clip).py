import torch
import model
import json
import random
import math
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOTS = [
    Path("/kaggle/input/after-clipdota-cvpr/features"),
    Path("/kaggle/input/datasets/artyom004/datasetcctv-part/archive/features"),
]
device = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_PATH = "/kaggle/input/models/artyom004/model3asformer/pytorch/default/1/model_epoch_5 (2).pt"

def padding(xs, ys):
    T_max = max(x.shape[0] for x in xs)
    all_x, all_y, all_masks = [], [], []
    for x, y in zip(xs, ys):
        T = x.shape[0]
        T_pad = T_max - T
        D = x.shape[1]
        X_padded = torch.cat([x, torch.zeros((T_pad, D), dtype=x.dtype, device=x.device)], dim=0)
        y_padded = torch.cat([y, torch.zeros((T_pad,), dtype=torch.long, device=y.device)], dim=0)
        mask = torch.cat([
            torch.ones((T,), dtype=torch.long, device=x.device),
            torch.zeros((T_pad,), dtype=torch.long, device=x.device)
        ], dim=0)
        all_x.append(X_padded)
        all_y.append(y_padded)
        all_masks.append(mask)

    x = torch.stack(all_x)
    y = torch.stack(all_y)
    mask = torch.stack(all_masks)
    return x, y, mask


class ASFormer(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.model_asformer = model.MyTransformer(
            num_decoders=3,
            num_layers=10,
            r1=2,
            r2=2,
            num_f_maps=64,
            input_dim=input_dim,
            num_classes=num_classes,
            channel_masking_rate=0.0,
        )
        self.dropout = nn.Dropout(p=0.3)
    def forward(self, x, mask):
        x = self.dropout(x)
        out = self.model_asformer(x, mask)
        logits = out[-1]
        return logits


@torch.no_grad()
def val_metric(x, y, mask):
    model_ASF.eval()
    logits = model_ASF(x, mask)
    T = min(logits.shape[-1], y.shape[-1])
    logits = logits[..., :T]
    y = y[..., :T]
    mask = mask[..., :T]
    loss_all = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    loss = (loss_all * mask.squeeze(1)).sum() / mask.squeeze(1).sum()
    return loss.item()


def train(x, y, mask, optimizer, accum_steps=8, do_step=False):
    model_ASF.train()
    logits = model_ASF(x, mask)
    T = min(logits.shape[-1], y.shape[-1])
    logits = logits[..., :T]
    y = y[..., :T]
    mask = mask[..., :T]
    loss_all = torch.nn.functional.cross_entropy(logits, y, reduction="none", label_smoothing=0.1)
    loss = (loss_all * mask.squeeze(1)).sum() / mask.squeeze(1).sum()
    (loss / accum_steps).backward()
    if do_step:
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


if __name__ == "__main__":
    logs = []
    train_logs_full = []
    val_logs_full = []
    files = []

    for root in ROOTS:
        files.extend([str(p) for p in root.rglob("*.pt")])
    files = sorted(set(files))
    if len(files) == 0:
        raise ValueError("No .pt files found")
    data0 = torch.load(files[0], map_location="cpu")
    d = data0["x"].shape[1]
    random.seed(42)
    random.shuffle(files)
    train_files, val_files = train_test_split(
        files,
        test_size=0.2,
        random_state=42,
        shuffle=True)
    B = 1
    video_buff = []
    target_buff = []
    epochs = 3
    N_val = len(val_files)
    N_train = len(train_files)
    num_batches = math.ceil(N_train / B)
    num_batches_val = math.ceil(N_val / B)
    model_id = "openai/clip-vit-large-patch14"
    model_ASF = ASFormer(input_dim=d).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model_ASF.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    optimizer = torch.optim.AdamW(model_ASF.parameters(), lr=5e-6, weight_decay=3e-4)
    accum_steps = 16

    for epoch in range(1, epochs + 1):
        update_count = 0
        optimizer.zero_grad()
        step = 0
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        step_val = 0
        epoch_loss_sum_val = 0.0
        epoch_loss_count_val = 0
        for i in range(num_batches):
            batch_file = train_files[i * B:(i + 1) * B]
            cls_per_batch = []
            targets_per_batch = []
            for feat_path in batch_file:
                data = torch.load(feat_path, map_location="cpu")
                x = data["x"].to(device)
                y = data["y"].long().to(device)
                cls_per_batch.append(x)
                targets_per_batch.append(y)
                step += 1

            for x, y in zip(cls_per_batch, targets_per_batch):
                video_buff.append(x)
                target_buff.append(y)
                if len(video_buff) == B:
                    x_padded, y_padded, mask_padded = padding(video_buff, target_buff)
                    x_padded = x_padded.permute(0, 2, 1)
                    mask_padded = mask_padded.unsqueeze(1)
                    update_count += 1
                    do_step = (update_count % accum_steps == 0)
                    loss = train(x_padded, y_padded, mask_padded, optimizer, accum_steps=accum_steps, do_step=do_step)
                    print(f"Train: Epoche:{epoch} Batch:{i + 1}/{num_batches} general step:{step} loss={loss:.4f}")

                    epoch_loss_sum += loss
                    epoch_loss_count += 1
                    train_logs_full.append({
                        "epoch": epoch,
                        "step": step,
                        "batch": i + 1,
                        "loss": loss,
                    })
                    video_buff = []
                    target_buff = []

        if len(video_buff) > 0:
            x_padded, y_padded, mask_padded = padding(video_buff, target_buff)
            x_padded = x_padded.permute(0, 2, 1)
            mask_padded = mask_padded.unsqueeze(1)
            update_count += 1
            do_step = (update_count % accum_steps == 0)
            loss = train(x_padded, y_padded, mask_padded, optimizer, accum_steps=accum_steps, do_step=do_step)
            epoch_loss_sum += loss
            epoch_loss_count += 1
        video_buff = []
        target_buff = []
        if update_count % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        for j in range(num_batches_val):
            batch_file = val_files[j * B:(j + 1) * B]
            cls_per_batch = []
            targets_per_batch = []
            for feat_path in batch_file:
                data = torch.load(feat_path, map_location="cpu")
                x = data["x"].to(device)
                y = data["y"].long().to(device)
                cls_per_batch.append(x)
                targets_per_batch.append(y)
                step_val += 1
            for x, y in zip(cls_per_batch, targets_per_batch):
                video_buff.append(x)
                target_buff.append(y)
                if len(video_buff) == B:
                    x_padded, y_padded, mask_padded = padding(video_buff, target_buff)
                    x_padded = x_padded.permute(0, 2, 1)
                    mask_padded = mask_padded.unsqueeze(1)

                    loss = val_metric(x_padded, y_padded, mask_padded)
                    print(
                        f"Val: Epoche:{epoch} Batch:{j + 1}/{num_batches_val} general step:{step_val} loss={loss:.4f}")
                    epoch_loss_sum_val += loss
                    epoch_loss_count_val += 1
                    val_logs_full.append({
                        "epoch": epoch,
                        "step": step_val,
                        "batch": j + 1,
                        "loss": loss,
                    })
                    video_buff = []
                    target_buff = []
        if len(video_buff) > 0:
            x_padded, y_padded, mask_padded = padding(video_buff, target_buff)
            x_padded = x_padded.permute(0, 2, 1)
            mask_padded = mask_padded.unsqueeze(1)
            loss = val_metric(x_padded, y_padded, mask_padded)
            epoch_loss_sum_val += loss
            epoch_loss_count_val += 1
            video_buff = []
            target_buff = []

        logs.append({
            "epoch": epoch,
            "step": step,
            "train_loss": epoch_loss_sum / max(epoch_loss_count, 1),
            "val_loss": epoch_loss_sum_val / max(epoch_loss_count_val, 1),
        })
        with open("logs.json", "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        with open("val_log.json", "w", encoding="utf-8") as f:
            json.dump(val_logs_full, f, ensure_ascii=False, indent=2)
        with open("train_log.json", "w", encoding="utf-8") as f:
            json.dump(train_logs_full, f, ensure_ascii=False, indent=2)
        torch.save({
            "model": model_ASF.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "clip_model_id": model_id,
            "input_dim": d,
            "accum_steps": accum_steps,
            "lr": 5e-6,
            "weight_decay": 3e-4,
            "fps": 1,
        }, f"model_epoch_{epoch}.pt")





