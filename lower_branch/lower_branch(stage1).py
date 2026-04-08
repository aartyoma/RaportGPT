
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch


import random
import json
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from transformers import Blip2Model, AutoTokenizer, AutoModelForCausalLM

model_name = "Salesforce/blip2-opt-2.7b"
llm_name = "Qwen/Qwen2.5-7B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

blip_model = Blip2Model.from_pretrained(
    model_name,
    torch_dtype=torch.float16
).to(device)
for p in blip_model.parameters():
    p.requires_grad = False
q_hidden = blip_model.config.qformer_config.hidden_size
qformer_encoder_dim = blip_model.config.qformer_config.encoder_hidden_size

tokenizer = AutoTokenizer.from_pretrained(llm_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

qwen = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.float16
).to(device)

qwen.config.use_cache = False
qwen.gradient_checkpointing_enable()

llm_dim = qwen.config.hidden_size

BASE = Path("/workspace/final5k")
ANNO = BASE / "annotations.jsonl"
FEAT = BASE / "features"

Path("/workspace/split").mkdir(parents=True, exist_ok=True)


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

STEP_LOG = LOG_DIR / "train_steps.jsonl"
EPOCH_LOG = LOG_DIR / "train_epochs.jsonl"
CKPT_DIR = LOG_DIR / "stage1_ckpts"
CKPT_DIR.mkdir(exist_ok=True)


LOG_EVERY = 20


TRAIN_PROMPT = """
Write a short factual draft of the observed traffic incident in plain English.

Requirements:
- Write one short paragraph of 3 to 6 sentences.
- Describe only what is clearly visible in the video.
- Mention the road scene only if it is visually evident.
- Describe the road users and their actions in chronological order.
- Describe the critical moment and the immediate aftermath.
- Do not invent hidden causes, intent, fault, exact speed, injuries, license plates, or legal conclusions.
- Keep the wording neutral, concrete, and report-oriented.
- Output only the paragraph.
""".strip()

PROMPT_TOK = tokenizer(TRAIN_PROMPT, return_tensors="pt", truncation=True)
PROMPT_IDS = PROMPT_TOK["input_ids"]
PROMPT_MASK = PROMPT_TOK["attention_mask"]


names = sorted([p.name for p in FEAT.glob("*.pt")])

random.seed(42)
random.shuffle(names)

len_names = len(names)
train_names = names[0:int(0.95 * len_names)]
val_names = names[int(0.95 * len_names):]


with open("/workspace/split/train_names.json", "w", encoding="utf-8") as f:
    json.dump(train_names, f, ensure_ascii=False, indent=4)

with open("/workspace/split/val_names.json", "w", encoding="utf-8") as f:
    json.dump(val_names, f, ensure_ascii=False, indent=4)


train_loader = DataLoader(train_names, batch_size=6, shuffle=True, num_workers=4, pin_memory=True, collate_fn=lambda x: x)
val_loader = DataLoader(val_names, batch_size=6, shuffle=False, num_workers=4, pin_memory=True, collate_fn=lambda x: x)


class RaportGPT(nn.Module):
    def __init__(self, d, hidden_dim, llm_dim):
        super().__init__()
        self.fc = nn.Linear(4 * d, d)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        self.toqformer_proj = nn.Linear(d, qformer_encoder_dim)
        self.qformer = blip_model.qformer
        self.query_tokens = nn.Parameter(blip_model.query_tokens.detach().clone())
        self.tollm_proj = nn.Linear(q_hidden, llm_dim)
        self.tokenizer = tokenizer
        self.qwen = qwen

    def forward(self, x, prompt_ids, prompt_mask, target_ids, target_mask, time_mask):
        B, T, N, D = x.shape
        x = x.reshape(B, T, N // 4, 4 * D)
        x = self.fc(x)
        x = x.permute(0, 2, 1, 3)  # [B, 64, T, 1024]
        x = x.reshape(B * (N // 4), T, D)  # [B*64, T, 1024]
        tf_mask = (time_mask == 0)
        tf_mask = tf_mask.unsqueeze(1).expand(B, N // 4, T).reshape(B * (N // 4), T)
        x = self.transformer(x, src_key_padding_mask=tf_mask)
        x = x.reshape(B, N // 4, T, D).permute(0, 2, 1, 3)  # [B, T, 64, D]
        x = self.toqformer_proj(x)  # [B, T, 64, q_hidden]
        x = x.to(self.query_tokens.dtype)

        qformer_mask = time_mask.unsqueeze(-1).expand(B, T, N // 4).reshape(B, T * (N // 4))
        x = x.reshape(B, T * (N // 4), qformer_encoder_dim)  # [B, T*64, qformer_encoder_dim]

        query_tokens = self.query_tokens.expand(B, -1, -1)
        qformer_out = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=x,
            encoder_attention_mask=qformer_mask,
            return_dict=True
        )
        qformer_out = qformer_out.last_hidden_state
        proj = self.tollm_proj(qformer_out)
        proj = proj.to(self.qwen.dtype)

        prompt_embeds = self.qwen.get_input_embeddings()(prompt_ids)
        target_embeds = self.qwen.get_input_embeddings()(target_ids)
        input_embeds = torch.cat([proj, prompt_embeds, target_embeds], dim=1)
        visual_mask = torch.ones(B, proj.shape[1], dtype=prompt_mask.dtype, device=device)
        attention_mask = torch.cat([visual_mask, prompt_mask, target_mask], dim=1)
        ignore_visual = torch.full((B, proj.shape[1]), -100, dtype=torch.long, device=device)
        ignore_prompt = torch.full_like(prompt_ids, -100)
        target_labels = target_ids.masked_fill(target_mask == 0, -100)

        labels = torch.cat([ignore_visual, ignore_prompt, target_labels], dim=1)
        out = self.qwen(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return out


def train_step(model, x_batch, ys, time_mask, optimizer, scaler):
    model.train()
    model.qformer.eval()
    model.qwen.eval()
    optimizer.zero_grad(set_to_none=True)

    B = len(ys)
    lmb= 0.5
    margin= 0.3
    x_wrong = torch.cat([x_batch[1:], x_batch[:1]], dim=0)
    time_mask_wrong = torch.cat([time_mask[1:], time_mask[:1]], dim=0)

    prompt_ids = PROMPT_IDS.expand(B, -1).to(device, non_blocking=True)
    prompt_mask = PROMPT_MASK.expand(B, -1).to(device, non_blocking=True)

    ys = [y.strip() + tokenizer.eos_token for y in ys]

    target_tok = tokenizer(ys, return_tensors="pt", padding=True, truncation=True)
    target_ids = target_tok["input_ids"].to(device, non_blocking=True)
    target_mask = target_tok["attention_mask"].to(device, non_blocking=True)

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        out_real = model(x_batch, prompt_ids, prompt_mask, target_ids, target_mask, time_mask)
        loss_real = out_real.loss


    if B > 1:
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                out_wrong = model(x_wrong, prompt_ids, prompt_mask, target_ids, target_mask, time_mask_wrong)
                loss_wrong = out_wrong.loss.detach()

        contrast_loss = torch.relu(margin - (loss_wrong - loss_real))
        loss = loss_real + lmb * contrast_loss
    else:
        loss = loss_real

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), grad_norm

@torch.no_grad()
def val_step(model, x_batch, ys, time_mask):
    model.eval()
    model.qformer.eval()
    model.qwen.eval()

    B = len(ys)

    prompt_ids = PROMPT_IDS.expand(B, -1).to(device, non_blocking=True)
    prompt_mask = PROMPT_MASK.expand(B, -1).to(device, non_blocking=True)

    ys = [y.strip() + tokenizer.eos_token for y in ys]

    target_tok = tokenizer(ys, return_tensors="pt", padding=True, truncation=True)
    target_ids = target_tok["input_ids"].to(device, non_blocking=True)
    target_mask = target_tok["attention_mask"].to(device, non_blocking=True)

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        out = model(x_batch, prompt_ids, prompt_mask, target_ids, target_mask, time_mask)
    return out.loss.item()

def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_jsonl(path):
    by_id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_id[row["id"]] = row["target"]

    return by_id

if __name__ == "__main__":
    model = RaportGPT(d=1024, hidden_dim=4096, llm_dim=llm_dim).to(device)
    for p in model.qformer.parameters():
        p.requires_grad = False
    model.query_tokens.requires_grad = False

    for p in model.qwen.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    epoches = 5
    global_step = 0

    anno_by_id = read_jsonl(ANNO)


    best_val = float("inf")
    for epoch in range(epoches):
        train_losses = []
        val_losses = []
        for batch in train_loader:
            xs = []
            ys = []
            for name in batch:
                stem = Path(name).stem
                x = torch.load(FEAT / name, map_location="cpu", weights_only=True).half()
                y = anno_by_id[stem]
                xs.append(x)
                ys.append(y)

            max_T = max(x.shape[0] for x in xs)
            N = xs[0].shape[1]
            D = xs[0].shape[2]
            B = len(xs)

            x_batch = torch.zeros(B, max_T, N, D, dtype=torch.float32)
            time_mask = torch.zeros(B, max_T, dtype=torch.long)

            for i, x in enumerate(xs):
                t = x.shape[0]
                x_batch[i, :t] = x
                time_mask[i, :t] = 1

            x_batch = x_batch.to(device, non_blocking=True)
            time_mask = time_mask.to(device, non_blocking=True)

            try:
                loss, grad_norm = train_step(model, x_batch, ys, time_mask, optimizer, scaler)
            except torch.OutOfMemoryError:
                print("OOM on batch, skipping")
                optimizer.zero_grad(set_to_none=True)
                del xs, ys, x_batch, time_mask
                torch.cuda.empty_cache()
                continue
            del xs, ys, x_batch, time_mask
            torch.cuda.empty_cache()
            global_step += 1
            train_losses.append(loss)
            if global_step % LOG_EVERY == 0:
                append_jsonl(STEP_LOG, {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "grad_norm": float(grad_norm),
                    "step_in_epoch": len(train_losses),
                    "train_loss": float(loss),
                    "lr": float(optimizer.param_groups[0]["lr"])
                })
                print(f"epoch={epoch + 1} step={global_step} loss={loss:.4f} grad={grad_norm:.2f} lr={optimizer.param_groups[0]['lr']:.2e}")

        for batch in val_loader:
            xs = []
            ys = []
            for name in batch:
                stem = Path(name).stem
                x = torch.load(FEAT / name, map_location="cpu", weights_only=True).half()
                y = anno_by_id[stem]
                xs.append(x)
                ys.append(y)

            max_T = max(x.shape[0] for x in xs)
            N = xs[0].shape[1]
            D = xs[0].shape[2]
            B = len(xs)

            x_batch = torch.zeros(B, max_T, N, D, dtype=torch.float32)
            time_mask = torch.zeros(B, max_T, dtype=torch.long)

            for i, x in enumerate(xs):
                t = x.shape[0]
                x_batch[i, :t] = x
                time_mask[i, :t] = 1

            x_batch = x_batch.to(device, non_blocking=True)
            time_mask = time_mask.to(device, non_blocking=True)

            try:
                loss = val_step(model, x_batch, ys, time_mask)

            except torch.OutOfMemoryError:
                print("OOM on batch, skipping")
                del xs, ys, x_batch, time_mask
                torch.cuda.empty_cache()
                continue

            del xs, ys, x_batch, time_mask
            torch.cuda.empty_cache()
            val_losses.append(loss)

        train_loss = sum(train_losses) / max(1,len(train_losses))
        val_loss = sum(val_losses) / max(1,len(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch + 1,
                "fc": model.fc.state_dict(),
                "transformer": model.transformer.state_dict(),
                "toqformer_proj": model.toqformer_proj.state_dict(),
                "tollm_proj": model.tollm_proj.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val
            }, LOG_DIR / "best_stage1.pt")


        append_jsonl(EPOCH_LOG, {
            "epoch": epoch + 1,
            "train_loss_mean": float(train_loss),
            "val_loss_mean": float(val_loss),
            "best_val_loss": float(best_val)
        })

        print(f"epoch={epoch + 1}/{epoches} train={train_loss:.4f} val={val_loss:.4f} best={best_val:.4f}")
        torch.save({
            "epoch": epoch + 1,
            "fc": model.fc.state_dict(),
            "transformer": model.transformer.state_dict(),
            "toqformer_proj": model.toqformer_proj.state_dict(),
            "tollm_proj": model.tollm_proj.state_dict(),
            "optimizer": optimizer.state_dict()
        }, CKPT_DIR / f"stage1_epoch_{epoch + 1}.pt")
































