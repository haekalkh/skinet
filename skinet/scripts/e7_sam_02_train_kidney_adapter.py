"""
e7_sam_02_train_kidney_adapter.py

SAM-Med2D + Kidney Adapter fine-tuning on the E7 split (leakage-safe).

    image_encoder (frozen) -> Kidney Adapter (trainable) -> mask_decoder (trainable)
    prompt_encoder (frozen)

Architecture and training recipe follow the previous-project E2B script (BCE+Dice
loss, AdamW, early stopping on val Dice), with two changes:
  - data comes from the E7 manifests (same split as the E7 Mask2Former model), so
    the refiner's train/test boundary matches M2F -> no leakage.
  - the train dataset applies BBOX-JITTER so the box-prompted decoder is robust to
    Mask2Former boxes at inference.

Usage (typically via submit_e7_sam_train.sh):
    python e7_sam_02_train_kidney_adapter.py \
        --train-manifest <E7>/data/sam/e7_clinical_pipeline/gtbox/manifests/train_gtbox.json \
        --val-manifest   <E7>/.../val_gtbox.json \
        --output-dir     <E7>/outputs/e7_clinical_pipeline/sam_kidney_adapter/E7_gtbox_256
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import e7_config as C

# SAM-Med2D repo + adapter (kidney_adapter.py kept alongside this script) + E7 dataset
SAM_REPO = C.OLD_ROOT / "models/SAM-Med2D"
sys.path.insert(0, str(SAM_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e7_sam_dataset import SAMKidneyDatasetE7
from segment_anything import sam_model_registry
from kidney_adapter import build_kidney_adapter

SAM_E7_DIR = C.ROOT / "data/sam/e7_clinical_pipeline/gtbox"
DEFAULT_OUT = C.ROOT / "outputs/e7_clinical_pipeline/sam_kidney_adapter/E7_gtbox_256"
DEFAULT_CKPT = C.OLD_ROOT / "pretrained/sam_med2d_b.pth"


def parse_args():
    p = argparse.ArgumentParser("E7 SAM-Med2D Kidney Adapter training")
    p.add_argument("--train-manifest",
                   default=str(SAM_E7_DIR / "manifests/train_gtbox.json"))
    p.add_argument("--val-manifest",
                   default=str(SAM_E7_DIR / "manifests/val_gtbox.json"))
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--model-type", default="vit_b")
    p.add_argument("--target-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--bbox-jitter", type=float, default=0.10,
                   help="train prompt-box jitter as fraction of side (0.10=+-10%)")
    p.add_argument("--adapter-variant", default="B")
    p.add_argument("--adapter-bottleneck", type=int, default=64)
    p.add_argument("--adapter-dropout", type=float, default=0.1)
    p.add_argument("--roi-crop", action="store_true",
                   help="crop to bbox(+pad) before resize -> higher res for thin structures")
    p.add_argument("--roi-pad", type=float, default=0.30,
                   help="context margin each side, fraction of box side")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


# ----------------------------- loss / metrics --------------------------------

def dice_loss(logits, target, smooth=1.0):
    pred = torch.sigmoid(logits).view(logits.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    return 1.0 - ((2 * inter + smooth) / (union + smooth)).mean()


def combined_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def dice_iou(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float().view(logits.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    dice = ((2 * inter + 1) / (union + 1)).mean().item()
    iou = ((inter + 1) / (union - inter + 1)).mean().item()
    return dice, iou


def forward_batch(model, adapter, batch, device):
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    boxes = batch["bbox"].to(device)
    with torch.no_grad():
        emb = model.image_encoder(images)
        sparse, dense = model.prompt_encoder(points=None, boxes=boxes, masks=None)
    adapted = adapter(emb)
    pred, _ = model.mask_decoder(
        image_embeddings=adapted,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    pred = F.interpolate(pred, size=masks.shape[-2:], mode="bilinear",
                         align_corners=False)
    return pred, masks


def run_epoch(model, adapter, loader, optimizer, device, train: bool):
    if train:
        model.mask_decoder.train(); adapter.train()
        model.image_encoder.eval(); model.prompt_encoder.eval()
    else:
        model.eval(); adapter.eval()

    tot_loss = tot_dice = tot_iou = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            pred, masks = forward_batch(model, adapter, batch, device)
            loss = combined_loss(pred, masks)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            d, i = dice_iou(pred.detach(), masks)
            tot_loss += loss.item(); tot_dice += d; tot_iou += i; n += 1
    return {"loss": tot_loss / n, "dice": tot_dice / n, "iou": tot_iou / n}


class EarlyStopping:
    def __init__(self, patience, min_delta=1e-4):
        self.patience = patience; self.min_delta = min_delta
        self.counter = 0; self.best = 0.0; self.stop = False

    def step(self, val):
        if val > self.best + self.min_delta:
            self.best = val; self.counter = 0
        else:
            self.counter += 1
            self.stop = self.counter >= self.patience
        return self.stop


def main():
    args = parse_args()
    args.image_size = args.target_size
    args.sam_checkpoint = args.checkpoint
    args.encoder_adapter = False   # E2: no built-in SAM adapter; we use our own

    out = Path(args.output_dir)
    ckpt_dir = out / "checkpoints"
    log_dir = out / "logs"
    for d in (ckpt_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_dir / "train.log"), logging.StreamHandler()])
    log = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 64)
    log.info("E7 SAM-Med2D Kidney Adapter training (leakage-safe E7 split)")
    log.info(f"  device      : {device}")
    log.info(f"  target size : {args.target_size}  batch {args.batch_size}")
    log.info(f"  lr {args.lr}  wd {args.weight_decay}  epochs {args.epochs}")
    log.info(f"  bbox jitter : {args.bbox_jitter}")
    log.info(f"  roi_crop    : {args.roi_crop}  (pad {args.roi_pad})")
    log.info(f"  adapter     : variant {args.adapter_variant} "
             f"bottleneck {args.adapter_bottleneck} dropout {args.adapter_dropout}")
    log.info(f"  train mani  : {args.train_manifest}")
    log.info(f"  output      : {out}")

    # datasets
    train_ds = SAMKidneyDatasetE7(args.train_manifest, root=C.ROOT,
                                  target_size=args.target_size, augment=True,
                                  bbox_jitter=args.bbox_jitter,
                                  roi_crop=args.roi_crop, roi_pad=args.roi_pad)
    val_ds = SAMKidneyDatasetE7(args.val_manifest, root=C.ROOT,
                                target_size=args.target_size, augment=False,
                                bbox_jitter=0.0,
                                roi_crop=args.roi_crop, roi_pad=args.roi_pad)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    log.info(f"  train samples {len(train_ds)}  val samples {len(val_ds)}")

    # model
    model = sam_model_registry[args.model_type](args)
    ck = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ck, dict) and "model" in ck:
        ck = ck["model"]
    missing, unexpected = model.load_state_dict(ck, strict=False)
    log.info(f"  loaded {args.checkpoint}  missing {len(missing)} unexpected {len(unexpected)}")
    model = model.to(device)

    for p in model.image_encoder.parameters():
        p.requires_grad = False
    for p in model.prompt_encoder.parameters():
        p.requires_grad = False
    for p in model.mask_decoder.parameters():
        p.requires_grad = True

    adapter = build_kidney_adapter(variant=args.adapter_variant, d_model=256,
                                   bottleneck=args.adapter_bottleneck,
                                   dropout=args.adapter_dropout).to(device)
    n_ad = sum(p.numel() for p in adapter.parameters())
    n_dec = sum(p.numel() for p in model.mask_decoder.parameters() if p.requires_grad)
    log.info(f"  trainable: adapter {n_ad:,} + decoder {n_dec:,} = {n_ad + n_dec:,}")

    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, model.mask_decoder.parameters()))
        + list(adapter.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)

    start_epoch, best_dice, history = 0, 0.0, []
    latest = ckpt_dir / "latest.pth"
    hist_path = out / "history.json"
    if args.resume and latest.exists():
        c = torch.load(latest, map_location=device)
        model.load_state_dict(c["model"]); adapter.load_state_dict(c["adapter"])
        optimizer.load_state_dict(c["optimizer"])
        start_epoch = c["epoch"] + 1
        best_dice = c["metrics"].get("best_dice", 0.0)
        if hist_path.exists():
            history = json.load(open(hist_path))
        log.info(f"  resumed epoch {start_epoch}  best dice {best_dice:.5f}")

    stopper = EarlyStopping(args.patience, args.min_delta)
    stopper.best = best_dice

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr = run_epoch(model, adapter, train_loader, optimizer, device, train=True)
        va = run_epoch(model, adapter, val_loader, optimizer, device, train=False)
        dt = time.time() - t0
        log.info(f"epoch {epoch:3d} | train loss {tr['loss']:.4f} dice {tr['dice']:.4f}"
                 f" | val loss {va['loss']:.4f} dice {va['dice']:.4f} iou {va['iou']:.4f}"
                 f" | {dt:.0f}s")

        rec = {"epoch": epoch, "train": tr, "val": va, "best_dice": max(best_dice, va["dice"])}
        history.append(rec)
        json.dump(history, open(hist_path, "w"), indent=2)

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "adapter": adapter.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": {"best_dice": max(best_dice, va["dice"]), **va}},
                   latest)
        if va["dice"] > best_dice:
            best_dice = va["dice"]
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "adapter": adapter.state_dict(),
                        "metrics": {"best_dice": best_dice, **va}},
                       ckpt_dir / "best_model.pth")
            log.info(f"  ** new best val dice {best_dice:.5f} -> best_model.pth")

        if stopper.step(va["dice"]):
            log.info(f"  early stopping at epoch {epoch} (best dice {best_dice:.5f})")
            break

    log.info(f"Done. Best val dice {best_dice:.5f}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()