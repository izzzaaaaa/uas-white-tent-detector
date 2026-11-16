#!/usr/bin/env python3
"""
train_yolo12.py - train a white-tent detector with Ultralytics YOLOv12.

    pip install -U ultralytics        # needs a version that ships YOLO12 (8.3.78+)
    python train_yolo12.py --data tents_white/data.yaml                 # baseline
    python train_yolo12.py --data tents_white/data.yaml --strong        # if overfitting

Overfitting signs: train box_loss keeps falling while val mAP50-95 flat/falling, or a big
train-vs-val gap. Fix in this order: more real data -> --strong -> smaller model -> --freeze.
"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="yolo12n.pt", help="yolo12n/s/m.pt (n = fastest, best for onboard)")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--strong", action="store_true", help="heavier augmentation + regularisation")
ap.add_argument("--freeze", type=int, default=0, help="freeze first N layers (try 10 for tiny datasets)")
ap.add_argument("--name", default="white_tent")
a = ap.parse_args()

# Top-down imagery: any heading is valid -> flips both ways, big rotation range.
# Target is WHITE: keep hue/saturation jitter small, vary brightness a lot.
aug = dict(
    degrees=180 if a.strong else 90, flipud=0.5, fliplr=0.5,
    hsv_h=0.01, hsv_s=0.3 if a.strong else 0.2, hsv_v=0.5 if a.strong else 0.35,
    scale=0.5,          # keep modest: tent pixel size is known
    translate=0.15, shear=2.0 if a.strong else 0.0,
    perspective=0.0005 if a.strong else 0.0,  # light camera tilt
    mosaic=1.0, close_mosaic=15, mixup=0.1 if a.strong else 0.0,
    erasing=0.3 if a.strong else 0.0,
)
reg = dict(
    weight_decay=0.001 if a.strong else 0.0005,
    patience=40, cos_lr=True, freeze=a.freeze or None,
)

model = YOLO(a.model)
model.train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, name=a.name,
            seed=0, plots=True, **aug, **reg)
print(model.val())  # validation metrics on the held-out (real) val split
