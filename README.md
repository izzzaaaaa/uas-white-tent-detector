# uas-white-tent-detector

White-tent detector for a UAS search mission (IEEE NEO competition 2026). A YOLOv12n model trained on a public top-down tent dataset that is rescaled to the drone's pixel size, recoloured to white, tiled and extended with synthetic composites. Preprocessing is done with OpenCV and NumPy.

## Task
The UAS surveys a defined search boundary, produces a stitched map (.png/.jpeg) and must autonomously detect a white tent placed on the ground.

## Link for Drive
https://drive.google.com/drive/folders/1RHdzsGHJVl1AwRadlNm-ASCLOhbTy-kE?usp=drive_link

## Setup and Results

### Flight and Camera Setup

| Item | Value |
|------|-------|
| **Altitude / Speed** | 25 m / 10 m/s |
| **Camera** | Raspberry Pi HQ Camera (IMX477) |
| **Camera Resolution** | 4056 × 3040 |
| **Lens** | 6.0 mm |
| **Ground Sampling Distance (GSD)** | ~0.65 cm/pixel |
| **Tent Size** | 5–7 sq ft |
| **Tent Size in Full Frame** | ~105–125 pixels per side |

### Model Configuration

| Item | Value |
|------|-------|
| **Model** | YOLOv12n |
| **Classes** | 1 (`white_tent`) |
| **Input** | 640 × 640 pixel tiles |

### Validation Results

Validation split: **33 tiles containing 61 tents**

| Metric | Result |
|--------|--------|
| **Precision** | 0.934 |
| **Recall** | 0.984 |
| **mAP@50** | 0.983 |
| **mAP@50–95** | 0.697 |
| **Inference Time** | ~15 ms per tile |
| **Hardware** | NVIDIA T4 GPU |
