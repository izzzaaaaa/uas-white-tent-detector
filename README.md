# uas-white-tent-detector

White-tent detector for a UAS search mission (IEEE NEO competition 2026). A YOLOv12n model trained on a public top-down tent dataset that is rescaled to the drone's pixel size, recoloured to white, tiled and extended with synthetic composites. Preprocessing is done with OpenCV and NumPy.

Task

The UAS surveys a defined search boundary, produces a stitched map (.png/.jpeg) and must autonomously detect a white tent placed on the ground.

Setup used
Item	Value
Altitude / speed	25 m / 10 m/s
Camera	Raspberry Pi HQ (IMX477), 4056 x 3040, 6.0 mm lens
Ground sampling distance	about 0.65 cm per pixel
Tent size	5 to 7 sq ft (about 105 to 125 px per side in a full frame)
Model	YOLOv12n, 1 class (white_tent), 640 px tiles
Results (validation split, 33 tiles / 61 tents)
Precision	Recall	mAP50	mAP50-95	Inference (T4 GPU)
0.934	0.984	0.983	0.697	about 15 ms per tile
