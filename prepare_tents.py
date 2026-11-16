#!/usr/bin/env python3
"""
prepare_tents.py - turn a YOLO-format tent dataset into a white-tent, drone-scale dataset.

Stages (all OpenCV / NumPy):
  1. load + clean   : drop unreadable / tiny / duplicate / (heuristic) ground-level images
  2. rescale        : resize each image so tents have TARGET pixel size (from GSD or --target-px)
  3. whiten         : GrabCut (or polygon) mask per tent -> HSV recolour to white; saves cut-out patches
  4. tile           : overlapping tiles, partially-cut tents handled, tent-free tiles kept as negatives
  5. synthesize     : paste white-tent patches onto real aerial backgrounds (shadow, rotation, degrade)
  6. write          : images/labels train+val, data.yaml  (split by SOURCE image -> no leakage)

Everything is merged into ONE class: white_tent.
"""
import argparse, random, math
from pathlib import Path
import cv2, numpy as np, yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------------- io helpers
def label_for(p: Path) -> Path:
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def read_yolo(lp: Path, w: int, h: int):
    """Boxes (5 values) or polygons/OBB (>=6 values). Returns px-space objects, class ignored."""
    objs = []
    if not lp.exists():
        return objs
    for line in lp.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        v = list(map(float, p[1:]))
        if len(v) == 4:
            cx, cy, bw, bh = v
            box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
            poly = None
        else:
            if len(v) % 2:
                continue
            pts = np.array(v, np.float32).reshape(-1, 2) * [w, h]
            box = [*pts.min(0), *pts.max(0)]
            poly = pts
        objs.append({"box": [float(b) for b in box], "poly": poly})
    return objs


def write_yolo(lp: Path, boxes, w, h):
    lines = []
    for x1, y1, x2, y2 in boxes:
        lines.append(f"0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
    lp.write_text("\n".join(lines))


# --------------------------------------------------------------- 1. cleaning
def dhash(img, s=8):
    g = cv2.cvtColor(cv2.resize(img, (s + 1, s), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return (g[:, 1:] > g[:, :-1]).tobytes()


def looks_ground_level(img):
    """Cheap heuristic: smooth, bright, bluish strip along the top = sky. REVIEW rejected.txt!"""
    h = img.shape[0]
    top = img[: max(8, h // 7)]
    b, g, r = [float(top[..., i].mean()) for i in range(3)]
    return b > r + 15 and (b + g + r) / 3 > 140 and float(cv2.cvtColor(top, cv2.COLOR_BGR2GRAY).std()) < 28


# ------------------------------------------------------- 2. target pixel size
def target_tent_px(a):
    if a.target_px:
        return a.target_px
    gsd = a.altitude * a.sensor_width_mm / (a.focal_mm * a.image_width_px)  # metres / pixel
    return a.tent_m / gsd


def rescale(img, objs, target):
    sizes = [max(o["box"][2] - o["box"][0], o["box"][3] - o["box"][1]) for o in objs]
    if not sizes:
        return img, objs
    s = float(np.clip(target / max(np.median(sizes), 1.0), 0.05, 4.0))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=interp)
    for o in objs:
        o["box"] = [v * s for v in o["box"]]
        if o["poly"] is not None:
            o["poly"] = o["poly"] * s
    return img, objs


# ------------------------------------------------------------ 3. white tents
def tent_mask(img, obj):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if obj["poly"] is not None and len(obj["poly"]) >= 3:
        cv2.fillPoly(mask, [obj["poly"].astype(np.int32)], 255)
        return mask
    x1, y1, x2, y2 = [int(round(v)) for v in obj["box"]]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return mask
    mask[y1:y2, x1:x2] = 255  # fallback = whole box
    if bw < 10 or bh < 10:
        return mask
    m = max(4, int(0.25 * max(bw, bh)))
    X1, Y1, X2, Y2 = max(0, x1 - m), max(0, y1 - m), min(w, x2 + m), min(h, y2 + m)
    crop = img[Y1:Y2, X1:X2]
    gm = np.zeros(crop.shape[:2], np.uint8)
    try:
        cv2.grabCut(crop, gm, (x1 - X1, y1 - Y1, bw, bh), np.zeros((1, 65)), np.zeros((1, 65)), 3,
                    cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return mask
    fg = ((gm == 1) | (gm == 3)).astype(np.uint8) * 255
    fg_box = fg[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1]
    ratio = (fg_box > 0).sum() / float(bw * bh)
    if not 0.30 < ratio < 0.98:  # GrabCut failed -> keep the box
        return mask
    out = np.zeros((h, w), np.uint8)
    out[y1:y2, x1:x2] = fg_box
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return out


def whiten(img, mask):
    m = mask > 0
    if not m.any():
        return img
    soft = cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    p90 = max(float(np.percentile(v[m], 90)), 1.0)
    gain = min(235.0 / p90, 6.0)
    shade = np.clip(v * gain / 255.0, 0, 1)      # keeps folds / shading
    v2 = 170.0 + 85.0 * shade                   # always bright
    s2 = s * 0.04                               # remove colour
    s = s * (1 - soft) + s2 * soft
    v = v * (1 - soft) + v2 * soft
    return cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2BGR)


def make_patch(img, mask, box, pad=4):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    a = mask[y1:y2, x1:x2]
    if (a > 0).sum() < 12:
        return None
    return np.dstack([img[y1:y2, x1:x2], a])  # BGRA


# ----------------------------------------------------------------- 4. tiling
def tile_image(img, objs, size, overlap, min_vis=0.5):
    H, W = img.shape[:2]
    if H < size or W < size:  # pad (constant, NOT reflect -> no duplicated tents)
        img = cv2.copyMakeBorder(img, 0, max(0, size - H), 0, max(0, size - W),
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        H, W = img.shape[:2]
    step = max(1, int(size * (1 - overlap)))
    xs = list(range(0, W - size + 1, step))
    ys = list(range(0, H - size + 1, step))
    if xs[-1] + size < W: xs.append(W - size)
    if ys[-1] + size < H: ys.append(H - size)
    pos, neg = [], []
    for y0 in ys:
        for x0 in xs:
            kept, bad = [], False
            for o in objs:
                bx1, by1, bx2, by2 = o["box"]
                cx1, cy1 = max(bx1, x0), max(by1, y0)
                cx2, cy2 = min(bx2, x0 + size), min(by2, y0 + size)
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                vis = (cx2 - cx1) * (cy2 - cy1) / max((bx2 - bx1) * (by2 - by1), 1e-6)
                if vis >= min_vis and (cx2 - cx1) >= 3 and (cy2 - cy1) >= 3:
                    kept.append([cx1 - x0, cy1 - y0, cx2 - x0, cy2 - y0])
                elif vis > 0.05:
                    bad = True  # half-cut tent we did not label -> would teach wrong negatives
            if bad:
                continue
            t = img[y0:y0 + size, x0:x0 + size]
            (pos if kept else neg).append((t, kept))
    return pos, neg


# ------------------------------------------------------------- 5. synthesis
def degrade(img, rng):
    """Make the whole image look like one camera: blur, noise, JPEG, exposure."""
    if rng.random() < 0.6:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.3:  # motion blur
        k = rng.choice([3, 5, 7])
        ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        img = cv2.filter2D(img, -1, cv2.warpAffine(ker, M, (k, k)))
    if rng.random() < 0.6:
        n = np.random.normal(0, rng.uniform(2, 8), img.shape)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.8, 1.15), beta=rng.uniform(-15, 15))
    if rng.random() < 0.6:
        q = rng.randint(55, 90)
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
    return img


def rot_rgba(p, angle):
    h, w = p.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c) + 1, int(h * c + w * s) + 1
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(p, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def synth_image(bgs, patches, size, target, rng):
    bg = cv2.imread(str(rng.choice(bgs)))
    if bg is None:
        return None
    if min(bg.shape[:2]) < size:
        f = size / min(bg.shape[:2])
        bg = cv2.resize(bg, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    y0 = rng.randint(0, bg.shape[0] - size); x0 = rng.randint(0, bg.shape[1] - size)
    canvas = bg[y0:y0 + size, x0:x0 + size].copy()
    sun = rng.uniform(0, 2 * math.pi)
    boxes, n = [], rng.choices([0, 1, 2], weights=[1, 4, 1])[0]
    for _ in range(n):
        p = rng.choice(patches).copy()
        if rng.random() < 0.5:
            p = p[:, ::-1]
        p = np.ascontiguousarray(p)
        sc = target * rng.uniform(0.7, 1.4) / max(p.shape[:2])
        p = cv2.resize(p, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
        p = rot_rgba(p, rng.uniform(0, 360))
        ph, pw = p.shape[:2]
        if ph >= size - 4 or pw >= size - 4:
            continue
        x = rng.randint(0, size - pw); y = rng.randint(0, size - ph)
        a = cv2.GaussianBlur(p[:, :, 3], (3, 3), 0).astype(np.float32) / 255.0
        ys, xs = np.where(p[:, :, 3] > 20)
        if len(xs) < 10:
            continue
        bx = [x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1]
        if any(not (bx[2] <= b[0] or bx[0] >= b[2] or bx[3] <= b[1] or bx[1] >= b[3]) for b in boxes):
            continue
        # shadow: shifted, blurred, darker copy of the mask
        d = max(2, int(0.12 * max(pw, ph)))
        dx, dy = int(d * math.cos(sun)), int(d * math.sin(sun))
        sh = np.zeros((size, size), np.float32)
        sh[y:y + ph, x:x + pw] = a
        sh = cv2.GaussianBlur(np.roll(sh, (dy, dx), (0, 1)), (0, 0), max(1.0, d / 3))
        canvas = (canvas * (1 - 0.35 * sh[..., None])).astype(np.uint8)
        # alpha-blend (not Poisson: seamlessClone tints a white tent towards the ground colour)
        roi = canvas[y:y + ph, x:x + pw].astype(np.float32)
        tint = p[:, :, :3].astype(np.float32) * rng.uniform(0.9, 1.05)
        canvas[y:y + ph, x:x + pw] = np.clip(roi * (1 - a[..., None]) + tint * a[..., None], 0, 255).astype(np.uint8)
        boxes.append([max(0, bx[0]), max(0, bx[1]), min(size, bx[2]), min(size, bx[3])])
    return degrade(canvas, rng), boxes


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with YOLO-format export (train/valid/test ok)")
    ap.add_argument("--backgrounds", help="folder of real aerial images WITHOUT tents (WiSARD, HERIDAL ...)")
    ap.add_argument("--out", default="tents_white")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    # scale: either --target-px or camera geometry
    ap.add_argument("--target-px", type=float, default=0, help="tent size (longest side) in px at your camera")
    ap.add_argument("--altitude", type=float, default=30, help="m")
    ap.add_argument("--tent-m", type=float, default=3.0, help="tent longest side in m")
    ap.add_argument("--sensor-width-mm", type=float, default=6.3)
    ap.add_argument("--focal-mm", type=float, default=4.5)
    ap.add_argument("--image-width-px", type=float, default=3840)
    ap.add_argument("--white-frac", type=float, default=1.0, help="share of tents recoloured white")
    ap.add_argument("--neg-ratio", type=float, default=0.5, help="negative tiles per positive tile")
    ap.add_argument("--n-synth", type=int, default=500)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--no-ground-filter", action="store_true")
    ap.add_argument("--blacklist", help="txt file of image filenames to drop")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed); np.random.seed(a.seed)
    out = Path(a.out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    black = set(Path(a.blacklist).read_text().split()) if a.blacklist else set()
    target = target_tent_px(a)
    print(f"[info] target tent size = {target:.1f}px")

    # 1. load + clean --------------------------------------------------------
    items, seen, rejected = [], set(), []
    for p in sorted(Path(a.src).rglob("*")):
        if p.suffix.lower() not in IMG_EXT or p.name in black:
            continue
        img = cv2.imread(str(p))
        if img is None or min(img.shape[:2]) < 64:
            rejected.append((p.name, "unreadable/tiny")); continue
        hsh = dhash(img)
        if hsh in seen:
            rejected.append((p.name, "duplicate")); continue
        seen.add(hsh)
        if not a.no_ground_filter and looks_ground_level(img):
            rejected.append((p.name, "ground-level?")); continue
        objs = read_yolo(label_for(p), img.shape[1], img.shape[0])
        items.append((p, img, objs))
    (out / "rejected.txt").write_text("\n".join(f"{n}\t{r}" for n, r in rejected))
    print(f"[clean] kept {len(items)}  rejected {len(rejected)} (see rejected.txt)")

    # split by SOURCE image
    rng.shuffle(items)
    nval = max(1, int(len(items) * a.val_frac))

    stats = {"train": [0, 0], "val": [0, 0]}  # [pos tiles, neg tiles]
    patches, k = [], 0
    for i, (p, img, objs) in enumerate(items):
        sp = "val" if i < nval else "train"
        img, objs = rescale(img, objs, target)                                  # 2
        for o in objs:                                                          # 3
            if rng.random() <= a.white_frac:
                m = tent_mask(img, o)
                img = whiten(img, m)
                if sp == "train":
                    pt = make_patch(img, m, o["box"])
                    if pt is not None:
                        patches.append(pt)
        pos, neg = tile_image(img, objs, a.tile, a.overlap)                     # 4
        rng.shuffle(neg)
        neg = neg[: int(a.neg_ratio * max(len(pos), 1))]
        for t, boxes in pos + neg:
            name = f"{p.stem}_{k:06d}"; k += 1
            cv2.imwrite(str(out / "images" / sp / f"{name}.jpg"), t, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_yolo(out / "labels" / sp / f"{name}.txt", boxes, a.tile, a.tile)
        stats[sp][0] += len(pos); stats[sp][1] += len(neg)
    print(f"[tile] train pos/neg {stats['train']}  val pos/neg {stats['val']}  patches {len(patches)}")

    # 5. synthetic (train only) ---------------------------------------------
    if a.backgrounds and patches and a.n_synth > 0:
        bgs = [q for q in Path(a.backgrounds).rglob("*") if q.suffix.lower() in IMG_EXT]
        made = 0
        for j in range(a.n_synth):
            r = synth_image(bgs, patches, a.tile, target, rng)
            if r is None:
                continue
            im, boxes = r
            name = f"synth_{j:06d}"
            cv2.imwrite(str(out / "images/train" / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_yolo(out / "labels/train" / f"{name}.txt", boxes, a.tile, a.tile)
            made += 1
        print(f"[synth] {made} synthetic train images from {len(bgs)} backgrounds")
    else:
        print("[synth] skipped (need --backgrounds and at least one patch)")

    yaml.safe_dump({"path": str(out.resolve()), "train": "images/train", "val": "images/val",
                    "names": {0: "white_tent"}}, (out / "data.yaml").open("w"))
    print(f"[done] {out/'data.yaml'}")


if __name__ == "__main__":
    main()
