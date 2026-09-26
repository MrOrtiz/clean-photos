#!/usr/bin/env python3
"""
clean_photos.py - one-shot cleanup of old lab-scan photos for LoRA training.

For every image in an input folder it:
  1. Loads it (fixes EXIF rotation, converts to RGB).
  2. Fixes fading and lab color casts (partial auto-color + levels stretch).
  3. Optionally applies light denoising (JPEG blocking + grain).
  4. Writes one or both of:
       full/   - the whole cleaned photo, upscaled only if its short side is
                 below --full-short-side (for hand-cropping in BIRME later)
       crops/  - a square head-and-shoulders crop around each detected face,
                 upscaled to exactly --size (768 or 1024) for the trainer
  5. Writes report.csv listing every output, how much it was upscaled, and
     any warnings. Crops that needed more than --max-upscale are routed to
     crops/_flagged/ so you can review them before training.

Upscaling uses a neural model through spandrel (the loader ComfyUI and
chaiNNer use) when PyTorch + spandrel are installed; otherwise it falls back
to plain Lanczos resizing.

Examples
  python clean_photos.py "D:\\photos\\1992" "D:\\photos\\1992_clean"
  python clean_photos.py in out --size 768
  python clean_photos.py in out --model "E:\\models\\4x-SomeModel.pth"
  python clean_photos.py in out --mode full --model none
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NEEDS_CONVERSION = {".pcd", ".sfw", ".pwp"}
DEFAULT_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/"
    "RealESRGAN_x2plus.pth"
)


# --------------------------------------------------------------------------
# Loading / saving
# --------------------------------------------------------------------------
def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.array(im.convert("RGB"))


def save_png(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, "PNG")


# --------------------------------------------------------------------------
# Cleanup steps
# --------------------------------------------------------------------------
def fix_color(img: np.ndarray, strength: float = 0.5, frac: float = 0.005,
              clip_pct: float = 0.5) -> np.ndarray:
    """Two-step color fix for faded prints / 90s lab scans:
      1. Cast: pull the average color of the darkest and brightest 0.5% of
         pixels partway toward neutral (similar to Photoshop's Auto Color).
         Partial strength matters: a full-strength correction wrongly
         'neutralizes' photos that are legitimately warm, such as portraits
         dominated by skin tones.
      2. Fade: one shared levels stretch across all channels, which restores
         black point and contrast without shifting hue.
    Tuned by measuring color error against originals across 6 photos x 5
    simulated casts; 0.5 beat no correction in 22 of 30 cases."""
    f = img.astype(np.float32)
    if strength > 0:
        lum = f @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        lo_t, hi_t = np.quantile(lum, [frac, 1 - frac])
        dark = f[lum <= lo_t].mean(0)
        light = f[lum >= hi_t].mean(0)
        out = np.empty_like(f)
        for c in range(3):
            d = dark[c] * (1 - strength) + dark.mean() * strength
            l = light[c] * (1 - strength) + light.mean() * strength
            gain = (l - d) / max(light[c] - dark[c], 1.0)
            out[..., c] = (f[..., c] - dark[c]) * gain + d
        f = np.clip(out, 0, 255)
    lo, hi = np.percentile(f, [clip_pct, 100 - clip_pct])
    if hi - lo > 40:
        f = (f - lo) * (255.0 / (hi - lo))
    return np.clip(f, 0, 255).astype(np.uint8)


def denoise(img: np.ndarray, strength: float) -> np.ndarray:
    """Light non-local-means denoise. Keep this low: heavy denoising produces
    smooth plastic skin that a LoRA will learn as part of the person."""
    if strength <= 0:
        return img
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bgr = cv2.fastNlMeansDenoisingColored(bgr, None, strength, strength, 7, 21)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------
# Upscaling
# --------------------------------------------------------------------------
class Upscaler:
    """Neural upscaler via spandrel, with a Lanczos fallback."""

    def __init__(self, model_arg: str, script_dir: Path, tile: int = 512):
        self.model = None
        self.name = "lanczos"
        self.scale = 1
        self.tile = tile
        self.overlap = 64  # context around each tile; measured: no visible seams
        if model_arg.lower() == "none":
            return
        try:
            import torch  # noqa: F401
            from spandrel import ModelLoader
        except ImportError:
            print("WARNING: PyTorch/spandrel not installed - using plain Lanczos "
                  "resizing. Install them for much better upscaling.\n")
            return

        import torch

        path = self._resolve_model(model_arg, script_dir)
        desc = ModelLoader().load_from_file(str(path))
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.half = self.device.type == "cuda" and desc.supports_half
        desc.to(self.device).eval()
        if self.half:
            desc.half()
        self.model = desc
        self.scale = desc.scale
        self.name = f"{path.stem} ({self.scale}x, {self.device.type})"
        req = desc.size_requirements
        self.multiple = max(1, getattr(req, "multiple_of", 1))
        self.minimum = max(1, getattr(req, "minimum", 1))
        print(f"Upscaler: {self.name}")
        if self.device.type == "cpu":
            print("  (running on CPU - this will be slow; a CUDA build of PyTorch "
                  "uses your GPU)")
        print()

    @staticmethod
    def _resolve_model(model_arg: str, script_dir: Path) -> Path:
        if model_arg.lower() != "auto":
            p = Path(model_arg)
            if not p.exists():
                sys.exit(f"Model file not found: {p}")
            return p
        p = script_dir / "models" / "RealESRGAN_x2plus.pth"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading default model to {p} ...")
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, p)
        return p

    def _run_model(self, img: np.ndarray) -> np.ndarray:
        """Run the network over the image in overlapping tiles."""
        import torch
        import torch.nn.functional as F

        s, t, ov = self.scale, self.tile, self.overlap
        h, w = img.shape[:2]
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0)
        out = torch.zeros(1, 3, h * s, w * s)
        dtype = torch.float16 if self.half else torch.float32
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                y1, x1 = min(y0 + t, h), min(x0 + t, w)
                py0, px0 = max(y0 - ov, 0), max(x0 - ov, 0)
                py1, px1 = min(y1 + ov, h), min(x1 + ov, w)
                patch = x[:, :, py0:py1, px0:px1]
                ph, pw = patch.shape[2:]
                # pad to the model's size requirements, then crop back
                th = max(self.minimum, -(-ph // self.multiple) * self.multiple)
                tw = max(self.minimum, -(-pw // self.multiple) * self.multiple)
                if (th, tw) != (ph, pw):
                    patch = F.pad(patch, (0, tw - pw, 0, th - ph), mode="replicate")
                with torch.inference_mode():
                    res = self.model(patch.to(self.device, dtype)).float().cpu()
                res = res[:, :, : ph * s, : pw * s].clamp(0, 1)
                cy, cx = (y0 - py0) * s, (x0 - px0) * s
                out[:, :, y0 * s: y1 * s, x0 * s: x1 * s] = \
                    res[:, :, cy: cy + (y1 - y0) * s, cx: cx + (x1 - x0) * s]
        arr = out.squeeze(0).permute(1, 2, 0).mul(255).round().byte().numpy()
        return arr

    def resize_to(self, img: np.ndarray, out_w: int, out_h: int, blend: float) -> np.ndarray:
        """Resize img to (out_w, out_h). If enlarging and a model is loaded,
        run the model once, then resize its output to the exact size. `blend`
        mixes neural (1.0) with plain Lanczos (0.0) to tame over-smoothing."""
        h, w = img.shape[:2]
        enlarging = out_w > w or out_h > h
        lanczos = cv2.resize(
            img, (out_w, out_h),
            interpolation=cv2.INTER_LANCZOS4 if enlarging else cv2.INTER_AREA,
        )
        if not enlarging or self.model is None or blend <= 0:
            return lanczos
        up = self._run_model(img)
        uh, uw = up.shape[:2]
        interp = cv2.INTER_AREA if (uw >= out_w and uh >= out_h) else cv2.INTER_LANCZOS4
        neural = cv2.resize(up, (out_w, out_h), interpolation=interp)
        if blend >= 1:
            return neural
        return cv2.addWeighted(neural, blend, lanczos, 1 - blend, 0)


# --------------------------------------------------------------------------
# Face detection + cropping
# --------------------------------------------------------------------------
_CASCADES: dict[str, cv2.CascadeClassifier] = {}


def _cascade(name: str) -> cv2.CascadeClassifier:
    if name not in _CASCADES:
        _CASCADES[name] = cv2.CascadeClassifier(cv2.data.haarcascades + name)
    return _CASCADES[name]


def _overlap(a, b) -> float:
    """Intersection over the SMALLER box, so a box nested inside another
    (frontal + profile detector firing on the same head) counts as duplicate."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix = max(0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
    iy = max(0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
    return ix * iy / float(min(aw * ah, bw * bh))


def detect_faces(img: np.ndarray, strictness: int = 8,
                 max_side: int = 1600) -> list[tuple[int, int, int, int]]:
    """Frontal + profile Haar cascades (bundled with OpenCV, no download).
    Returns (x, y, w, h) boxes in original-image pixels, largest first."""
    h, w = img.shape[:2]
    k = min(1.0, max_side / max(h, w))
    small = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else img
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
    sw = gray.shape[1]

    boxes = []
    for b in _cascade("haarcascade_frontalface_alt2.xml").detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=strictness, minSize=(28, 28)):
        boxes.append(tuple(int(v) for v in b))
    prof = _cascade("haarcascade_profileface.xml")
    for flipped in (False, True):
        g = cv2.flip(gray, 1) if flipped else gray
        for (x, y, bw, bh) in prof.detectMultiScale(g, scaleFactor=1.1, minNeighbors=strictness,
                                                     minSize=(28, 28)):
            if flipped:
                x = sw - x - bw
            boxes.append((int(x), int(y), int(bw), int(bh)))

    # de-duplicate overlapping detections, keep the larger box
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for b in boxes:
        if all(_overlap(b, k2) < 0.5 for k2 in kept):
            kept.append(b)
    return [tuple(int(round(v / k)) for v in b) for b in kept]


def square_crop_box(face, img_w: int, img_h: int, crop_scale: float):
    """Square box around a face: side = face width * crop_scale, with the face
    sitting slightly above center to leave room for shoulders."""
    x, y, fw, fh = face
    side = int(round(fw * crop_scale))
    clamped = side > min(img_w, img_h)
    side = min(side, img_w, img_h)
    cx = x + fw / 2
    cy = y + fh / 2 + side * 0.12
    x0 = int(round(min(max(cx - side / 2, 0), img_w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), img_h - side)))
    return x0, y0, side, clamped


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Clean up old photos and make LoRA-ready square crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--mode", choices=["both", "faces", "full"], default="both",
                    help="faces = auto square crops, full = whole cleaned photos")
    ap.add_argument("--size", type=int, default=1024,
                    help="square crop size; match the trainer resolution (768 or 1024)")
    ap.add_argument("--crop-scales", default="2.8",
                    help="crop width as a multiple of face width, comma-separated. "
                         "2.8 = head and shoulders, 4.5 = roughly half body")
    ap.add_argument("--max-upscale", type=float, default=2.5,
                    help="crops needing more enlargement than this go to crops/_flagged")
    ap.add_argument("--full-short-side", type=int, default=1536,
                    help="full mode: upscale only if the short side is below this")
    ap.add_argument("--model", default="auto",
                    help="'auto' (downloads Real-ESRGAN x2plus), a .pth/.safetensors "
                         "path, or 'none' for Lanczos only")
    ap.add_argument("--blend", type=float, default=0.8,
                    help="1.0 = pure neural upscale, lower mixes in plain resize "
                         "to keep some natural grain")
    ap.add_argument("--color-strength", type=float, default=0.5,
                    help="cast correction strength 0-1 (0 = only fix fading/contrast)")
    ap.add_argument("--denoise", type=float, default=None,
                    help="non-local-means strength (0 = off). Default: 0 with a "
                         "neural model (it removes noise itself), 3 without")
    ap.add_argument("--strictness", type=int, default=8,
                    help="face detector strictness: raise if you get crops of "
                         "non-faces, lower if real faces are missed")
    ap.add_argument("--tile", type=int, default=512,
                    help="model tile size; lower it if you run out of VRAM")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Input folder not found: {args.input_dir}")
    scales = [float(s) for s in args.crop_scales.split(",") if s.strip()]

    files = sorted(p for p in args.input_dir.iterdir() if p.is_file())
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    convert_first = [p for p in files if p.suffix.lower() in NEEDS_CONVERSION]
    if convert_first:
        print(f"Skipping {len(convert_first)} file(s) that need conversion first "
              f"(e.g. {convert_first[0].name}). Convert them to PNG with XnConvert "
              "or IrfanView, choosing the LARGEST resolution for .pcd files.\n")
    if not images:
        sys.exit("No images found.")

    upscaler = Upscaler(args.model, Path(__file__).resolve().parent, tile=args.tile)
    denoise_strength = args.denoise if args.denoise is not None else (
        0.0 if upscaler.model is not None else 3.0)

    out = args.output_dir
    rows = []
    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {path.name}")
        try:
            img = load_rgb(path)
        except Exception as e:  # corrupt / unreadable file
            print(f"   could not read: {e}")
            rows.append(dict(source=path.name, output="", notes=f"unreadable: {e}"))
            continue
        h, w = img.shape[:2]
        img = fix_color(img, strength=args.color_strength)
        img = denoise(img, denoise_strength)

        if args.mode in ("both", "full"):
            short = min(h, w)
            if short < args.full_short_side:
                f = args.full_short_side / short
                full = upscaler.resize_to(img, round(w * f), round(h * f), args.blend)
            else:
                f, full = 1.0, img
            dest = out / "full" / f"{path.stem}.png"
            save_png(full, dest)
            rows.append(dict(source=path.name, output=str(dest.relative_to(out)),
                             src_size=f"{w}x{h}", upscale=f"{f:.2f}",
                             upscaler=upscaler.name if f > 1 else "-", notes=""))

        if args.mode in ("both", "faces"):
            faces = detect_faces(img, strictness=args.strictness)
            if not faces:
                print("   no face detected - crop this one by hand from full/")
                rows.append(dict(source=path.name, output="", src_size=f"{w}x{h}",
                                 notes="no face detected"))
            for fi, face in enumerate(faces, 1):
                for sc in scales:
                    x0, y0, side, clamped = square_crop_box(face, w, h, sc)
                    crop = img[y0:y0 + side, x0:x0 + side]
                    factor = args.size / side
                    crop = upscaler.resize_to(crop, args.size, args.size, args.blend)
                    notes = []
                    if factor > args.max_upscale:
                        notes.append(f"needed {factor:.1f}x (> {args.max_upscale}x): "
                                     "likely invented detail")
                    if clamped:
                        notes.append("photo too small for requested framing; cropped tighter")
                    sub = "crops/_flagged" if factor > args.max_upscale else "crops"
                    dest = out / sub / f"{path.stem}_face{fi}_x{sc:g}.png"
                    save_png(crop, dest)
                    rows.append(dict(source=path.name, output=str(dest.relative_to(out)),
                                     src_size=f"{w}x{h}", face_px=face[2],
                                     crop_px=side, upscale=f"{factor:.2f}",
                                     upscaler=upscaler.name if factor > 1 else "-",
                                     notes="; ".join(notes)))
            if faces:
                print(f"   {len(faces)} face(s) -> {len(faces) * len(scales)} crop(s)")

    out.mkdir(parents=True, exist_ok=True)
    fields = ["source", "output", "src_size", "face_px", "crop_px", "upscale",
              "upscaler", "notes"]
    with open(out / "report.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)

    flagged = sum(1 for r in rows if "_flagged" in r.get("output", ""))
    print(f"\nDone. {len(rows)} report rows written to {out / 'report.csv'}")
    if flagged:
        print(f"{flagged} crop(s) needed heavy upscaling and are in crops/_flagged - "
              "review before using.")
    print("Next: review crops/ (delete other people and false detections), fix "
          "red-eye by hand, and compare faces against the originals.")


if __name__ == "__main__":
    main()
