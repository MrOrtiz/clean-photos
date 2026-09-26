# clean-photos

Batch cleanup for old, low-resolution photos (1990s lab scans, Photo CD exports, early digital files) that also produces square, face-centered crops ready for LoRA training.

Everything runs locally. No photos are uploaded anywhere.

## What it does

For every image in a folder:

1. **Fixes color and fading.** It pulls the tint of the darkest and brightest tones partway toward neutral (similar to Photoshop's Auto Color), then applies a shared levels stretch to restore contrast. Strength is deliberately partial: full-strength auto-correction wrongly "neutralizes" warm, skin-dominated portraits.
2. **Cleans and upscales.** It uses a neural upscaler through [spandrel](https://github.com/chaiNNer-org/spandrel) (the same loader ComfyUI and chaiNNer use). By default it downloads Real-ESRGAN x2plus on first run; any ESRGAN-family `.pth` or `.safetensors` model also works. Without PyTorch it falls back to Lanczos resizing.
3. **Writes two kinds of output:**
   - `full/`: each whole photo, cleaned, and upscaled only if its short side is under 1536 px. Use these to hand-crop anything the face detector missed.
   - `crops/`: a square head-and-shoulders crop around each detected face, sized exactly to your training resolution.
4. **Flags risky crops.** Any crop that needed more than 2.5x enlargement goes to `crops/_flagged/`, since at that point the upscaler is inventing facial detail. The per-image sizes and upscale factors are listed in `report.csv`.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate

# 1. PyTorch: get the right command for your GPU from https://pytorch.org
# 2. Everything else:
pip install -r requirements.txt
```

If you already run ComfyUI, its Python environment already has PyTorch and spandrel. Run the script with that interpreter and install `opencv-python` if it's missing.

## Usage

```bash
python clean_photos.py INPUT_FOLDER OUTPUT_FOLDER [options]

# typical: small source files, 768 training resolution
python clean_photos.py "D:\photos\1992" "D:\photos\1992_clean" --size 768
```

| Option | Default | Meaning |
|---|---|---|
| `--mode` | `both` | `faces` = crops only, `full` = whole photos only |
| `--size` | `1024` | Crop size. Match your trainer's resolution (768 or 1024) |
| `--crop-scales` | `2.8` | Crop width as a multiple of face width. `2.8` gives head and shoulders, `4.5` roughly half body. Comma-separate to get both |
| `--max-upscale` | `2.5` | Crops needing more enlargement than this go to `_flagged/` |
| `--model` | `auto` | `auto`, a path to a model file, or `none` for Lanczos only |
| `--blend` | `0.8` | 1.0 = pure neural upscale; lower mixes in plain resizing to keep natural grain |
| `--color-strength` | `0.5` | Cast correction strength, 0–1. 0 still fixes fading and contrast |
| `--denoise` | auto | Non-local-means strength. Off with a neural model (it denoises itself), 3 without |
| `--strictness` | `8` | Face detector strictness. Raise it if you get junk crops, lower it if faces are missed |
| `--tile` | `512` | Model tile size. Lower it if you run out of VRAM |

## After running

1. Delete crops of other people and any false detections.
2. Review `crops/_flagged/` and usually leave those crops out of training.
3. Fix red-eye by hand; the script doesn't handle it.
4. Compare faces against the originals at 100% zoom before training.

## Notes and limitations

- **Face detection** uses OpenCV's bundled Haar cascades (no download needed). They're reliable on frontal and three-quarter faces but miss strongly tilted ones, and they occasionally fire on non-faces.
- **Kodak Photo CD `.pcd` and Seattle FilmWorks `.sfw`/`.pwp` files** are skipped. Convert them to PNG first with XnConvert or IrfanView, and for `.pcd` choose the largest stored resolution.
- **Testing:** color correction was tuned by measuring color error against originals across 6 photos x 5 simulated casts, and beat no correction in 22 of 30 cases. Tiled upscaling matches single-pass output within 7/255 per pixel, so there are no visible seams.

## Credits

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al. (BSD-3-Clause). The weights are downloaded from its releases page and not bundled here.
- [spandrel](https://github.com/chaiNNer-org/spandrel) for model loading.
- [OpenCV](https://opencv.org/) for face detection and image processing.
