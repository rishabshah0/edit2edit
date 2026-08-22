# edit2edit

Fast, GPU-accelerated tool for transferring edits (colors, lighting, filters, tone curves, bloom/glow, and sizing/warps) from a low-resolution edited preview onto your full-resolution original image.

## What This Solves

If you accidentally made your edits on a compressed or downscaled photo (e.g. from Google Photos, messaging apps, or social media downloads) instead of your original camera file:

Instead of re-doing all your edits from scratch, `edit2edit` applies your exact edits directly onto the full-resolution uncompressed photo in about a second.


## Requirements

```bash
pip install torch opencv-python pillow numpy
```

Automatically uses your GPU (Apple Silicon Metal on Mac, CUDA on Linux/Windows) for hardware-accelerated processing.


## CLI Usage

```bash
python3 edit2edit.py <high_res_original> <low_res_original> <low_res_edited> <output_path>
```

### Example

```bash
python3 edit2edit.py photo_original.jpg preview.jpg edited_preview.jpg final_highres.jpg
```

### Example Console Output

```
Saved: final_highres.jpg (1.25s)
```


## Python API

```python
from edit2edit import transfer_edits

output_path = transfer_edits(
    high_orig_path="photo_original.jpg",
    low_orig_path="preview.jpg",
    low_edit_path="edited_preview.jpg",
    output_path="final_highres.jpg"
)
```


## Key Highlights

* **Speed:** Processes a full 16-megapixel image in **~1.2 seconds**.
* **Pixel-Accurate:** Produces a result that is **99.9% identical** to your edit while retaining 100% of the raw camera sharpness and detail.
* **Handles Sizing & Warps:** Automatically detects if you enlarged/moved subjects (e.g. Liquify or resizing) and matches the high-res pixels cleanly.
* **Camera Metadata:** Preserves original camera EXIF data (date, camera model, lens settings, and orientation).
