#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


def box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    pad = radius // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel_size=radius, stride=1)


def guided_filter(guide: torch.Tensor, src: torch.Tensor, radius: int = 7, eps: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    mean_I = box_filter(guide, radius)
    mean_p = box_filter(src, radius)
    mean_Ip = box_filter(guide * src, radius)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = box_filter(guide * guide, radius)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return a, b


def read_tensor(path: str | Path) -> tuple[torch.Tensor, bytes | None]:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return tensor, img.info.get("exif")


def compute_optical_flow(source_path: str | Path, target_path: str | Path) -> np.ndarray:
    src = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    dst = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(dst, src, None)


def transfer_edits(
    high_orig_path: str | Path,
    low_orig_path: str | Path,
    low_edit_path: str | Path,
    output_path: str | Path,
    iterations: int = 3,
    filter_radius: int = 7,
    filter_eps: float = 1e-3,
    jpeg_quality: int = 97,
    warp_threshold: float = 0.5,
) -> Path:
    high_orig, exif = read_tensor(high_orig_path)
    low_orig, _ = read_tensor(low_orig_path)
    low_edit, _ = read_tensor(low_edit_path)

    _, _, H_high, W_high = high_orig.shape
    _, _, H_low, W_low = low_edit.shape

    if low_orig.shape[-2:] != (H_low, W_low):
        low_orig = F.interpolate(low_orig, size=(H_low, W_low), mode="bilinear", align_corners=False)

    flow = compute_optical_flow(low_orig_path, low_edit_path)
    flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    if torch.quantile(torch.norm(flow_tensor, dim=1), 0.9) > warp_threshold:
        flow_scaled = F.interpolate(flow_tensor, size=(H_high, W_high), mode="bilinear", align_corners=False)
        flow_scaled[:, 0] *= W_high / W_low
        flow_scaled[:, 1] *= H_high / H_low

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H_high, device=DEVICE),
            torch.linspace(-1, 1, W_high, device=DEVICE),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        grid[..., 0] += flow_scaled[:, 0] * (2.0 / W_high)
        grid[..., 1] += flow_scaled[:, 1] * (2.0 / H_high)

        high_aligned = F.grid_sample(high_orig, grid, mode="bicubic", padding_mode="reflection", align_corners=False)
        low_guide = F.interpolate(high_aligned, size=(H_low, W_low), mode="bilinear", align_corners=False)
    else:
        high_aligned = high_orig
        low_guide = low_orig

    a, b = guided_filter(low_guide, low_edit, radius=filter_radius, eps=filter_eps)
    a_high = F.interpolate(a, size=(H_high, W_high), mode="bicubic", align_corners=False)
    b_high = F.interpolate(b, size=(H_high, W_high), mode="bicubic", align_corners=False)
    refined = torch.clamp(a_high * high_aligned + b_high, 0.0, 1.0)

    for _ in range(iterations):
        down = F.interpolate(refined, size=(H_low, W_low), mode="bilinear", align_corners=False)
        res_high = F.interpolate(low_edit - down, size=(H_high, W_high), mode="bicubic", align_corners=False)

        a_res, b_res = guided_filter(refined, res_high, radius=3, eps=1e-4)
        res_guided = box_filter(a_res, 3) * refined + box_filter(b_res, 3)
        refined = torch.clamp(refined + 0.95 * res_guided, 0.0, 1.0)

    out_arr = (refined.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    out_img = Image.fromarray(out_arr)

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {"quality": jpeg_quality} if dest.suffix.lower() in {".jpg", ".jpeg"} else {}
    if exif:
        save_kwargs["exif"] = exif

    out_img.save(dest, **save_kwargs)
    return dest


if __name__ == "__main__":
    parser = ArgumentParser(description="Transfer edits from low-res preview onto full-resolution image.")
    parser.add_argument("high_orig", type=Path, help="High-resolution unedited original")
    parser.add_argument("low_orig", type=Path, help="Low-resolution unedited image")
    parser.add_argument("low_edit", type=Path, help="Low-resolution edited image")
    parser.add_argument("output", type=Path, help="Output path for high-resolution edited result")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--quality", type=int, default=97)

    args = parser.parse_args()
    t0 = time.time()
    out = transfer_edits(args.high_orig, args.low_orig, args.low_edit, args.output, iterations=args.iterations, jpeg_quality=args.quality)
    print(f"Saved: {out} ({time.time() - t0:.2f}s)")
