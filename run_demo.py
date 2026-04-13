"""Minimal MapAnything demo for MacBook Air M4 (MPS backend)."""

import torch
from mapanything.models import MapAnything
from mapanything.utils.device import get_device
from mapanything.utils.image import load_images
from mapanything.utils.geometry import depthmap_to_world_frame
import numpy as np
import time

device = get_device()
print(f"Device: {device}")

# Load model (downloads ~600MB on first run)
print("Loading model (this downloads weights on first run)...")
t0 = time.time()
model = MapAnything.from_pretrained("facebook/map-anything-apache").to(device)
print(f"Model loaded in {time.time()-t0:.1f}s")

# Load images
views = load_images("demo_input")
print(f"Loaded {len(views)} views")

# Run inference
print("Running inference...")
t0 = time.time()
outputs = model.infer(
    views,
    memory_efficient_inference=True,
    minibatch_size=1,
    use_amp=True,
    amp_dtype="fp16",  # MPS doesn't support bf16
    apply_mask=True,
    mask_edges=True,
)
elapsed = time.time() - t0
print(f"Inference done in {elapsed:.1f}s")

# Print results summary
for i, pred in enumerate(outputs):
    depth = pred["depth_z"][0].squeeze(-1)
    intrinsics = pred["intrinsics"][0]
    pose = pred["camera_poses"][0]
    pts3d, valid = depthmap_to_world_frame(depth, intrinsics, pose)
    mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
    mask = mask & valid.cpu().numpy()

    d = depth.cpu().numpy()
    p = pts3d.cpu().numpy()
    print(f"\nView {i}:")
    print(f"  Depth range: {d[d>0].min():.2f} - {d[d>0].max():.2f} meters")
    print(f"  3D points: {mask.sum()} valid out of {mask.size}")
    print(f"  Camera focal length: fx={intrinsics[0,0]:.1f}, fy={intrinsics[1,1]:.1f}")
    print(f"  Camera position: {pose[:3,3].cpu().numpy()}")

print("\nDemo complete!")
