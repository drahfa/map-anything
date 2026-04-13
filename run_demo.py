"""Minimal MapAnything demo for MacBook Air M4 (MPS backend)."""

import torch
from mapanything.models import MapAnything
from mapanything.utils.device import get_device
from mapanything.utils.image import load_images
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.viz import predictions_to_glb
import numpy as np
import time
import subprocess
import sys

OUTPUT_GLB = "output.glb"

device = get_device()
print(f"Device: {device}")

# Load model (downloads ~600MB on first run)
print("Loading model (cached weights load fast)...")
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

# Collect results for GLB export
world_points_list = []
images_list = []
masks_list = []

for i, pred in enumerate(outputs):
    depth = pred["depth_z"][0].squeeze(-1)
    intrinsics = pred["intrinsics"][0]
    pose = pred["camera_poses"][0]
    pts3d, valid = depthmap_to_world_frame(depth, intrinsics, pose)
    mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
    mask = mask & valid.cpu().numpy()
    image_np = pred["img_no_norm"][0].cpu().numpy()

    world_points_list.append(pts3d.cpu().numpy())
    images_list.append(image_np)
    masks_list.append(mask)

    d = depth.cpu().numpy()
    print(f"\nView {i}:")
    print(f"  Depth range: {d[d>0].min():.2f} - {d[d>0].max():.2f} meters")
    print(f"  3D points: {mask.sum()} valid out of {mask.size}")
    print(f"  Camera focal length: fx={intrinsics[0,0]:.1f}, fy={intrinsics[1,1]:.1f}")

# Export 3D scene as GLB
print(f"\nExporting 3D scene to {OUTPUT_GLB}...")
predictions = {
    "world_points": np.stack(world_points_list, axis=0),
    "images": np.stack(images_list, axis=0),
    "final_masks": np.stack(masks_list, axis=0),
}
scene_3d = predictions_to_glb(predictions, as_mesh=True)
scene_3d.export(OUTPUT_GLB)
print(f"Saved {OUTPUT_GLB}")

# Open in macOS viewer
print("Opening 3D viewer...")
subprocess.Popen(["open", OUTPUT_GLB])
