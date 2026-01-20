#!/usr/bin/env python3
"""Verify that camera overlays are being drawn correctly on maps."""

import json
import rasterio
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

# Load camera data
with open("danby_forest/audit.geojson") as f:
    geojson = json.load(f)

cameras = []
for feature in geojson["features"]:
    props = feature["properties"]
    # Skip non-camera features
    if props.get("layer") != "recommended_cameras":
        continue
    coords = feature["geometry"]["coordinates"]
    cameras.append({
        "lon": coords[0],
        "lat": coords[1],
        "type": props["type"],
        "score": props["score"]
    })

print(f"Found {len(cameras)} cameras:")
for i, cam in enumerate(cameras, 1):
    print(f"  {i}. {cam['type']}: ({cam['lat']:.4f}, {cam['lon']:.4f}) - score: {cam['score']:.2f}")

# Load hillshade and create test image
hillshade_path = Path("danby_forest/debug/hillshade.tif")
output_path = Path("danby_forest/debug/test_overlay.png")

print(f"\nProcessing {hillshade_path}...")

with rasterio.open(hillshade_path) as src:
    hillshade = src.read(1)
    transform = src.transform
    
    print(f"Hillshade shape: {hillshade.shape}")
    print(f"Transform: {transform}")
    
    # Convert to RGB
    img_array = np.stack([hillshade, hillshade, hillshade], axis=-1)
    img = Image.fromarray(img_array, mode="RGB")
    
    # Draw camera markers
    draw = ImageDraw.Draw(img)
    
    markers_drawn = 0
    for i, cam in enumerate(cameras, 1):
        # Convert lat/lon to pixel coordinates
        col, row = ~transform * (cam["lon"], cam["lat"])
        col, row = int(col), int(row)
        
        print(f"  Camera {i}: ({cam['lat']:.4f}, {cam['lon']:.4f}) -> pixel ({col}, {row})")
        
        # Check if inside bounds
        if 0 <= row < hillshade.shape[0] and 0 <= col < hillshade.shape[1]:
            # Draw different colors for different camera types
            color = (255, 0, 0) if cam["type"] == "pinch" else (0, 255, 0)
            radius = 8
            
            # Draw circle marker
            draw.ellipse(
                [col - radius, row - radius, col + radius, row + radius],
                fill=color,
                outline=(255, 255, 255),
                width=2,
            )
            
            # Draw camera number (larger for visibility)
            draw.text(
                (col - 3, row - 5),
                str(i),
                fill=(255, 255, 255),
            )
            
            markers_drawn += 1
            print(f"    ✓ Drew marker")
        else:
            print(f"    ✗ Outside bounds!")
    
    print(f"\nDrew {markers_drawn}/{len(cameras)} markers")
    
    # Save test image
    img.save(output_path)
    print(f"Saved test image to: {output_path}")
    print(f"\nPlease open {output_path} in an image viewer to verify the markers are visible.")
