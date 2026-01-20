#!/usr/bin/env python3
"""Check if camera markers are actually visible in the saved PNG files."""

from PIL import Image
import sys

def check_markers_in_image(image_path):
    """Check if there are colored markers (non-grayscale pixels) in the image."""
    print(f"\nChecking {image_path}...")
    img = Image.open(image_path)
    print(f"  Size: {img.width}x{img.height}")
    print(f"  Mode: {img.mode}")
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Check for non-grayscale pixels (markers should be colored)
    colored_pixels = 0
    for y in range(0, img.height, 10):  # Sample every 10 pixels
        for x in range(0, img.width, 10):
            r, g, b = img.getpixel((x, y))
            # Check if pixel is not grayscale (R≠G≠B)
            if not (r == g == b):
                colored_pixels += 1
    
    total_sampled = (img.height // 10) * (img.width // 10)
    colored_percentage = (colored_pixels / total_sampled) * 100
    
    print(f"  Colored pixels: {colored_pixels}/{total_sampled} sampled ({colored_percentage:.1f}%)")
    
    if colored_pixels > 0:
        print(f"  ✓ Image contains colored markers!")
        return True
    else:
        print(f"  ✗ No colored markers found")
        return False

# Check all three images
files = [
    "danby_forest_test/debug/overview_map.png",
    "danby_forest_test/debug/heatmap_pinch.png",
    "danby_forest_test/debug/heatmap_bedding.png"
]

all_good = True
for f in files:
    if not check_markers_in_image(f):
        all_good = False

if all_good:
    print("\n✅ All images have markers!")
else:
    print("\n⚠️  Some images may be missing markers")
