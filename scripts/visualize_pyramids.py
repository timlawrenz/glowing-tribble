import torch
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm

def visualize_pyramid(pyramid_path, original_image_path, output_path):
    """
    Creates and saves a visualization of a feature pyramid.

    Args:
        pyramid_path (Path): Path to the pre-computed .pt pyramid file.
        original_image_path (Path): Path to the corresponding original cropped image.
        output_path (Path): Path to save the composite visualization image.
    """
    # Load the pyramid and the original image
    pyramid = torch.load(pyramid_path)
    original_image = Image.open(original_image_path)
    
    # Get the spatial pyramid keys (e.g., '4x4', '16x16'), excluding 'cls_token'
    spatial_keys = [k for k in pyramid.keys() if 'x' in k]

    # Fit PCA on the highest resolution features to find the primary components
    highest_res_key = sorted(spatial_keys, key=lambda k: int(k.split('x')[0]), reverse=True)[0]
    highest_res_features = pyramid[highest_res_key].cpu().numpy()
    
    pca = PCA(n_components=3)
    pca.fit(highest_res_features)

    # Create a list of images to composite, starting with the original
    images_to_display = [original_image]

    # Process each level of the pyramid
    for key in sorted(spatial_keys, key=lambda k: int(k.split('x')[0])):
        features = pyramid[key].cpu().numpy()
        grid_size = int(key.split('x')[0])

        # Transform features to 3D for RGB visualization
        features_3d = pca.transform(features)

        # Normalize to 0-255 range
        normalized_features = (features_3d - features_3d.min()) / (features_3d.max() - features_3d.min())
        img_array = (normalized_features * 255).astype(np.uint8)
        
        # Reshape to an image
        img = Image.fromarray(img_array.reshape(grid_size, grid_size, 3))
        
        # Resize to match the original image for easy viewing
        img = img.resize(original_image.size, Image.Resampling.NEAREST)
        images_to_display.append(img)

    # Create the final composite image
    total_width = original_image.width * len(images_to_display)
    max_height = original_image.height
    
    composite_image = Image.new('RGB', (total_width, max_height))
    
    for i, img in enumerate(images_to_display):
        composite_image.paste(img, (i * original_image.width, 0))
        
    composite_image.save(output_path)

def main():
    """Main function to run the visualization."""
    PYRAMID_DIR = Path("data/feature_pyramids")
    RAW_IMAGE_DIR = Path("data/raw_images")
    OUTPUT_DIR = Path("data/visualizations")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pyramid_files = sorted(list(PYRAMID_DIR.glob("*.pt")))
    if not pyramid_files:
        print(f"Error: No feature pyramids found in {PYRAMID_DIR}.")
        print("Please run the preprocessing script first.")
        return

    print(f"Found {len(pyramid_files)} feature pyramids to visualize.")

    for pyramid_path in tqdm(pyramid_files):
        # Find the corresponding original image
        original_image_name = pyramid_path.with_suffix(".jpg").name
        original_image_path = RAW_IMAGE_DIR / original_image_name
        
        if not original_image_path.exists():
            print(f"  - Warning: Could not find original image for {pyramid_path.name}. Skipping.")
            continue
            
        output_path = OUTPUT_DIR / pyramid_path.with_suffix(".png").name
        visualize_pyramid(pyramid_path, original_image_path, output_path)

    print(f"Visualizations saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
