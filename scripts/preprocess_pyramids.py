import torch
from transformers import Dinov2Model, AutoImageProcessor
import os
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ImageDataset(Dataset):
    """A simple dataset to load images from a directory."""
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image, return_tensors="pt")
        return image, image_path.name

def get_model_and_processor():
    """Loads the DINOv2 model and image processor from Hugging Face."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Dinov2Model.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor, device

def construct_feature_pyramid(patch_embeddings, target_scales=[4, 8, 16]):
    """Constructs a feature pyramid from high-resolution patch embeddings."""
    pyramid = {}
    # Assuming patch_embeddings are square, e.g., (1, 256, 768) -> 16x16 grid
    bs, num_patches, dim = patch_embeddings.shape
    grid_size = int(num_patches**0.5)
    
    # Reshape to (batch_size, grid_size, grid_size, dim) for easier downsampling
    features_grid = patch_embeddings.reshape(bs, grid_size, grid_size, dim)

    for scale in sorted(target_scales, reverse=True):
        if grid_size == scale:
            # Store the features directly
            pyramid[f"{scale}x{scale}"] = features_grid.reshape(bs, scale*scale, dim)
        elif grid_size > scale:
            # Downsample using average pooling
            stride = grid_size // scale
            pooled_features = torch.nn.functional.avg_pool2d(
                features_grid.permute(0, 3, 1, 2), # (bs, dim, H, W)
                kernel_size=stride,
                stride=stride
            ).permute(0, 2, 3, 1) # (bs, H', W', dim)
            
            pyramid[f"{scale}x{scale}"] = pooled_features.reshape(bs, scale*scale, dim)
            
    return pyramid

def main():
    """
    Main function to run the preprocessing pipeline.
    """
    # --- Configuration ---
    RAW_IMAGE_DIR = Path("data/raw_images")
    OUTPUT_DIR = Path("data/feature_pyramids")
    BATCH_SIZE = 16 # Adjust based on your GPU memory
    PYRAMID_SCALES = [4, 8, 16] # The grid sizes for the feature pyramid

    # --- Setup ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_IMAGE_DIR.exists() or not any(RAW_IMAGE_DIR.iterdir()):
        print(f"Error: Raw image directory not found or is empty: {RAW_IMAGE_DIR}")
        print("Please download a dataset (e.g., a subset of FFHQ) and place it there.")
        return

    model, processor, device = get_model_and_processor()

    # --- Dataset and DataLoader ---
    # We pass the processor directly to the dataset now
    dataset = ImageDataset(RAW_IMAGE_DIR, transform=processor)
    # We can't batch images and filenames separately like this easily.
    # Let's handle filenames inside the loop.
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Found {len(dataset)} images to process.")

    # --- Processing Loop ---
    print("Starting feature pyramid extraction...")
    with torch.no_grad():
        for i, (batch, filenames) in enumerate(tqdm(dataloader)):
            inputs = batch['pixel_values'][0].to(device)

            # Get patch embeddings from DINOv2
            outputs = model(inputs, output_hidden_states=True)
            # The last hidden state contains the patch embeddings
            patch_embeddings = outputs.last_hidden_state

            # Construct the feature pyramid for each item in the batch
            pyramids = construct_feature_pyramid(patch_embeddings, target_scales=PYRAMID_SCALES)
            
            # Save each pyramid to a file
            for j in range(patch_embeddings.size(0)):
                single_pyramid = {key: val[j] for key, val in pyramids.items()}
                output_filename = Path(filenames[j]).with_suffix(".pt")
                torch.save(single_pyramid, OUTPUT_DIR / output_filename)

    print(f"Successfully processed {len(dataset)} images.")
    print(f"Feature pyramids saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()