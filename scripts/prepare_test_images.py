from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from pathlib import Path
from tqdm import tqdm

def crop_faces(input_dir, output_dir, output_size=(256, 256)):
    """
    Detects and crops faces from a directory of local images.

    Args:
        input_dir (Path): The directory containing the source images.
        output_dir (Path): The directory to save the cropped face images.
        output_size (tuple): The target size for the cropped images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = MTCNN()
    
    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    print(f"Found {len(image_paths)} images to process.")

    for image_path in tqdm(image_paths):
        try:
            img = Image.open(image_path).convert('RGB')
            pixels = np.array(img)

            results = detector.detect_faces(pixels)

            if not results:
                print(f"  - Warning: No face detected in {image_path.name}. Skipping.")
                continue

            main_face = max(results, key=lambda r: r['box'][2] * r['box'][3])
            x, y, width, height = main_face['box']
            x, y = max(x, 0), max(y, 0)
            
            face = img.crop((x, y, x + width, y + height))
            face = face.resize(output_size, Image.Resampling.LANCZOS)

            output_path = output_dir / f"cropped_{image_path.name}"
            face.save(output_path)

        except Exception as e:
            print(f"  - An unexpected error occurred with {image_path.name}: {e}")

def main():
    """Main function to run the face cropping process."""
    # By default, this script will process the images downloaded by download_images.py
    # You can change this to your own directory of local images.
    INPUT_DIR = Path("data/downloads")
    OUTPUT_DIR = Path("data/raw_images")
    
    if not INPUT_DIR.exists() or not any(INPUT_DIR.iterdir()):
        print(f"Error: Input directory not found or is empty: {INPUT_DIR}")
        print("Please download images first using download_images.py or place your own images there.")
        return

    print(f"--- Starting Face Cropping from '{INPUT_DIR}' ---")
    crop_faces(INPUT_DIR, OUTPUT_DIR)
    print(f"--- Finished Face Cropping. Results are in '{OUTPUT_DIR}' ---")

if __name__ == "__main__":
    main()