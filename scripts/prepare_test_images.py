import requests
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from pathlib import Path
import io

def prepare_test_images(image_urls, output_dir, output_size=(256, 256)):
    """
    Downloads, crops, and saves a list of images for the test dataset.

    Args:
        image_urls (list): A list of URLs for the images to process.
        output_dir (Path): The directory to save the processed images.
        output_size (tuple): The target size for the cropped images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = MTCNN()

    for i, url in enumerate(image_urls):
        try:
            print(f"Processing image {i+1}/{len(image_urls)}: {url}")
            response = requests.get(url, timeout=20)
            response.raise_for_status()

            img = Image.open(io.BytesIO(response.content)).convert('RGB')
            pixels = np.array(img)

            # Detect faces
            results = detector.detect_faces(pixels)

            if not results:
                print(f"  - Warning: No face detected in image {i+1}. Skipping.")
                continue

            # Select the most prominent face (largest bounding box)
            main_face = max(results, key=lambda r: r['box'][2] * r['box'][3])
            x, y, width, height = main_face['box']
            
            # Ensure coordinates are within image bounds
            x, y = max(x, 0), max(y, 0)
            
            # Crop the face from the image
            face = img.crop((x, y, x + width, y + height))
            
            # Resize to the target output size
            face = face.resize(output_size, Image.Resampling.LANCZOS)

            # Save the processed image
            output_path = output_dir / f"test_image_{i+1:02d}.jpg"
            face.save(output_path)
            print(f"  - Successfully saved cropped face to {output_path}")

        except requests.RequestException as e:
            print(f"  - Error downloading image {i+1}: {e}")
        except Exception as e:
            print(f"  - An unexpected error occurred with image {i+1}: {e}")

def main():
    """Main function to run the image preparation."""
    IMAGE_URLS = [
        'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d',
        'https://images.unsplash.com/photo-1552058544-f2b08422138a',
        'https://images.unsplash.com/photo-1544005313-94ddf0286df2',
        'https://images.unsplash.com/photo-1554151228-14d9def656e4',
        'https://images.unsplash.com/photo-1494790108377-be9c29b29330'
    ]
    OUTPUT_DIR = Path("data/raw_images")
    
    print("--- Starting Test Image Preparation ---")
    prepare_test_images(IMAGE_URLS, OUTPUT_DIR)
    print("--- Finished Test Image Preparation ---")

if __name__ == "__main__":
    main()
