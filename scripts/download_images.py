import requests
from pathlib import Path
import io

def download_images(image_urls, output_dir):
    """
    Downloads a list of images from URLs and saves them to a directory.

    Args:
        image_urls (list): A list of URLs for the images to download.
        output_dir (Path): The directory to save the downloaded images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(image_urls):
        try:
            print(f"Downloading image {i+1}/{len(image_urls)}: {url}")
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            
            # Get the filename from the URL
            filename = url.split('/')[-1]
            # A basic attempt to get a reasonable filename
            if '?' in filename:
                filename = filename.split('?')[0]
            
            # Get the extension
            extension = Path(filename).suffix
            if not extension:
                extension = ".jpg" # Default to jpg if no extension found

            output_path = output_dir / f"download_{i+1:02d}{extension}"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"  - Successfully saved to {output_path}")

        except requests.RequestException as e:
            print(f"  - Error downloading image {i+1}: {e}")
        except Exception as e:
            print(f"  - An unexpected error occurred with image {i+1}: {e}")

def main():
    """Main function to run the image download."""
    IMAGE_URLS = [
        'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d',
        'https://images.unsplash.com/photo-1552058544-f2b08422138a',
        'https://images.unsplash.com/photo-1544005313-94ddf0286df2',
        'https://images.unsplash.com/photo-1554151228-14d9def656e4',
        'https://images.unsplash.com/photo-1494790108377-be9c29b29330'
    ]
    OUTPUT_DIR = Path("data/downloads")
    
    print("--- Starting Image Download ---")
    download_images(IMAGE_URLS, OUTPUT_DIR)
    print("--- Finished Image Download ---")

if __name__ == "__main__":
    main()
