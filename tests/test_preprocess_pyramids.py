import torch
import pytest
from PIL import Image
from pathlib import Path
import sys

# Add the script's directory to the Python path to allow importing
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from preprocess_pyramids import construct_feature_pyramid, ImageDataset

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def dummy_image_dir(tmpdir_factory):
    """Creates a temporary directory with a dummy image for testing."""
    img_dir = Path(tmpdir_factory.mktemp("images"))
    dummy_image = Image.new('RGB', (224, 224), color = 'red')
    dummy_image.save(img_dir / "test_image.png")
    return img_dir

@pytest.fixture(scope="module")
def precomputed_pyramid(tmpdir_factory):
    """
    Generates a sample feature pyramid to test against.
    This simulates the output of the main script without needing a real model.
    """
    # Create a realistic-looking high-resolution feature map
    # DINOv2-base has a dimension of 768. A 224x224 image results in a 16x16 grid of patches.
    batch_size = 1
    num_patches = 16 * 16
    embedding_dim = 768
    high_res_features = torch.randn(batch_size, num_patches, embedding_dim)

    # Use the actual function from the script to create the pyramid
    pyramid = construct_feature_pyramid(high_res_features, target_scales=[4, 8, 16])
    
    # We only care about the features for the single image in the batch
    single_pyramid = {key: val[0] for key, val in pyramid.items()}
    single_pyramid['cls_token'] = torch.randn(embedding_dim)

    # Save it to a temporary file
    pyramid_dir = Path(tmpdir_factory.mktemp("pyramids"))
    pyramid_path = pyramid_dir / "test_pyramid.pt"
    torch.save(single_pyramid, pyramid_path)
    
    return pyramid_path

# --- Test Cases ---

def test_image_dataset(dummy_image_dir):
    """Tests that the ImageDataset correctly finds and loads images."""
    dataset = ImageDataset(dummy_image_dir)
    assert len(dataset) == 1
    image, filename = dataset[0]
    assert filename == "test_image.png"
    assert isinstance(image, Image.Image)

def test_pyramid_read_write_integrity(precomputed_pyramid):
    """
    Test Case 1: Verifies that a saved pyramid can be loaded back from disk.
    """
    assert precomputed_pyramid.exists()
    loaded_pyramid = torch.load(precomputed_pyramid)
    assert loaded_pyramid is not None, "Loaded pyramid should not be None."

def test_pyramid_format_and_structure(precomputed_pyramid):
    """
    Test Case 2: Validates the structure and types of the loaded pyramid.
    """
    loaded_pyramid = torch.load(precomputed_pyramid)
    
    assert isinstance(loaded_pyramid, dict), "Pyramid should be a dictionary."
    
    expected_keys = ['cls_token', '4x4', '8x8', '16x16']
    assert sorted(list(loaded_pyramid.keys())) == sorted(expected_keys), f"Pyramid keys do not match expected keys: {expected_keys}"
    
    for key in expected_keys:
        assert isinstance(loaded_pyramid[key], torch.Tensor), f"Value for key '{key}' should be a torch.Tensor."

def test_pyramid_tensor_shapes_and_types(precomputed_pyramid):
    """
    Test Case 3: Verifies the shape and dtype of the tensors in the pyramid.
    """
    loaded_pyramid = torch.load(precomputed_pyramid)
    embedding_dim = 768 # For DINOv2-base

    expected_shapes = {
        'cls_token': (embedding_dim,),
        '4x4': (4 * 4, embedding_dim),
        '8x8': (8 * 8, embedding_dim),
        '16x16': (16 * 16, embedding_dim)
    }

    for key, expected_shape in expected_shapes.items():
        tensor = loaded_pyramid[key]
        assert tensor.shape == expected_shape, f"Shape for '{key}' is incorrect. Expected {expected_shape}, got {tensor.shape}."
        assert tensor.dtype == torch.float32, f"Dtype for '{key}' should be torch.float32, but got {tensor.dtype}."