# Milestone 1: DINOv2 Feature Pyramid Pre-computation

## Objective

The goal of this milestone is to create the "ground truth" dataset that our progressive generator will be trained on. This involves processing a dataset of face images with a pre-trained DINOv2 model to extract multi-scale patch embeddings (a "feature pyramid") for each image and saving them to disk for the training phase.

This process is critical for the success of the project, as the quality and correctness of this pre-computed data will directly determine the generator's ability to learn.

## Technical Setup for Reproducibility

To ensure this research is transparent and easily reproducible, we will establish a clean and well-documented environment.

### 1. Environment and Dependencies

The project will use Python 3.11. All dependencies will be managed in a `requirements.txt` file to ensure consistent setups.

**Action:** Create a `requirements.txt` file in the root directory with the following initial libraries:

```
torch
transformers
accelerate
tqdm
pytest
```

To install these, one would run:
`pip install -r requirements.txt`

### 2. Dataset

We will use a high-quality, standardized dataset of human faces. A subset of 5,000 to 10,000 images from one of the following is recommended:

*   **FFHQ (Flickr-Faces-HQ)**
*   **CelebA-HQ**

A script or instructions should be provided to download and prepare this dataset, placing it in a `data/raw_images/` directory (which should be added to `.gitignore`).

## Implementation Plan: `preprocess.py`

A Python script, let's call it `preprocess.py`, will be created to execute the feature pyramid generation. It will perform the following major steps:

### Step 1: Initialization

*   Load the pre-trained DINOv2 model (e.g., `facebook/dinov2-base`) and image processor from the Hugging Face `transformers` library. The model should be in evaluation mode (`.eval()`) and moved to the appropriate device (GPU if available).
*   Create a PyTorch `Dataset` and `DataLoader` to efficiently load the source images from `data/raw_images/`. The loader should handle image transformations required by the DINOv2 processor.

### Step 2: Main Processing Loop

*   Iterate through the `DataLoader` with a `tqdm` progress bar for visibility.
*   For each batch of images:
    *   Run a forward pass through the frozen DINOv2 model to get the final hidden states (patch embeddings). This will yield a high-resolution grid of feature vectors (e.g., a 16x16 grid for a 224x224 input).

### Step 3: Feature Pyramid Construction

*   For each image's high-resolution feature grid, create a multi-scale pyramid by progressively down-sampling.
*   **Example:**
    1.  Start with the `16x16` grid of patch vectors.
    2.  Create an `8x8` grid by averaging `2x2` blocks of vectors from the `16x16` grid.
    3.  Create a `4x4` grid by averaging `2x2` blocks from the `8x8` grid.
*   This process continues until the coarsest desired scale is reached. The resulting pyramid for one image will be a list or dictionary of tensors: `{'4x4': tensor, '8x8': tensor, '16x16': tensor, ...}`.

### Step 4: Serialization and Storage

*   Create an output directory, e.g., `data/feature_pyramids/`.
*   For each image, save its corresponding feature pyramid to a file. Using `torch.save()` to a `.pt` file is efficient.
*   The filename should correspond to the original image's filename to maintain a clear link (e.g., `00001.png` -> `00001.pt`).

## Testing and Validation Plan

To ensure the integrity of our pre-computed dataset, we will create a test suite using `pytest`. A file `test_preprocessing.py` will be created.

### Test Case 1: Data Integrity and Read/Write Verification

*   **Purpose:** To confirm that a saved feature pyramid can be loaded back from disk without corruption.
*   **Steps:**
    1.  Run the `preprocess.py` script on a single, known image.
    2.  The test will load the resulting `.pt` file using `torch.load()`.
    3.  Assert that the loaded object is not `None`.

### Test Case 2: Format and Structure Validation

*   **Purpose:** To ensure the loaded data has the expected structure (e.g., a dictionary with specific keys).
*   **Steps:**
    1.  Load a pre-computed pyramid file.
    2.  Assert that the loaded object is a dictionary.
    3.  Assert that it contains the expected keys (e.g., `'4x4'`, `'8x8'`, `'16x16'`).
    4.  Assert that the value for each key is a `torch.Tensor`.

### Test Case 3: Shape and Type Verification

*   **Purpose:** To ensure the tensors in the pyramid have the correct dimensions and data type.
*   **Steps:**
    1.  Load a pre-computed pyramid file.
    2.  For each key-value pair in the pyramid:
        *   Assert that the tensor's shape matches the expected dimensions (e.g., the `'8x8'` tensor has a shape like `[1, 64, 768]`, assuming a batch size of 1, 64 patches, and a 768-dim model).
        *   Assert that the tensor's `dtype` is `torch.float32`.

## Definition of Done

Milestone 1 will be considered complete when:

*   A `requirements.txt` file is created and populated.
*   The `preprocess.py` script is implemented and successfully runs on the dataset subset, generating the feature pyramid files.
*   The `test_preprocessing.py` test suite is implemented and all tests pass, validating the correctness of the generated data.
*   The generated data is stored in `data/feature_pyramids/`.
