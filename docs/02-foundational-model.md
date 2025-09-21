# Milestone 2: Foundational Model Training

## Objective

The goal of this milestone is to build and train the initial, low-resolution stages of our progressive generator. The core hypothesis to be validated is that a generative model can be trained to produce coherent, structured images using the pre-computed DINOv2 feature pyramids as its sole guidance, without the need for a traditional GAN discriminator.

Success in this milestone will prove that our core generative mechanism is viable.

## Generator Architecture

We will implement a simple progressive generator inspired by the architecture used in Progressive GANs. The model will consist of a series of upscaling blocks, each responsible for doubling the image resolution.

### Initial Block (From Latent to 4x4)

*   **Input:** A latent vector `z` of 512 dimensions.
*   **Process:** A dense layer will project the latent vector to a higher-dimensional space, which is then reshaped into a `4x4` grid of feature vectors. This will serve as the initial input to the generator's convolutional layers.
*   **Output:** A `4x4` grid of feature vectors.

### Upscaling Blocks (e.g., 4x4 -> 8x8)

*   **Input:** A feature map from the previous block (e.g., `4x4`).
*   **Process:** Each block will consist of:
    1.  An upsampling layer (e.g., `torch.nn.Upsample` with `scale_factor=2`).
    2.  A series of 2D convolutional layers (`Conv2D`) with LeakyReLU activations to learn to refine the features at the new, higher resolution.
*   **Output:** A feature map at double the resolution (e.g., `8x8`).

### ToRGB Layer

*   At each resolution level, a separate `1x1 Conv2D` layer will be used to transform the feature map from its high-dimensional representation into a 3-channel (RGB) image. This allows us to generate an image at every stage of the generator, which is crucial for the progressive training process.

## Training Plan: Progressive Growing

We will train the model progressively, starting with the lowest resolution and gradually adding and training new upscaling blocks. This stabilizes training and allows the model to learn coarse features before moving on to fine details.

### Phase 1: Training the 4x4 Generator

1.  **Model:** The generator will only consist of the initial block and a `ToRGB` layer to produce a `4x4` image.
2.  **Training Loop:**
    *   Generate a `4x4` image from a random latent vector `z`.
    *   Feed this tiny `4x4` image into the frozen DINOv2 model to get its `4x4` patch embeddings.
    *   Fetch the pre-computed `4x4` DINO feature map for a real image from our dataset.
    *   **Loss Calculation:** The loss will be the Mean Squared Error (MSE) between the DINO embeddings of the generated image and the pre-computed DINO features of the real image.
    *   Backpropagate and update the weights of the `4x4` generator block.
3.  **Goal:** The model learns to generate `4x4` pixel blobs that are semantically similar to the `4x4` DINO features of real faces.

### Phase 2: Fading in the 8x8 Generator

1.  **Model:** Add the next upscaling block (`4x4 -> 8x8`) and its corresponding `ToRGB` layer. The weights of the `4x4` block are initially frozen.
2.  **Fade-in Technique:** To ensure a smooth transition, for a certain number of training iterations, the output image will be a weighted average of the upsampled `4x4` image and the new `8x8` image. This "alpha" weight will be gradually increased from 0 to 1.
3.  **Training Loop:**
    *   The loss is calculated using the `8x8` DINO features as the target.
    *   Backpropagate and update the weights of **only the new `8x8` block**.
4.  **Goal:** The model learns to generate coherent `8x8` images without destabilizing the already-trained lower-resolution layers. Once the fade-in is complete, both the `4x4` and `8x8` blocks will be trained together.

This process of adding a block, fading it in, and then training it will be repeated for `16x16`, `32x32`, and so on, up to our target resolution of `256x256`.

## Implementation Details

*   **Script:** A new script, `scripts/train_generator.py`, will be created.
*   **Optimizer:** We will use the Adam optimizer, a standard choice for generative models.
*   **Batch Size:** A small batch size (e.g., 8 or 16) will be used, depending on GPU memory constraints.

## Success Criteria

Milestone 2 will be considered complete when:

*   **Quantitative:** The training loss for each stage (4x4, 8x8, 16x16) steadily decreases, proving that the model is successfully learning to mimic the DINO feature targets at each scale.
*   **Qualitative (Visual Inspection):**
    *   The low-resolution outputs (e.g., up to 32x32) show coherent, face-like structures. We should be able to discern the general location of eyes, a nose, and a mouth in the generated pixel blobs.
    *   This visual evidence is the primary indicator that our core hypothesis is correct and that the DINO-guided loss is sufficient to guide the generator.

## Results: 4x4 Stage

The initial 4x4 stage was trained for 500 epochs on the 5-image test dataset. The training was successful and validated our core hypothesis.

### Loss Curve

The loss decreased steadily and converged, demonstrating that the generator was successfully learning to minimize the DINO feature distance.

![Loss Curve 4x4](https://raw.githubusercontent.com/timlawrenz/glowing-tribble/main/examples/visualizations/loss_curve_4x4.png)

### Sample Output

The following image grid shows the final 4x4 pixel output from the generator after 500 epochs, upscaled to 256x256 for visibility. While abstract, the images show clear structure and consistency, having learned the general color palette and layout of the training faces.

![Sample 4x4 Output](https://raw.githubusercontent.com/timlawrenz/glowing-tribble/main/examples/visualizations/epoch_500_4x4.png)

## Results: 8x8 Stage

Following the successful 4x4 training, the 8x8 stage was trained for 20 epochs. The model successfully learned to generate features at this higher resolution, demonstrating the viability of the progressive growing approach.

### Loss Curve

The loss for the 8x8 stage also shows a clear downward trend, validating the fade-in and training process for the new layer.

![Loss Curve 8x8](https://raw.githubusercontent.com/timlawrenz/glowing-tribble/main/examples/visualizations/loss_curve_8x8.png)

### Sample Output

The 8x8 images show a clear increase in detail and coherence compared to the 4x4 stage, with more defined shapes emerging.

![Sample 8x8 Output](https://raw.githubusercontent.com/timlawrenz/glowing-tribble/main/examples/visualizations/epoch_20_8x8.png)

## Results: 16x16 Stage

The 16x16 stage was trained for 30 epochs. At this resolution, recognizable facial features begin to emerge, further validating the effectiveness of the DINO-guided progressive training approach.

### Loss Curve

The loss curve for the 16x16 stage continues to show a healthy downward trend.

![Loss Curve 16x16](https://raw.githubusercontent.com/timlawrenz/glowing-tribble/main/examples/visualizations/loss_curve_16x16.png)

### Sample Output

The generated images now clearly show emerging facial structures, such as eyes, noses, and mouths, confirming that the model is learning meaningful representations.

![Sample 16x16 Output](https://raw.githubusercontent.com/timlawrenz/glowing-tribble/main/examples/visualizations/epoch_28_16x16.png)

## Results: 32x32 Stage

The 32x32 stage was trained for 40 epochs. The generated images continue to improve in quality and coherence.

### Loss Curve

![Loss Curve 32x32](https://raw.githubusercontent.com/timlawrenz/glowing-tribble/main/examples/visualizations/loss_curve_32x32.png)

### Sample Output

![Sample 32x32 Output](https://raw.githubusercontent.com/timlawrenz/glowing-tribble/main/examples/visualizations/epoch_40_32x32.png)

## Results: 64x64 Stage

The 64x64 stage was trained for 50 epochs. The generated images now show significantly more detail and are beginning to look like plausible, albeit blurry, human faces.

### Loss Curve

![Loss Curve 64x64](https://raw.githubusercontent.com/timlawrenz/glowing-tribble/main/examples/visualizations/loss_curve_64x64.png)

### Sample Output

![Sample 64x64 Output](https://raw.githubusercontent.com/timlawrenz/glowing-tribble/main/examples/visualizations/final_64x64.png)
