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

The second milestone was to develop and train the progressive generator. The model starts by generating a 4x4 image and is progressively grown by adding new blocks to double the resolution at each stage, all the way up to 256x256.

The generator is guided by a Mean Squared Error (MSE) loss between the DINOv2 patch embeddings of its generated images and the pre-computed ground truth embeddings from Milestone 1.

## Final Results

The model was trained successfully through all stages. The final images and loss curves for each stage are presented below.

### 4x4 Stage
| Final Image | Loss Curve |
| :---: | :---: |
| ![Final 4x4 Output](/.gemini-file-_home_tim_source_activity_glowing-tribble_examples_visualizations_final_4x4.png-428025565) | ![4x4 Loss Curve](/.gemini-file-_home_tim_source_activity_glowing-tribble_examples_visualizations_loss_curve_4x4.png-428025566) |

### 8x8 Stage
| Final Image | Loss Curve |
| :---: | :---: |
| ![Final 8x8 Output](/.gemini-file-_home_tim_source_activity_glowing-tribble_examples_visualizations_final_8x8.png-428025567) | ![8x8 Loss Curve](/.gemini-file-_home_tim_source_activity_glowing-tribble_examples_visualizations_loss_curve_8x8.png-428025568) |

### 16x16 Stage
| Final Image | Loss Curve |
| :---: | :---: |
| ![Final 16x16 Output](/.gemini-file-_home_tim_source_activity_glowing-tribble_examples_visualizations_final_16x16.png-428025569) | ![16x16 Loss Curve](/.gemini-file-_home_tim_source_activity_glowing-tribble_examples_visualizations_loss_curve_16x16.png-428025570) |

### 32x32 Stage
| Final Image | Loss Curve |
| :---: | :---: |
| ![Final 32x32 Output](/.gemini-file-_home_tim_source_activity_glowing-tribble_examples_visualizations_final_32x32.png-428025571) | ![32x32 Loss Curve](/.gemini-file-_home_tim_source_activity_glowing-tribble_examples_visualizations_loss_curve_32x32.png-428025572) |

### 64x64 Stage
| Final Image | Loss Curve |
| :---: | :---: |
| ![Final 64x64 Output](/.gemini-file-_home_tim_source_activity_glowing-tribble_examples_visualizations_final_64x64.png-428025573) | ![64x64 Loss Curve](/.gemini-file-_home_tim_source_activity_glowing-tribble_examples_visualizations_loss_curve_64x64.png-428025574) |

### 128x128 Stage
| Final Image | Loss Curve |
| :---: | :---: |
| ![Final 128x128 Output](/.gemini-file-_home_tim_source_activity_glowing-tribble_examples_visualizations_final_128x128.png-428025575) | ![128x128 Loss Curve](/.gemini-file-_home_tim_source_activity_glowing-tribble_examples_visualizations_loss_curve_128x128.png-428025576) |

### 256x256 Stage
| Final Image | Loss Curve |
| :---: | :---: |
| ![Final 256x256 Output](/.gemini-file-_home_tim_source_activity_glowing-tribble_examples_visualizations_final_256x256.png-428025577) | ![256x256 Loss Curve](/.gemini-file-_home_tim_source_activity_glowing-tribble_examples_visualizations_loss_curve_256x256.png-428025578) |

