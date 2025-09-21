import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms import Normalize
from torchvision.utils import save_image
from transformers import Dinov2Model
import matplotlib.pyplot as plt

# --- 1. Dataset ---
class FeaturePyramidDataset(Dataset):
    """Loads pre-computed feature pyramids from disk."""
    def __init__(self, pyramid_dir):
        self.pyramid_dir = Path(pyramid_dir)
        self.pyramid_files = sorted(list(self.pyramid_dir.glob("*.pt")))

    def __len__(self):
        return len(self.pyramid_files)

    def __getitem__(self, idx):
        return torch.load(self.pyramid_files[idx])

# --- 2. Generator Architecture (Improved for PoC) ---

class MappingNetwork(nn.Module):
    """A simple MLP to map latent z to intermediate w."""
    def __init__(self, latent_dim, hidden_dim, w_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, w_dim)
        )

    def forward(self, z):
        return self.net(z)

class GeneratorBlock(nn.Module):
    """An upscaling block with simplified style injection."""
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.style_bias = nn.Linear(w_dim, out_channels)

    def forward(self, x, w):
        x = self.activation(self.conv1(x))
        bias = self.style_bias(w).unsqueeze(-1).unsqueeze(-1)
        x = x + bias
        x = self.activation(self.conv2(x))
        bias2 = self.style_bias(w).unsqueeze(-1).unsqueeze(-1)
        x = x + bias2
        return x

class ToRGB(nn.Module):
    """Layer to convert a feature map to an RGB image."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    """The complete progressive generator with improved architecture."""
    def __init__(self, latent_dim=512, w_dim=512, hidden_dim=512):
        super().__init__()
        self.mapping_network = MappingNetwork(latent_dim, hidden_dim, w_dim)
        self.initial_constant = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        # --- Initialize for 4x4 stage ---
        self.block0 = GeneratorBlock(512, 512, w_dim)
        self.to_rgb0 = ToRGB(512)
        
        self.blocks = nn.ModuleList([self.block0])
        self.to_rgb_layers = nn.ModuleList([self.to_rgb0])

    def forward(self, z, current_stage=0):
        w = self.mapping_network(z)
        x = self.initial_constant.repeat(z.shape[0], 1, 1, 1)
        
        # For now, we only use the first block
        x = self.blocks[current_stage](x, w)
        return self.to_rgb_layers[current_stage](x)


# --- 3. Training Loop ---
def train():
    """Main function to run the training process."""
    # --- Configuration ---
    PYRAMID_DIR = Path("data/feature_pyramids")
    OUTPUT_DIR = Path("training_progress")
    LATENT_DIM = 512
    W_DIM = 512
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 5 # Use all 5 images
    EPOCHS = 500 # More epochs for a small dataset
    SAVE_IMAGE_INTERVAL = 50

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    OUTPUT_DIR.mkdir(exist_ok=True)

    dataset = FeaturePyramidDataset(PYRAMID_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    generator = Generator(latent_dim=LATENT_DIM, w_dim=W_DIM).to(device)
    optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Load the frozen DINOv2 model and freeze its weights
    dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device)
    for param in dino_model.parameters():
        param.requires_grad = False
    dino_model.eval()

    # DINOv2 normalization transform
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # --- Training the 4x4 Stage ---
    print("--- Starting Training for 4x4 Stage ---")
    
    fixed_noise = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
    losses = []

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            # 1. Get real features from the dataset
            real_features_16x16 = batch['16x16'].to(device)
            
            # Downsample the real features to the target size for this stage (4x4)
            reshaped_real_features = real_features_16x16.reshape(BATCH_SIZE, 16, 16, -1)
            real_features_4x4 = nn.functional.avg_pool2d(
                reshaped_real_features.permute(0, 3, 1, 2),
                kernel_size=4,
                stride=4
            ).permute(0, 2, 3, 1).reshape(BATCH_SIZE, 4*4, -1)

            # 2. Generate images from latent vectors
            z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
            fake_images_4x4 = generator(z, current_stage=0)

            # 3. Get DINO features of the generated images
            upsampled_fake_images = nn.functional.interpolate(fake_images_4x4, size=(224, 224), mode='bilinear', align_corners=False)
            normalized_fake_images = normalize(upsampled_fake_images)
            
            outputs = dino_model(normalized_fake_images, output_hidden_states=True)
            generated_features_16x16 = outputs.last_hidden_state[:, 1:, :]
            
            # Downsample the generated features to match the target size
            reshaped_generated_features = generated_features_16x16.reshape(BATCH_SIZE, 16, 16, -1)
            generated_features_4x4 = nn.functional.avg_pool2d(
                reshaped_generated_features.permute(0, 3, 1, 2),
                kernel_size=4,
                stride=4
            ).permute(0, 2, 3, 1).reshape(BATCH_SIZE, 4*4, -1)

            # 4. Calculate loss and backpropagate
            loss = criterion(generated_features_4x4, real_features_4x4)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_epoch_loss:.4f}")

        # 5. Save sample images
        if (epoch + 1) % SAVE_IMAGE_INTERVAL == 0:
            with torch.no_grad():
                sample_images = generator(fixed_noise, current_stage=0)
                # Upscale for visibility
                sample_images = nn.functional.interpolate(sample_images, size=(256, 256), mode='nearest')
                # Denormalize for viewing: output is not activated, so we clamp to [-1, 1]
                sample_images = torch.clamp(sample_images, -1, 1)
                sample_images = (sample_images + 1) / 2
                save_image(sample_images, OUTPUT_DIR / f"epoch_{epoch+1}_4x4.png", nrow=5)

    print("--- Finished Training for 4x4 Stage ---")
    
    # 6. Plot and save the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Generator Loss Curve (4x4 Stage)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig(OUTPUT_DIR / "loss_curve_4x4.png")
    print(f"Loss curve saved to {OUTPUT_DIR / 'loss_curve_4x4.png'}")

    # Save the model checkpoint
    torch.save(generator.state_dict(), OUTPUT_DIR / "generator_stage_4x4.pth")


if __name__ == "__main__":
    train()
