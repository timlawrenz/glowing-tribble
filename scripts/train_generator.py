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
    """The complete progressive generator with a unified architecture."""
    def __init__(self, latent_dim=512, w_dim=512, hidden_dim=512, initial_res=4):
        super().__init__()
        self.mapping_network = MappingNetwork(latent_dim, hidden_dim, w_dim)
        self.initial_constant = nn.Parameter(torch.randn(1, 512, initial_res, initial_res))
        
        # Unified lists for all blocks and ToRGB layers
        self.blocks = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()
        
        # Add the initial 4x4 stage
        self.add_stage(512, 512, w_dim)

    def add_stage(self, in_channels, out_channels, w_dim):
        """Adds a new upscaling block and ToRGB layer."""
        if not self.blocks: # This is the first block (4x4)
            self.blocks.append(GeneratorBlock(in_channels, out_channels, w_dim))
        else: # Subsequent blocks are upsampling blocks
            # A proper progressive GAN would have different block structures
            # but for this PoC, we'll keep it simple.
            self.blocks.append(GeneratorBlock(in_channels, out_channels, w_dim))
        
        self.to_rgb_layers.append(ToRGB(out_channels))

    def forward(self, z, stage, alpha=1.0):
        w = self.mapping_network(z)
        x = self.initial_constant.repeat(z.shape[0], 1, 1, 1)

        # Pass through the initial block (which is now blocks[0])
        x = self.blocks[0](x, w)

        if stage == 0:
            return self.to_rgb_layers[0](x)

        # Upsample and pass through subsequent blocks
        for i in range(1, stage + 1):
            x_prev = x
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            x = self.blocks[i](x, w)

        # Fade-in logic
        rgb_prev = self.to_rgb_layers[stage - 1](x_prev)
        rgb_prev = nn.functional.interpolate(rgb_prev, scale_factor=2, mode='nearest')
        rgb_new = self.to_rgb_layers[stage](x)
        
        return torch.lerp(rgb_prev, rgb_new, alpha)


# --- 3. Training Loop ---
def train_stage(stage, generator, optimizer, dataloader, dino_model, device, config):
    """A function to handle the training for a single progressive stage."""
    
    resolution = 4 * (2 ** stage)
    stage_output_dir = config['output_dir'] / f"{resolution}x{resolution}"
    stage_output_dir.mkdir(exist_ok=True)
    
    target_features_key = f"{resolution}x{resolution}"
    epochs = config[f"epochs_{resolution}x{resolution}"]
    
    print(f"--- Starting Training for {resolution}x{resolution} Stage ---")
    
    losses = []
    fixed_noise = torch.randn(config['batch_size'], config['latent_dim'], device=device)
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            alpha = min(1.0, (epoch * len(dataloader) + i) / (epochs * len(dataloader) * 0.5)) if stage > 0 else 1.0

            real_features = batch[target_features_key].to(device)

            z = torch.randn(config['batch_size'], config['latent_dim'], device=device)
            fake_images = generator(z, stage, alpha)

            upsampled_fake = nn.functional.interpolate(fake_images, size=(224, 224), mode='bilinear', align_corners=False)
            normalized_fake = normalize(upsampled_fake)
            
            generated_features_full = dino_model(normalized_fake, output_hidden_states=True).last_hidden_state[:, 1:, :]
            
            # Downsample to target resolution
            reshaped_generated = generated_features_full.reshape(config['batch_size'], 16, 16, -1)
            downsample_factor = 16 // resolution
            generated_features = nn.functional.avg_pool2d(
                reshaped_generated.permute(0, 3, 1, 2), 
                kernel_size=downsample_factor, 
                stride=downsample_factor
            ).permute(0, 2, 3, 1).reshape(config['batch_size'], resolution*resolution, -1)

            loss = criterion(generated_features, real_features)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

        if (epoch + 1) % (epochs // 4 or 1) == 0: # Ensure interval is at least 1
            with torch.no_grad():
                sample_images = generator(fixed_noise, stage, alpha=1.0)
                sample_images = nn.functional.interpolate(sample_images, size=(256, 256), mode='nearest')
                sample_images = torch.clamp(sample_images, -1, 1) / 2 + 0.5
                save_image(sample_images, stage_output_dir / f"epoch_{epoch+1}.png", nrow=5)

    print(f"--- Finished Training for {resolution}x{resolution} Stage ---")
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(f"Generator Loss Curve ({resolution}x{resolution} Stage)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig(stage_output_dir / "loss_curve.png")
    print(f"Loss curve saved to {stage_output_dir / 'loss_curve.png'}")
    
    torch.save(generator.state_dict(), stage_output_dir / "generator.pth")


def main():
    """Main function to orchestrate the progressive training."""
    # --- Configuration ---
    config = {
        "pyramid_dir": Path("data/feature_pyramids"),
        "output_dir": Path("training_progress"),
        "latent_dim": 512,
        "w_dim": 512,
        "learning_rate": 1e-4,
        "batch_size": 5,
        "epochs_4x4": 10, # Reduced for faster testing
        "epochs_8x8": 20,
    }

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    config['output_dir'].mkdir(exist_ok=True)

    dataset = FeaturePyramidDataset(config['pyramid_dir'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    generator = Generator(latent_dim=config['latent_dim'], w_dim=config['w_dim']).to(device)
    optimizer = optim.Adam(generator.parameters(), lr=config['learning_rate'])
    
    dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device)
    for param in dino_model.parameters():
        param.requires_grad = False
    dino_model.eval()

    # --- Run Training Stages ---
    
    # Check if a 4x4 checkpoint exists and train if not
    checkpoint_4x4_path = config['output_dir'] / "4x4" / "generator.pth"
    if not checkpoint_4x4_path.exists():
        train_stage(0, generator, optimizer, dataloader, dino_model, device, config)
    else:
        print("Found 4x4 checkpoint. Loading weights.")
        generator.load_state_dict(torch.load(checkpoint_4x4_path))

    # Add and train the 8x8 stage
    generator.add_stage(512, 512, config['w_dim'])
    generator.to(device)
    optimizer = optim.Adam(generator.parameters(), lr=config['learning_rate'])
    
    checkpoint_8x8_path = config['output_dir'] / "8x8" / "generator.pth"
    if not checkpoint_8x8_path.exists():
        train_stage(1, generator, optimizer, dataloader, dino_model, device, config)
    else:
        print("Found 8x8 checkpoint. Loading weights.")
        generator.load_state_dict(torch.load(checkpoint_8x8_path))


if __name__ == "__main__":
    main()
