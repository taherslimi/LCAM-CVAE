import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

# Channel Attention Mechanism (Lightweight)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # Adaptive average pooling to summarize each channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Fully connected network to reduce and recalculate channel importance
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()  # Generate weights between 0 and 1 for each channel
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Calculate attention weights for each channel
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # Apply the attention weights to the channels
        return x * y


# Denoising with LCAM
class DenoisingNetwork(nn.Module):
    def __init__(self):
        super(DenoisingNetwork, self).__init__()
        # Layer 1: First convolution to extract basic features
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # Layer 2: Convolution to refine features
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Layer 3: Convolution to learn more complex features
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Layer 4: Convolution to capture global information
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Layer 5: Final convolution before applying attention mechanism
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Layer 6: First channel attention mechanism
        self.attention1 = ChannelAttention(64)
        # Layer 7: Generate the mean μ for the latent space
        self.mean = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # Layer 8: Generate the standard deviation σ for the latent space
        self.std = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # Layer 9: First deconvolution to start the reconstruction
        self.deconv1 = nn.ConvTranspose2d(32, 64, kernel_size=3, padding=1)
        # Layer 10: Second channel attention mechanism
        self.attention2 = ChannelAttention(64)
        # Layer 11: Second deconvolution to refine the reconstruction
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        # Layer 12: Third deconvolution to recover finer details
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        # Layer 13: Final deconvolution with Sigmoid activation to produce the final image
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Feature extraction (Layers 1 to 5)
        x = F.relu(self.conv1(x))  # Layer 1
        x = F.relu(self.conv2(x))  # Layer 2
        x = F.relu(self.conv3(x))  # Layer 3
        x = F.relu(self.conv4(x))  # Layer 4
        x = F.relu(self.conv5(x))  # Layer 5
        # Apply attention mechanism (Layer 6)
        x = self.attention1(x)
        # Latent space with sampling (Layers 7 and 8)
        mu = self.mean(x)  # Generate μ
        log_var = self.std(x)  # Generate σ
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)  # Reparametrization trick
        # Reconstruction (Layers 9 to 13)
        x = F.relu(self.deconv1(z))  # Layer 9
        x = self.attention2(x)  # Layer 10 (Attention mechanism)
        x = F.relu(self.deconv2(x))  # Layer 11
        x = F.relu(self.deconv3(x))  # Layer 12
        x = torch.sigmoid(self.deconv4(x))  # Layer 13
        return x


# Example dataset for training
class UltrasoundDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform=None):
        self.clean_dir = clean_dir  # Path to directory with clean images
        self.noisy_dir = noisy_dir  # Path to directory with noisy images
        self.transform = transform
        self.clean_images = sorted(os.listdir(clean_dir))
        self.noisy_images = sorted(os.listdir(noisy_dir))

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        # Load images from the directories
        clean_image_path = os.path.join(self.clean_dir, self.clean_images[idx])
        noisy_image_path = os.path.join(self.noisy_dir, self.noisy_images[idx])

        # Open the images (convert to grayscale if needed)
        clean_image = Image.open(clean_image_path).convert('L')
        noisy_image = Image.open(noisy_image_path).convert('L')

        # Convert to numpy arrays
        clean_image = np.array(clean_image, dtype=np.float32) / 255.0
        noisy_image = np.array(noisy_image, dtype=np.float32) / 255.0

        # Add an additional channel dimension for the model
        clean_image = np.expand_dims(clean_image, axis=0)
        noisy_image = np.expand_dims(noisy_image, axis=0)

        # Convert to tensors
        clean_image = torch.tensor(clean_image)
        noisy_image = torch.tensor(noisy_image)

        return noisy_image, clean_image


# Specify directories containing the images
clean_dir = "path/to/clean/images"  # Replace with the actual path to clean images
noisy_dir = "path/to/noisy/images"  # Replace with the actual path to noisy images

# Data loading
train_dataset = UltrasoundDataset(clean_dir=clean_dir, noisy_dir=noisy_dir)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model
model = DenoisingNetwork()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Initial learning rate
criterion = nn.MSELoss()  # Mean Squared Error loss function

# Training
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for noisy_images, clean_images in train_loader:
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Make predictions
        denoised_images = model(noisy_images)

        # Calculate loss
        loss = criterion(denoised_images, clean_images)
        epoch_loss += loss.item()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

    # Print average loss for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss / len(train_loader):.4f}")

print("Training completed.")