import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DATA_DIR   = r"C:\Users\Aditya\Desktop\Medical_Image_Anomaly_Detection\data\lgg_segmentation_dataset"
MODEL_PATH = "models/unet_brain_segmentation.pth"
IMG_SIZE   = 256
BATCH_SIZE = 8
LR         = 1e-4
EPOCHS     = 30
PATIENCE   = 7
VAL_SPLIT  = 0.2

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class LGGDataset(Dataset):
    def __init__(self, image_mask_pairs, img_size=256):
        self.pairs    = image_mask_pairs
        self.img_size = img_size
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")     # grayscale
        return self.img_transform(image), self.mask_transform(mask)


def load_pairs(data_dir):
    """Collect all (image, mask) pairs from all patient folders."""
    pairs = []
    patient_folders = sorted(glob.glob(os.path.join(data_dir, "TCGA_*")))
    for folder in patient_folders:
        masks = sorted(glob.glob(os.path.join(folder, "*_mask.tif")))
        for mask_path in masks:
            img_path = mask_path.replace("_mask.tif", ".tif")
            if os.path.exists(img_path):
                pairs.append((img_path, mask_path))
    print(f"Total image-mask pairs found: {len(pairs)}")
    return pairs


# ─────────────────────────────────────────────
# UNET ARCHITECTURE
# ─────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs     = nn.ModuleList()
        self.ups       = nn.ModuleList()
        self.pool      = nn.MaxPool2d(2, 2)

        # Encoder
        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x    = self.ups[i](x)
            skip = skip_connections[i // 2]
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final(x)


# ─────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = torch.sigmoid(pred)
        pred   = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class CombinedLoss(nn.Module):
    """Dice + BCE — works best for medical image segmentation."""
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.dice(pred, target) + self.bce(pred, target)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def dice_score(pred, target, threshold=0.5):
    pred   = (torch.sigmoid(pred) > threshold).float()
    pred   = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2 * intersection + 1) / (pred.sum() + target.sum() + 1)


def iou_score(pred, target, threshold=0.5):
    pred   = (torch.sigmoid(pred) > threshold).float()
    pred   = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return (intersection + 1) / (union + 1)


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    best_dice    = 0.0
    patience_ctr = 0

    for epoch in range(EPOCHS):
        # ── Train
        model.train()
        running_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # ── Validate
        model.eval()
        val_loss = val_dice = val_iou = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds     = model(images)
                val_loss += criterion(preds, masks).item()
                val_dice += dice_score(preds, masks).item()
                val_iou  += iou_score(preds, masks).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_iou  = val_iou  / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_val_dice)
        history["val_iou"].append(avg_val_iou)

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]  "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  "
              f"Dice: {avg_val_dice:.4f}  "
              f"IoU: {avg_val_iou:.4f}")

        scheduler.step(avg_val_dice)

        if avg_val_dice > best_dice:
            best_dice    = avg_val_dice
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✅ Best model saved (Dice: {best_dice:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n⏹ Early stopping after {epoch+1} epochs.")
                break

    print(f"\n🏆 Best Dice Score: {best_dice:.4f}")


# ─────────────────────────────────────────────
# PLOT HISTORY
# ─────────────────────────────────────────────
def plot_history():
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], 'b-o', label='Train Loss')
    ax1.plot(epochs, history["val_loss"],   'r-o', label='Val Loss')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()

    ax2.plot(epochs, history["val_dice"], 'g-o', label='Dice Score')
    ax2.plot(epochs, history["val_iou"],  'm-o', label='IoU Score')
    ax2.set_title('Metrics'); ax2.set_xlabel('Epoch'); ax2.legend()

    plt.tight_layout()
    plt.savefig("outputs/segmentation_history.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# VISUALIZE PREDICTIONS
# ─────────────────────────────────────────────
def visualize_predictions(model, val_loader, num_samples=4):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    images_shown = 0
    fig, axes    = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3))
    fig.suptitle("Segmentation Results", fontsize=16, fontweight='bold')

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds = torch.sigmoid(model(images))

            for i in range(images.shape[0]):
                if images_shown >= num_samples:
                    break

                # Denormalize image for display
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)

                true_mask = masks[i].cpu().squeeze().numpy()
                pred_mask = (preds[i].cpu().squeeze().numpy() > 0.5).astype(np.float32)

                axes[images_shown, 0].imshow(img)
                axes[images_shown, 0].set_title("MRI Image")
                axes[images_shown, 1].imshow(true_mask, cmap='gray')
                axes[images_shown, 1].set_title("True Mask")
                axes[images_shown, 2].imshow(pred_mask, cmap='gray')
                axes[images_shown, 2].set_title("Predicted Mask")

                for ax in axes[images_shown]:
                    ax.axis('off')

                images_shown += 1

            if images_shown >= num_samples:
                break

    plt.tight_layout()
    plt.savefig("outputs/segmentation_predictions.png", dpi=150)
    plt.show()
    print("✅ Segmentation predictions saved to outputs/segmentation_predictions.png")


# ─────────────────────────────────────────────
# OVERLAY VISUALIZATION (tumor circled on MRI)
# ─────────────────────────────────────────────
def visualize_overlay(model, val_loader, num_samples=4):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    images_shown = 0
    fig, axes    = plt.subplots(num_samples, 2, figsize=(8, num_samples * 3))
    fig.suptitle("Tumor Detection Overlay", fontsize=16, fontweight='bold')

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            preds  = torch.sigmoid(model(images))

            for i in range(images.shape[0]):
                if images_shown >= num_samples:
                    break

                # Denormalize
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)

                pred_mask = (preds[i].cpu().squeeze().numpy() > 0.5).astype(np.float32)

                # Create red overlay on tumor region
                overlay      = img.copy()
                tumor_pixels = pred_mask > 0.5
                overlay[tumor_pixels, 0] = 1.0   # red channel
                overlay[tumor_pixels, 1] = 0.0   # green channel
                overlay[tumor_pixels, 2] = 0.0   # blue channel

                axes[images_shown, 0].imshow(img)
                axes[images_shown, 0].set_title("Original MRI")
                axes[images_shown, 1].imshow(overlay)
                axes[images_shown, 1].set_title("Tumor Highlighted")

                for ax in axes[images_shown]:
                    ax.axis('off')

                images_shown += 1

            if images_shown >= num_samples:
                break

    plt.tight_layout()
    plt.savefig("outputs/tumor_overlay.png", dpi=150)
    plt.show()
    print("✅ Tumor overlay saved to outputs/tumor_overlay.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Load data
    pairs = load_pairs(DATA_DIR)

    # Train/val split
    val_size   = int(len(pairs) * VAL_SPLIT)
    train_size = len(pairs) - val_size
    train_pairs, val_pairs = random_split(pairs, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(42))

    train_dataset = LGGDataset(list(train_pairs), IMG_SIZE)
    val_dataset   = LGGDataset(list(val_pairs),   IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")

    # Build model
    model     = UNet(in_channels=3, out_channels=1).to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Train
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)

    # Plot training history
    plot_history()

    # Visualize predictions (mask vs predicted)
    visualize_predictions(model, val_loader, num_samples=4)

    # Visualize tumor overlay (red highlight on MRI)
    visualize_overlay(model, val_loader, num_samples=4)

    print("\n🎉 Segmentation training complete!")
    print("Outputs saved:")
    print("  - models/unet_brain_segmentation.pth")
    print("  - outputs/segmentation_history.png")
    print("  - outputs/segmentation_predictions.png")
    print("  - outputs/tumor_overlay.png")
