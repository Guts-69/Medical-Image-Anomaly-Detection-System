import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DATA_DIR  = r"C:\Users\Aditya\Desktop\Medical_Image_Anomaly_Detection\data"
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR  = os.path.join(DATA_DIR, "Testing")

BATCH_SIZE  = 32
LR          = 1e-4
EPOCHS      = 20
NUM_CLASSES = 4
PATIENCE    = 5
MODEL_PATH  = "models/efficientnet_brain_tumor_best.pth"

# ─────────────────────────────────────────────
# TRANSFORMS  (cleaner augmentation for MRI)
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# DATASETS & LOADERS
# ─────────────────────────────────────────────
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

CLASS_NAMES = train_dataset.classes
print("Classes:", CLASS_NAMES)

# ─────────────────────────────────────────────
# MODEL  (simpler classifier head = better results)
# ─────────────────────────────────────────────
def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze backbone initially
    for param in model.features.parameters():
        param.requires_grad = False

    # Simple classifier — works better than deep head for small datasets
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(model.classifier[1].in_features, NUM_CLASSES),
    )
    return model.to(device)

model = build_model()

# No label smoothing — plain CrossEntropyLoss works better here
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
history = {"train_loss": [], "train_acc": [], "test_acc": []}

def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train_model():
    best_acc      = 0.0
    patience_ctr  = 0
    unfreeze_done = False

    for epoch in range(EPOCHS):
        # Unfreeze backbone after epoch 5
        if epoch == 5 and not unfreeze_done:
            print("\n🔓 Unfreezing backbone for fine-tuning...")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer.add_param_group({"params": model.features.parameters(), "lr": LR / 10})
            unfreeze_done = True

        model.train()
        running_loss = correct = total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted  = torch.max(outputs, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        test_acc  = evaluate()
        avg_loss  = running_loss / len(train_loader)

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.4f}  "
              f"Train Acc: {train_acc:.2f}%  Test Acc: {test_acc:.2f}%")

        scheduler.step(test_acc)

        if test_acc > best_acc:
            best_acc     = test_acc
            patience_ctr = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✅ Best model saved (Test Acc: {best_acc:.2f}%)")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n⏹ Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"\n🏆 Best Test Accuracy: {best_acc:.2f}%")


# ─────────────────────────────────────────────
# DETAILED EVALUATION
# ─────────────────────────────────────────────
def detailed_evaluation():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n📊 Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix"); plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# TRAINING HISTORY PLOT
# ─────────────────────────────────────────────
def plot_history():
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], 'b-o', label='Train Loss')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()

    ax2.plot(epochs, history["train_acc"], 'g-o', label='Train Acc')
    ax2.plot(epochs, history["test_acc"],  'r-o', label='Test Acc')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()

    plt.tight_layout()
    plt.savefig("outputs/training_history.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor, class_idx=None):
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = self.model(image_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        pooled_grads = self.gradients.mean(dim=[0, 2, 3])
        activations  = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[i]

        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (heatmap.max() + 1e-8)
        return heatmap, class_idx


def overlay_gradcam(original_pil, heatmap, alpha=0.4):
    img   = np.array(original_pil.resize((224, 224)))
    hmap  = cv2.resize(heatmap, (224, 224))
    hmap  = np.uint8(255 * hmap)
    hmap  = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    hmap  = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
    out   = (1 - alpha) * img + alpha * hmap
    return np.uint8(out)


def run_gradcam_on_samples(num_samples=4):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    gradcam = GradCAM(model, target_layer=model.features[-1])

    test_ds_raw = datasets.ImageFolder(TEST_DIR, transform=transforms.ToTensor())
    indices     = np.random.choice(len(test_ds_raw), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3))
    fig.suptitle("Grad-CAM Visualization", fontsize=16, fontweight='bold')

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    for row, idx in enumerate(indices):
        img_tensor, true_label = test_ds_raw[idx]
        pil_img     = transforms.ToPILImage()(img_tensor)
        norm_tensor = norm(img_tensor)

        heatmap, pred_idx = gradcam.generate(norm_tensor)
        overlay = overlay_gradcam(pil_img, heatmap)

        axes[row, 0].imshow(pil_img)
        axes[row, 0].set_title(f"Original\nTrue: {CLASS_NAMES[true_label]}")
        axes[row, 1].imshow(heatmap, cmap='jet')
        axes[row, 1].set_title("Grad-CAM Heatmap")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"Overlay\nPred: {CLASS_NAMES[pred_idx]}")

        for ax in axes[row]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("outputs/gradcam_results.png", dpi=150)
    plt.show()
    print("✅ Grad-CAM results saved to outputs/gradcam_results.png")


# ─────────────────────────────────────────────
# PREPROCESSING VISUALIZATION  (for professor)
# ─────────────────────────────────────────────
def visualize_preprocessing_pipeline(image_path: str):
    original = Image.open(image_path).convert("RGB")
    steps    = {}

    steps["1. Original"]          = original
    steps["2. Resized (224x224)"] = original.resize((224, 224))
    steps["3. H-Flip (augment)"]  = transforms.RandomHorizontalFlip(p=1.0)(original.resize((224, 224)))
    steps["4. Rotation"]          = transforms.RandomRotation(15)(original.resize((224, 224)))
    steps["5. Color Jitter"]      = transforms.ColorJitter(brightness=0.3, contrast=0.3)(original.resize((224, 224)))

    tensor = transforms.ToTensor()(original.resize((224, 224)))
    steps["6. ToTensor"]          = transforms.ToPILImage()(tensor.mean(0).unsqueeze(0).repeat(3, 1, 1))

    norm_tensor = transforms.Normalize([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])(tensor)
    norm_vis    = norm_tensor - norm_tensor.min()
    norm_vis    = norm_vis / norm_vis.max()
    steps["7. Normalized"]        = transforms.ToPILImage()(norm_vis)

    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3.5))
    fig.suptitle("MRI Preprocessing Pipeline", fontsize=14, fontweight='bold')

    for ax, (title, img) in zip(axes, steps.items()):
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("outputs/preprocessing_pipeline.png", dpi=150)
    plt.show()
    print("✅ Preprocessing pipeline saved to outputs/preprocessing_pipeline.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models",  exist_ok=True)

    train_model()
    plot_history()
    detailed_evaluation()
    run_gradcam_on_samples(num_samples=4)

    sample_image = os.path.join(TEST_DIR, CLASS_NAMES[0],
                                os.listdir(os.path.join(TEST_DIR, CLASS_NAMES[0]))[0])
    visualize_preprocessing_pipeline(sample_image)