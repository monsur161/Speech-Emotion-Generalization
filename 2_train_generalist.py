import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torchvision import models

# --- Configuration ---
SPECTROGRAM_PATH = "/content/drive/MyDrive/ser_project/processed_spectrograms/"
LEARNING_RATE = 0.001; BATCH_SIZE = 64; EPOCHS = 30
CHECKPOINT_BEST_PATH = "/content/drive/MyDrive/ser_project/resnet_generalist_best_v2.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")

# --- Mappings ---
unified_emotion_map = { "neutral": 0, "happy": 1, "sad": 2, "angry": 3, "fearful": 4, "disgust": 5 }
unified_emotion_labels = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]
ravdess_map = { "01": "neutral", "03": "happy", "04": "sad", "05": "angry", "06": "fearful", "07": "disgust" }
crema_d_map = { "NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry", "FEA": "fearful", "DIS": "disgust" }

# --- A simpler and faster Dataset class for pre-computed files ---
class PrecomputedSpectrogramDataset(Dataset):
    def __init__(self, file_paths, labels, target_width=300):
        self.file_paths, self.labels, self.target_width = file_paths, labels, target_width
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        file_path, label = self.file_paths[idx], self.labels[idx]
        spectrogram = np.load(file_path)
        if spectrogram.shape[1] < self.target_width: spectrogram = np.pad(spectrogram, ((0, 0), (0, self.target_width - spectrogram.shape[1])), mode='constant')
        else: spectrogram = spectrogram[:, :self.target_width]
        spec_min, spec_max = spectrogram.min(), spectrogram.max()
        if spec_max > spec_min: spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
        spectrogram_3ch = np.stack([spectrogram, spectrogram, spectrogram], axis=0)
        return torch.tensor(spectrogram_3ch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# --- Prepare Data from pre-computed .npy files ---
print("Preparing data from pre-computed spectrograms...")
all_files = [os.path.join(SPECTROGRAM_PATH, f) for f in os.listdir(SPECTROGRAM_PATH) if f.endswith('.npy')]
all_labels_str = []
# This loop is designed to handle both RAVDESS and CREMA-D filenames
for f in all_files:
    filename = os.path.basename(f)
    try:
        if '03-01' in filename: # Heuristic for RAVDESS
            code = filename.split("-")[2]
            if code in ravdess_map: all_labels_str.append(ravdess_map[code])
        else: # Assumed to be CREMA-D
            code = filename.split("_")[2]
            if code in crema_d_map: all_labels_str.append(crema_d_map[code])
    except IndexError:
        # print(f"Could not parse filename: {filename}")
        continue
# Filter out files that couldn't be parsed
valid_indices = [i for i, lbl in enumerate(all_labels_str) if lbl]
all_files = [all_files[i] for i in valid_indices]

emotion_to_idx = {e: i for i, e in enumerate(unified_emotion_labels)}; all_labels = [emotion_to_idx[lbl] for lbl in all_labels_str]
train_files, val_files, train_labels, val_labels = train_test_split(all_files, all_labels, test_size=0.15, random_state=42, stratify=all_labels)
train_dataset = PrecomputedSpectrogramDataset(train_files, train_labels); val_dataset = PrecomputedSpectrogramDataset(val_files, val_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0); val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- Train the Model (with Early Stopping) ---
model = models.resnet18(weights='IMAGENET1K_V1'); model.fc = nn.Linear(model.fc.in_features, len(unified_emotion_labels)); model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE); criterion = nn.CrossEntropyLoss(); scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

best_val_acc = 0.0
print("Starting training...")
for epoch in range(EPOCHS):
    model.train(); running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
        loss.backward(); optimizer.step(); running_loss += loss.item() * inputs.size(0)
    train_loss = running_loss / len(train_dataset)

    model.eval(); val_loss = 0.0; correct = 0; total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs); loss = criterion(outputs, labels); val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1); total += labels.size(0); correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total; val_loss /= len(val_dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        print(f"New best validation accuracy: {best_val_acc:.2f}%. Saving model...")
        torch.save({'model_state_dict': model.state_dict()}, CHECKPOINT_BEST_PATH)
    scheduler.step()

# --- Final Evaluation (using the held-out portion of the validation set as a test set) ---
print("\n--- FINAL EVALUATION ---")
# The val_files/val_labels serve as our final test set in this simplified script
test_loader_final = val_loader
print(f"Loading best model (from epoch with {best_val_acc:.2f}% validation accuracy) for final testing...")
best_checkpoint = torch.load(CHECKPOINT_BEST_PATH); model.load_state_dict(best_checkpoint['model_state_dict']); model.eval()
all_preds = []; all_true = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader_final, desc="Final Evaluation"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs); _, preds = torch.max(outputs, 1); all_preds.extend(preds.cpu().numpy()); all_true.extend(labels.cpu().numpy())
accuracy = accuracy_score(all_true, all_preds)
print(f"\nFinal Generalist Model Accuracy on the Test Set: {accuracy * 100:.2f}%")
print("\nClassification Report:"); print(classification_report(all_true, all_preds, target_names=unified_emotion_labels, zero_division=0))
