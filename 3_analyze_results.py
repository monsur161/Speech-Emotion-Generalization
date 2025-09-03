# ===================================================================
# FINAL SCRIPT: Evaluating the Generalist Model on Separate Domains
# ===================================================================
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from torchvision import models

# --- Configuration ---
SPECTROGRAM_PATH = "/content/drive/MyDrive/ser_project/processed_spectrograms/"
CHECKPOINT_BEST_PATH = "/content/drive/MyDrive/ser_project/resnet_generalist_best_v2.pth"
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")

# --- Mappings and Dataset Class (same as before) ---
unified_emotion_labels = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]
class PrecomputedSpectrogramDataset(Dataset):
    def __init__(self, file_paths, labels, target_width=300): #... (rest of the class is the same)
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

# --- Load the Best Generalist Model ---
print("Loading the best generalist model...")
model = models.resnet18(); model.fc = nn.Linear(model.fc.in_features, len(unified_emotion_labels));
best_checkpoint = torch.load(CHECKPOINT_BEST_PATH); model.load_state_dict(best_checkpoint['model_state_dict']);
model = model.to(device)
model.eval()

# --- Prepare the separate Test Sets ---
# We need to recreate the exact same split to get our test set
all_files = [os.path.join(SPECTROGRAM_PATH, f) for f in os.listdir(SPECTROGRAM_PATH) if f.endswith('.npy')]
all_labels_str = []
ravdess_map = { "01": "neutral", "03": "happy", "04": "sad", "05": "angry", "06": "fearful", "07": "disgust" }
crema_d_map = { "NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry", "FEA": "fearful", "DIS": "disgust" }
for f in all_files:
    filename = os.path.basename(f)
    try:
        if '03-01' in filename:
            code = filename.split("-")[2]
            if code in ravdess_map: all_labels_str.append(ravdess_map[code])
        else:
            code = filename.split("_")[2]
            if code in crema_d_map: all_labels_str.append(crema_d_map[code])
    except IndexError: continue
valid_indices = [i for i, lbl in enumerate(all_labels_str) if lbl]
all_files = [all_files[i] for i in valid_indices]
emotion_to_idx = {e: i for i, e in enumerate(unified_emotion_labels)}; all_labels = [emotion_to_idx[lbl] for lbl in all_labels_str]
_, test_files, _, test_labels = train_test_split(all_files, all_labels, test_size=0.1, random_state=42, stratify=all_labels)

# Filter the test set for each dataset
ravdess_test_files = [f for f in test_files if 'RAVDESS' in f.upper()]
ravdess_test_labels = [l for i, l in enumerate(test_labels) if 'RAVDESS' in test_files[i].upper()]
crema_d_test_files = [f for f in test_files if 'CREMA-D' in f.upper() or '10' in os.path.basename(f)] # Heuristic for CREMA-D files
crema_d_test_labels = [l for i, l in enumerate(test_labels) if 'CREMA-D' in test_files[i].upper() or '10' in os.path.basename(test_files[i])]

# --- Run the Evaluations ---
def evaluate(files, labels, name):
    dataset = PrecomputedSpectrogramDataset(files, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_preds, all_true = [], []
    with torch.no_grad():
        for inputs, labs in tqdm(loader, desc=f"Evaluating on {name}"):
            inputs, labs = inputs.to(device), labs.to(device)
            outputs = model(inputs); _, preds = torch.max(outputs, 1); all_preds.extend(preds.cpu().numpy()); all_true.extend(labs.cpu().numpy())
    accuracy = accuracy_score(all_true, all_preds)
    print(f"\n>>> Accuracy on {name}: {accuracy * 100:.2f}%")
    print(f"Classification Report for {name}:"); print(classification_report(all_true, all_preds, target_names=unified_emotion_labels, zero_division=0))

if ravdess_test_files: evaluate(ravdess_test_files, ravdess_test_labels, "RAVDESS Test Set")
if crema_d_test_files: evaluate(crema_d_test_files, crema_d_test_labels, "CREMA-D Test Set")
