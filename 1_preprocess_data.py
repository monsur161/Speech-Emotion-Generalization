#seg_project/preprocess_datasets_locally.py
# ===================================================================
# LOCAL SCRIPT: preprocess_datasets_locally.py
# ===================================================================
import os
import librosa
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Configuration ---
# IMPORTANT: Update these paths to match the locations on YOUR LAPTOP
RAVDESS_PATH = "D:/ser_project/ravdess_data/"
CREMA_D_PATH = "D:/ser_project/crema_d_data/AudioWAV/"
OUTPUT_SPECTROGRAM_PATH = "D:/ser_project/processed_spectrograms/"

os.makedirs(OUTPUT_SPECTROGRAM_PATH, exist_ok=True)

# --- List of all files to process ---
all_files = []
# Get RAVDESS files
for root, dirs, files in os.walk(RAVDESS_PATH):
    for file in files:
        if file.endswith('.wav'):
            all_files.append(os.path.join(root, file))
# Get CREMA-D files
for file in os.listdir(CREMA_D_PATH):
    if file.endswith('.wav'):
        all_files.append(os.path.join(CREMA_D_PATH, file))

print(f"Found {len(all_files)} total audio files to process.")

# --- Processing Function for a single file ---
def process_file(file_path):
    try:
        # Load audio
        audio, sr = librosa.load(file_path, res_type='kaiser_fast', duration=3, sr=44100, offset=0.5)
        
        # Create spectrogram
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        db_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        # Save the spectrogram as a NumPy array
        filename = os.path.basename(file_path)
        output_filepath = os.path.join(OUTPUT_SPECTROGRAM_PATH, f"{os.path.splitext(filename)[0]}.npy")
        np.save(output_filepath, db_spectrogram)
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

# --- Main Parallel Execution ---
if __name__ == "__main__":
    # Use all available CPU cores for maximum speed
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, all_files), total=len(all_files)))
    
    success_count = sum(results)
    print(f"\nPreprocessing complete. Successfully processed {success_count}/{len(all_files)} files.")
    print(f"Spectrograms are saved in: {OUTPUT_SPECTROGRAM_PATH}")
