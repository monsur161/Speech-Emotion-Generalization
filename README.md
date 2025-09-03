# Speech-Emotion-Generalization
# Improving Domain Generalization in Speech Emotion Recognition

This repository contains the code and findings for an independent research project investigating domain generalization in Speech Emotion Recognition (SER). The project benchmarks classical and deep learning models, identifies the "domain gap" problem between acted and crowd-sourced datasets, and implements a transfer learning solution to create a robust "generalist" model.

## Key Findings

1.  **State-of-the-Art Baseline:** A ResNet18 model fine-tuned on the RAVDESS dataset achieved **~80% accuracy**, establishing a state-of-the-art specialist model.
2.  **Domain Gap Identified:** This specialist model failed to generalize, with accuracy dropping to **~24%** when tested on the CREMA-D dataset.
3.  **Successful Generalization:** A new "generalist" model trained on a combined RAVDESS and CREMA-D dataset successfully solved the problem, achieving **~65% accuracy** on the CREMA-D test set—a **41% improvement** in generalization.
4.  **Bias Analysis:** A detailed analysis revealed a measurable gender bias in the specialist model's performance on the RAVDESS dataset.

## Master Results Table

| Model / Experiment          | Training Data     | Test Data | Accuracy  |
| --------------------------- | ----------------- | --------- | --------- |
| Hybrid SVM                  | RAVDESS           | RAVDESS   | 70.14%    |
| ResNet18 ("Specialist")     | RAVDESS           | RAVDESS   | **79.86%**|
| ResNet18 ("Specialist")     | RAVDESS           | CREMA-D   | 23.85%    |
| **ResNet18 ("Generalist")** | RAVDESS + CREMA-D | CREMA-D   | **64.61%**|

## How to Run

1.  **Setup:** Clone this repository. The required packages are listed in `requirements.txt`.
2.  **Preprocessing:** Run `1_preprocess_data.py` locally to convert the raw audio from the RAVDESS and CREMA-D datasets into spectrograms.
3.  **Training:** Upload the processed spectrograms to a cloud environment (like Google Colab with GPU) and run `2_train_generalist.py` to train the final model.
4.  **Analysis:** Run `3_analyze_results.py` to replicate the gender bias and cross-corpus analyses.

## Detailed Setup & Usage

Follow these steps to replicate the experiments and run the models.

### **Prerequisites**
1. Python 3.9+
2. Git for cloning the repository.
3. (Optional but Recommended) An NVIDIA GPU with CUDA for faster model training.

### **Step 1:** Clone the Repository
First, clone this repository to your local machine and navigate into the directory.

```bash
git clone [https://github.com/your-username/Speech-Emotion-Generalization.git](https://github.com/your-username/Speech-Emotion-Generalization.git)
cd Speech-Emotion-Generalization
```

### **Step 2:** Install Dependencies
All the required Python packages are listed in the ```requirements.txt``` file. Install them using pip:

```bash
pip install -r requirements.txt
```

### **Step 3:** Download and Organize Datasets
This project requires two datasets: **RAVDESS** and **CREMA-D**. You must download them manually and place them in a ```data/``` directory.

1. **Create the ```data``` directory** inside the project folder.

2. **Download RAVDESS:**

      2.1 **Source:** Zenodo (https://zenodo.org/records/1188976)

      2.2 Download the "Audio_Speech_Actors_01-24.zip" file.

      2.3 Extract it to get folders named ```Actor_01```, ```Actor_02```, etc.

      2.4 Place all ```Actor_XX``` folders inside a new folder named ```ravdess```.

3. **Download CREMA-D:**

      3.1 **Source:** Kaggle (https://www.kaggle.com/datasets/ejlok1/cremad)

      3.2 Download the ```AudioWAV``` folder.

      3.3 Place the ```AudioWAV``` folder inside a new folder named ```crema-d```.

After completing these steps, your ```data/``` directory structure must look exactly like this:

```
Speech-Emotion-Generalization/
├── data/
│   ├── ravdess/
│   │   ├── Actor_01/
│   │   │   └── ... (wav files)
│   │   ├── Actor_02/
│   │   └── ...
│   └── crema-d/
│       └── AudioWAV/
│           └── ... (wav files)
├── 1_preprocess_data.py
├── 2_train_generalist.py
├── 3_analyze_results.py
└── README.md
```

### **Step 4:** Run the Experimental Pipeline
The process is broken into three scripts that must be run in order.

1. **Preprocess Data:** Run this script locally. It will process the raw audio files from the ```data/``` directory and convert them into spectrogram images, saving them into a new ```processed_data/``` directory.
```bash
python 1_preprocess_data.py
```

2. **Train the Model:** This step is computationally intensive and is best run on a machine with a GPU (e.g., Google Colab, Kaggle). If using a cloud service, upload the entire ```processed_data/``` directory. This script trains the "generalist" model and saves the final model weights (```.pth``` file) and a results log.
```bash
python 2_train_generalist.py
```

3. **Analyze Results:** This final script uses the saved model and log files to generate the gender bias and cross-corpus evaluation results presented in this study.
```bash
python 3_analyze_results.py
```
