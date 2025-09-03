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
