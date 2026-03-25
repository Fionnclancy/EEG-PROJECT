# EEG Motor Imagery Classification using CSP + LDA

## Overview
This project investigates EEG-based classification of motor execution and motor imagery using the EEGBCI MNE dataset.

A Common Spatial Patterns (CSP) + Linear Discriminant Analysis (LDA) pipeline is applied across multiple subjects to evaluate both within-subject and cross-subject performance in predicting the correct stimulus 'applied' from the EEG trace data.

---

## Results

- **Single-subject accuracy (49 subjects) Motor Imagery:** ~ The model gave very meaningful predictions for 15/49 Subjects, defined as where mean - std >= 0.5 (aforementioned subjects highlighted as purple in figures). Average prediction accuracy for these subjects ~ 0.75.  
- **Single-subject accuracy (49 subjects) Motor execution :** Meaningful predictions for 19/49 subjects, defined as above, with an average for these subjects also ~ 0.75  

### Execution vs Imagery Multi-subject (Across All Subjects EEG data)
- **Execution:** ~0.60 ± 0.03 
- **Imagery:** ~0.58 ± 0.04 
---

## Visualisations

### Per-Subject Performance
![Intra-Subject Results](per_subject_execution.png)![Diagram](per_subject_imagery/diagram.png)

### Execution vs Imagery Comparison
![Multi-Subject Execution vs Imagery](execution_vs_imagery.png)

---

## Key Insights

- EEG classification performance decreases significantly when scaling across subjects  
- Inter-subject variability is a major limiting factor in generalisation  
- Motor execution produces stronger signals than imagery at the individual level  
- However, this advantage diminishes when analysing large populations  

---

## Method

- **Dataset:** EEGBCI (PhysioNet)  
- **Preprocessing:** Bandpass filtering (8–30 Hz / 8–20 Hz)   
- **Feature Extraction:** Common Spatial Patterns (CSP)  
- **Classifier:** Linear Discriminant Analysis (LDA)  
- **Evaluation:** Stratified cross-validation  

---

## Usage

Install dependencies:

```bash
pip install mne scikit-learn matplotlib