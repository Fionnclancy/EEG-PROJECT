# EEG Motor Imagery Classification using CSP + LDA

## Overview
This project investigates EEG-based classification of motor execution and motor imagery using the EEGBCI MNE dataset.

A Common Spatial Patterns (CSP) + Linear Discriminant Analysis (LDA) pipeline is applied across multiple subjects to evaluate both within-subject and cross-subject performance in predicting the correct stimulus 'applied' from the EEG trace data.

---

## Results

- **Single-subject accuracy (49 subjects) Motor Imagery:** ~ The model gave meaningful predictions for 15/49 Subjects, defined as where mean - std >= 0.5 (aforementioned subjects highlighted as purple in figures). Average prediction accuracy for these subjects ~ 0.75.  
- **Single-subject accuracy (49 subjects) Motor execution :** Meaningful predictions for 19/49 subjects, defined as above, with an average for these subjects also ~ 0.75  

### Execution vs Imagery Multi-subject (Across All Subjects EEG data)
- **Execution:** ~0.60 ± 0.03 
- **Imagery:** ~0.58 ± 0.04 
---

## Visualisations

### Per-Subject Performance
![Intra-Subject Motor Execution](per_subject_execution.png)![Intra-Subject Motor Imagery](per_subject_imagery/diagram.png)

### Execution vs Imagery Comparison
![Multi-Subject Execution vs Imagery](execution_vs_imagery.png)

---

## Key Insights

- EEG classification performance decreases significantly when scaling across subjects as person-to-person EEG responses to stimuli vary heavily and the CSP model does not do well with such varaible patterns so naturally Inter-subject variability is a major limiting factor in generalisation  
- Motor execution produces stronger signals than imagery at the individual level but only by a very small margin, we would have expected it to be larger but again given the large number of subjects used and again under the limitation of CSP. CSP extracts EEG components that maximize variance differences between two conditions, such as left- versus right-hand motor imagery. CSP is highly subject-specific: the spatial filters it computes are tailored to each individual’s brain activity. When applied across a large group of participants (e.g., 40–50 people, in our case 49), the variability between individuals causes these patterns to average out, reducing the clarity of any group-level signal. For this reason, CSP is typically used for single-subject analysis, with classification performed independently for each participant rather than using a single global filter, hence why we saw ~0.58 accuracies and hence why the motor execution was not well differentiated from the motor imagery result.
- Really interested to learn about/try to apply the more modern EEG analysis techniques like Riemannian geometry which proves to be much more efficient at predicting motor imagery by taking into account the shape of the EEG trace rather than just the relative amplitude of the EEG signal for different epochs as we have done here.
- NB: The bandpass filtering was chosen between 8-20/30 Hz as these correspond to the mu and beta rhythms which are the main oscillations linked to the motor cortex. Lower than this will pick up movement artefacts like blinking, higher than this will pick up muscle movement and noise.
- I played around with the n_components in the CSP (number of pairs used), and the sweet spot seemed to be two or three pairs (n=4 or 6), additionally we included some regularization to ensure no overfitting was exhibited.

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