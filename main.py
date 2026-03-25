import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.datasets import eegbci
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold

subject = range(1,50)
runs = [3,7,11]
raws = []
subject_accuracies = []
subject_stds = []

for sub in subject:
    files = eegbci.load_data(sub, runs)
    raws = [mne.io.read_raw_edf(f, preload=True) for f in files]
    raw = mne.concatenate_raws(raws)
    

    #print(raw.ch_names)
    # for motor_chs: T0 -> T2 down pulse, T0 -> T1 up pulse
    # only initial half of somatosentory follows same trend use first half of the data

    somatosensory_chs = ['C3..', 'Cz..', 'C4..', 'Cp3.', 'Cpz.', 'Cp4.']
    events, _ = mne.events_from_annotations(raw)
    event_id = {'T1': 2, 'T2': 3}
    raw.filter(8.,30., fir_design='firwin')
    motor_chs = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
    'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
    'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.'
    ]
    epochs = mne.Epochs(raw, events, event_id, tmin=0.5, tmax=3.5, baseline=None, preload=True)
    epochs.pick_channels(motor_chs)
    print(events[:10])
    #raw.plot(picks=motor_chs, n_channels=len(motor_chs), title='Motor Cortex', block=False)
    #raw.plot(picks=somatosensory_chs, n_channels=len(somatosensory_chs), title='Somatosensory Cortex', block=True)
    X = epochs.get_data()
    y = epochs.events[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y)  # 2->0, 3->1


    clf = Pipeline([
        ('csp', CSP(n_components=4, reg=0.1, log=True)),
        ('lda', LinearDiscriminantAnalysis())
    ])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    scores = cross_val_score(clf, X, y, cv=cv)

    mean_acc = scores.mean()
    std_acc = scores.std()

    subject_accuracies.append(mean_acc)
    subject_stds.append(std_acc)

    print(f"Subject {sub}: Accuracy = {mean_acc:.2f} ± {std_acc:.2f}")

plt.figure(figsize=(10,5))
colors = ['#b452cd' if mean - std >= 0.5 else 'skyblue' 
          for mean, std in zip(subject_accuracies, subject_stds)]
plt.bar(range(1, len(subject_accuracies)+1), subject_accuracies, yerr=subject_stds, capsize=5, color=colors, alpha=0.7)
plt.xticks(range(1, len(subject_accuracies)+1), rotation=90, fontsize=6)
plt.ylim(0, 1)
plt.axhline(0.5, color='red', linestyle='--', label='Random Guessing')
plt.xlabel('Subject')
plt.ylabel('Accuracy')
plt.title('Per-subject EEG Classification Accuracy for LH vs RH (Motor Execution)')
plt.show()