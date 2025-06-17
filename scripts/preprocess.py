import os
import numpy as np
import mne
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=128):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)

def extract_features(eeg_data):
    """
    Extracts statistical features from EEG channels.
    """
    features = []
    for channel in eeg_data:
        mean = np.mean(channel)
        std = np.std(channel)
        max_val = np.max(channel)
        min_val = np.min(channel)
        features.extend([mean, std, max_val, min_val])
    return np.array(features)

def preprocess_eeg_file(file_path, fs=128):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.pick_types(eeg=True)
    data = raw.get_data()
    filtered_data = apply_bandpass_filter(data, fs=fs)
    features = extract_features(filtered_data)
    return features

if __name__ == "__main__":
    input_dir = "data/raw"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".edf"):
            print(f"Processing {file}...")
            features = preprocess_eeg_file(os.path.join(input_dir, file))
            out_path = os.path.join(output_dir, file.replace(".edf", ".npy"))
            np.save(out_path, features)
