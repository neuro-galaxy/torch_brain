"""
Unified data loading utilities for both raw and processed data.
"""
import os
import h5py
import torch
import numpy as np
import neuroprobe.config as neuroprobe_config
import neuroprobe.train_test_splits as neuroprobe_train_test_splits
from neuroprobe.braintreebank_subject import BrainTreebankSubject
from temporaldata import Data


# ============================================================================
# Raw Data Loading
# ============================================================================

def subset_electrodes(subject, lite=False, nano=False):
    """Subset electrodes based on lite/nano flags."""
    all_electrode_labels = subject.electrode_labels
    if lite:
        all_electrode_labels = neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[subject.subject_identifier]
    elif nano:
        all_electrode_labels = neuroprobe_config.NEUROPROBE_NANO_ELECTRODES[subject.subject_identifier]
    subject.set_electrode_subset(all_electrode_labels)
    return all_electrode_labels


def create_folds(subject, trial_id, eval_name, splits_type, cfg):
    """Create train/test folds based on split type."""
    bins_start_before_word_onset_seconds = cfg.get("bins_start_before_word_onset_seconds", 0.5)
    bins_end_after_word_onset_seconds = cfg.get("bins_end_after_word_onset_seconds", 1.5)
    lite = cfg.get("lite", True)
    nano = cfg.get("nano", False)
    binary_tasks = cfg.get("binary_tasks", True)
    
    if splits_type == "WithinSession":
        folds = neuroprobe_train_test_splits.generate_splits_within_session(
            subject, trial_id, eval_name, dtype=torch.float32,
            output_indices=False,
            start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds * neuroprobe_config.SAMPLING_RATE),
            end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds * neuroprobe_config.SAMPLING_RATE),
            lite=lite, nano=nano, binary_tasks=binary_tasks
        )
    elif splits_type == "CrossSession":
        folds = neuroprobe_train_test_splits.generate_splits_cross_session(
            subject, trial_id, eval_name, dtype=torch.float32,
            output_indices=False,
            start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds * neuroprobe_config.SAMPLING_RATE),
            end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds * neuroprobe_config.SAMPLING_RATE),
            lite=lite, binary_tasks=binary_tasks
        )
    else:
        raise ValueError(f"Unknown splits_type: {splits_type}. Must be 'WithinSession' or 'CrossSession'.")
    
    return folds


def get_time_bins(cfg):
    """Get time bin starts and ends based on configuration."""
    only_1second = cfg.get("only_1second", False)
    bins_start_before_word_onset_seconds = cfg.get("bins_start_before_word_onset_seconds", 0.5)
    bins_end_after_word_onset_seconds = cfg.get("bins_end_after_word_onset_seconds", 1.5)
    bin_size_seconds = cfg.get("bin_size_seconds", 0.25)
    bin_step_size_seconds = cfg.get("bin_step_size_seconds", 0.125)
    
    bin_starts = []
    bin_ends = []
    
    if not only_1second:
        for bin_start in np.arange(-bins_start_before_word_onset_seconds, bins_end_after_word_onset_seconds - bin_size_seconds, bin_step_size_seconds):
            bin_end = bin_start + bin_size_seconds
            if bin_end > bins_end_after_word_onset_seconds:
                break
            bin_starts.append(bin_start)
            bin_ends.append(bin_end)
        bin_starts += [-bins_start_before_word_onset_seconds]
        bin_ends += [bins_end_after_word_onset_seconds]
    
    bin_starts += [0]
    bin_ends += [1]
    
    return bin_starts, bin_ends


def prepare_fold_data(fold, preprocessor, bin_start, bin_end, cfg, subject):
    """Prepare data arrays from fold for a specific time window (raw data)."""
    bins_start_before_word_onset_seconds = cfg.get("bins_start_before_word_onset_seconds", 0.5)
    sampling_rate = neuroprobe_config.SAMPLING_RATE
    
    data_idx_from = int((bin_start + bins_start_before_word_onset_seconds) * sampling_rate)
    data_idx_to = int((bin_end + bins_start_before_word_onset_seconds) * sampling_rate)
    
    train_dataset = fold["train_dataset"]
    test_dataset = fold["test_dataset"]
    electrode_labels = subject.electrode_labels
    
    # Process training data
    X_train_list = []
    y_train_list = []
    for item in train_dataset:
        data = item[0][:, data_idx_from:data_idx_to].unsqueeze(0)
        preprocessed = preprocessor.preprocess(data, electrode_labels)
        if isinstance(preprocessed, torch.Tensor):
            preprocessed = preprocessed.float().numpy()
        X_train_list.append(preprocessed)
        y_train_list.append(item[1])
    
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.array(y_train_list)
    
    # Process test data
    X_test_list = []
    y_test_list = []
    for item in test_dataset:
        data = item[0][:, data_idx_from:data_idx_to].unsqueeze(0)
        preprocessed = preprocessor.preprocess(data, electrode_labels)
        if isinstance(preprocessed, torch.Tensor):
            preprocessed = preprocessed.float().numpy()
        X_test_list.append(preprocessed)
        y_test_list.append(item[1])
    
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.array(y_test_list)
    
    return X_train, y_train, X_test, y_test


def prepare_for_model(X_train, X_test, model_name):
    """Prepare data for model (flatten for sklearn models, keep shape for torch models)."""
    if model_name == "logistic":
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    return X_train, X_test


# ============================================================================
# Processed Data Loading
# ============================================================================

def load_processed_data(processed_data_path, subject_id, trial_id):
    """Load processed data from HDF5 file. Returns both data and file handle.
    
    Note: Caller is responsible for closing the file handle.
    """
    file_path = f'{processed_data_path}/sub_{subject_id}_trial{trial_id:03d}.h5'
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Processed data file not found: {file_path}\n"
            f"Expected filename pattern: sub_{{subject_id}}_trial{{trial_id:03d}}.h5"
        )
    
    f = h5py.File(file_path, 'r')
    data = Data.from_hdf5(f, lazy=True)
    return data, f


def get_processed_folds(data, eval_name, n_folds=None):
    """Get folds from processed data (splits are pre-computed)."""
    if n_folds is None:
        n_folds = neuroprobe_config.NEUROPROBE_LITE_N_FOLDS
    
    folds = []
    for fold_idx in range(n_folds):
        train_split = getattr(data, f'{eval_name}_fold{fold_idx}_train')
        test_split = getattr(data, f'{eval_name}_fold{fold_idx}_test')
        folds.append({
            'train_split': train_split,
            'test_split': test_split,
            'fold_idx': fold_idx
        })
    
    return folds


def prepare_processed_fold_data(fold, data, preprocessor, eval_name, fold_idx):
    """Prepare data arrays from processed fold.
    
    Splits test_split into validation (first half by start time) and test (second half).
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test) as numpy arrays
    """
    train_split = fold['train_split']
    test_split = fold['test_split']
    
    # Sort test_split by start time and split into val (first half) and test (second half)
    test_start_times = test_split.start
    sorted_indices = np.argsort(test_start_times)
    n_test_samples = len(test_split)
    n_val_samples = n_test_samples // 2
    val_indices = sorted_indices[:n_val_samples]
    test_indices = sorted_indices[n_val_samples:]
    
    # Create val_split and test_split by selecting indices
    # We'll work with indices directly when processing
    
    # Get included channels
    split_included_channels_train = getattr(
        data.channels, f'included_{eval_name}_fold{fold_idx}_train'
    )
    split_included_channels_test = getattr(
        data.channels, f'included_{eval_name}_fold{fold_idx}_test'
    )
    
    # Get electrode labels
    train_electrode_labels = data.channels.name[split_included_channels_train].tolist()
    test_electrode_labels = data.channels.name[split_included_channels_test].tolist()
    
    # Calculate expected dimensions for preprocessed data
    preprocess_type = preprocessor.cfg.name
    if 'stft' in preprocess_type:
        n_raw_samples_train = int(np.unique(train_split.end - train_split.start)[0] * neuroprobe_config.SAMPLING_RATE)
        n_raw_samples_test = int(np.unique(test_split.end - test_split.start)[0] * neuroprobe_config.SAMPLING_RATE)
        
        def _n_timebins_(n_samples, nperseg, poverlap, center=True):
            hop_length = int(nperseg * (1 - poverlap))
            if center:
                n_samples_eff = n_samples + 2 * (nperseg // 2)
            else:
                n_samples_eff = n_samples
            n_timebins = np.floor((n_samples_eff - nperseg) / hop_length) + 1
            return int(n_timebins)
        
        def _n_freqs_(sampling_rate, nperseg, fmin, fmax):
            freqs = np.fft.rfftfreq(nperseg, d=1.0 / sampling_rate)
            mask = (freqs >= fmin) & (freqs <= fmax)
            return int(mask.sum())
        
        nperseg = preprocessor.cfg.get("nperseg", 512)
        poverlap = preprocessor.cfg.get("poverlap", 0.75)
        min_freq = preprocessor.cfg.get("min_frequency", 0)
        max_freq = preprocessor.cfg.get("max_frequency", 150)
        
        n_timebins_train = _n_timebins_(n_raw_samples_train, nperseg, poverlap, True)
        n_freqs_train = _n_freqs_(neuroprobe_config.SAMPLING_RATE, nperseg, min_freq, max_freq)
        num_samples_train = n_timebins_train * n_freqs_train * sum(split_included_channels_train)
        
        n_timebins_test = _n_timebins_(n_raw_samples_test, nperseg, poverlap, True)
        n_freqs_test = _n_freqs_(neuroprobe_config.SAMPLING_RATE, nperseg, min_freq, max_freq)
        num_samples_val = n_timebins_test * n_freqs_test * sum(split_included_channels_test)
        num_samples_test = n_timebins_test * n_freqs_test * sum(split_included_channels_test)
    else:
        num_samples_train = neuroprobe_config.SAMPLING_RATE * sum(split_included_channels_train)
        num_samples_val = neuroprobe_config.SAMPLING_RATE * sum(split_included_channels_test)
        num_samples_test = neuroprobe_config.SAMPLING_RATE * sum(split_included_channels_test)
    
    # Process training data
    X_train = np.zeros((len(train_split), num_samples_train), dtype=np.float32)
    y_train = np.zeros(len(train_split), dtype=np.int32)
    
    for i in range(len(train_split)):
        data_train = data.slice(train_split.start[i], train_split.end[i])
        neural_data = data_train.seeg_data.data[:, split_included_channels_train].T
        preprocessed = preprocessor.preprocess(neural_data, train_electrode_labels)
        if hasattr(preprocessed, 'numpy'):
            preprocessed = preprocessed.numpy()
        elif hasattr(preprocessed, 'float'):
            preprocessed = preprocessed.float().numpy()
        X_train[i, :] = preprocessed.flatten()
        y_train[i] = train_split.label[i]
    
    # Process validation data (first half of test_split by start time)
    X_val = np.zeros((len(val_indices), num_samples_val), dtype=np.float32)
    y_val = np.zeros(len(val_indices), dtype=np.int32)
    
    for idx, i in enumerate(val_indices):
        data_val = data.slice(test_split.start[i], test_split.end[i])
        neural_data = data_val.seeg_data.data[:, split_included_channels_test].T
        preprocessed = preprocessor.preprocess(neural_data, test_electrode_labels)
        if hasattr(preprocessed, 'numpy'):
            preprocessed = preprocessed.numpy()
        elif hasattr(preprocessed, 'float'):
            preprocessed = preprocessed.float().numpy()
        X_val[idx, :] = preprocessed.flatten()
        y_val[idx] = test_split.label[i]
    
    # Process test data (second half of test_split by start time)
    X_test = np.zeros((len(test_indices), num_samples_test), dtype=np.float32)
    y_test = np.zeros(len(test_indices), dtype=np.int32)
    
    for idx, i in enumerate(test_indices):
        data_test = data.slice(test_split.start[i], test_split.end[i])
        neural_data = data_test.seeg_data.data[:, split_included_channels_test].T
        preprocessed = preprocessor.preprocess(neural_data, test_electrode_labels)
        if hasattr(preprocessed, 'numpy'):
            preprocessed = preprocessed.numpy()
        elif hasattr(preprocessed, 'float'):
            preprocessed = preprocessed.float().numpy()
        X_test[idx, :] = preprocessed.flatten()
        y_test[idx] = test_split.label[i]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

