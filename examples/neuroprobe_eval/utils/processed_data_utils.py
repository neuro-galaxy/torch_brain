"""
Data utility functions for processed data with pre-computed splits.
"""
import os
import h5py
import numpy as np
import neuroprobe.config as neuroprobe_config
from temporaldata import Data


def load_processed_data(processed_data_path, subject_id, trial_id):
    """
    Load processed data from HDF5 file.
    
    Args:
        processed_data_path: Path to processed data directory
        subject_id: Subject ID
        trial_id: Trial ID
    
    Returns:
        Data object from temporaldata
    
    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    file_path = f'{processed_data_path}/sub_{subject_id}_trial{trial_id:03d}.h5'
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Processed data file not found: {file_path}\n"
            f"Expected filename pattern: sub_{{subject_id}}_trial{{trial_id:03d}}.h5"
        )
    with h5py.File(file_path, 'r') as f:
        data = Data.from_hdf5(f)
    return data


def get_processed_folds(data, eval_name, n_folds=None):
    """
    Get folds from processed data (splits are pre-computed).
    
    Args:
        data: Data object from temporaldata
        eval_name: Evaluation name
        n_folds: Number of folds (if None, uses NEUROPROBE_LITE_N_FOLDS)
    
    Returns:
        List of fold dictionaries with 'train_split' and 'test_split' Interval objects
    """
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
    """
    Prepare data arrays from processed fold.
    
    Args:
        fold: Fold dictionary with train_split and test_split
        data: Data object from temporaldata
        preprocessor: Preprocessor instance
        eval_name: Evaluation name
        fold_idx: Fold index
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test) as numpy arrays
    """
    train_split = fold['train_split']
    test_split = fold['test_split']
    
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
        # Calculate dimensions for STFT
        n_raw_samples_train = int(np.unique(train_split.end - train_split.start)[0] * neuroprobe_config.SAMPLING_RATE)
        n_raw_samples_test = int(np.unique(test_split.end - test_split.start)[0] * neuroprobe_config.SAMPLING_RATE)
        
        # Helper functions from notebook
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
        
        # Access STFT config (flat structure, parameters directly in cfg)
        # DictConfig supports .get() method like a dict
        nperseg = preprocessor.cfg.get("nperseg", 512)
        poverlap = preprocessor.cfg.get("poverlap", 0.75)
        min_freq = preprocessor.cfg.get("min_frequency", 0)
        max_freq = preprocessor.cfg.get("max_frequency", 150)
        
        n_timebins_train = _n_timebins_(n_raw_samples_train, nperseg, poverlap, True)
        n_freqs_train = _n_freqs_(neuroprobe_config.SAMPLING_RATE, nperseg, min_freq, max_freq)
        num_samples_train = n_timebins_train * n_freqs_train * sum(split_included_channels_train)
        
        n_timebins_test = _n_timebins_(n_raw_samples_test, nperseg, poverlap, True)
        n_freqs_test = _n_freqs_(neuroprobe_config.SAMPLING_RATE, nperseg, min_freq, max_freq)
        num_samples_test = n_timebins_test * n_freqs_test * sum(split_included_channels_test)
    else:
        # Raw voltage
        num_samples_train = neuroprobe_config.SAMPLING_RATE * sum(split_included_channels_train)
        num_samples_test = neuroprobe_config.SAMPLING_RATE * sum(split_included_channels_test)
    
    # Process training data
    X_train = np.zeros((len(train_split), num_samples_train), dtype=np.float32)
    y_train = np.zeros(len(train_split), dtype=np.int32)
    
    for i in range(len(train_split)):
        data_train = data.slice(train_split.start[i], train_split.end[i])
        # Extract neural data and transpose to (n_electrodes, n_samples)
        neural_data = data_train.seeg_data.data[:, split_included_channels_train].T
        # Preprocess
        preprocessed = preprocessor.preprocess(neural_data, train_electrode_labels)
        if hasattr(preprocessed, 'numpy'):
            preprocessed = preprocessed.numpy()
        elif hasattr(preprocessed, 'float'):
            preprocessed = preprocessed.float().numpy()
        X_train[i, :] = preprocessed.flatten()
        y_train[i] = train_split.label[i]
    
    # Process test data
    X_test = np.zeros((len(test_split), num_samples_test), dtype=np.float32)
    y_test = np.zeros(len(test_split), dtype=np.int32)
    
    for i in range(len(test_split)):
        data_test = data.slice(test_split.start[i], test_split.end[i])
        neural_data = data_test.seeg_data.data[:, split_included_channels_test].T
        preprocessed = preprocessor.preprocess(neural_data, test_electrode_labels)
        if hasattr(preprocessed, 'numpy'):
            preprocessed = preprocessed.numpy()
        elif hasattr(preprocessed, 'float'):
            preprocessed = preprocessed.float().numpy()
        X_test[i, :] = preprocessed.flatten()
        y_test[i] = test_split.label[i]
    
    return X_train, y_train, X_test, y_test

