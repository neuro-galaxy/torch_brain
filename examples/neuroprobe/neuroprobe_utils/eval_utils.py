"""
================================================================================
Script taken from public Neuroprobe repository:
    https://github.com/insight-neuro/neuroprobe/blob/f40b205ff9f7894a1da9910859aaccbfc5e7cafd/examples/eval_utils.py

Reference:
    Zahorodnii, A., Wang, C., Stankovits, B., Moraitaki, C., Chau, G., Barbu, A.,
    Katz, B., & Fiete, I. R. (2025).
    Neuroprobe: Evaluating Intracranial Brain Responses to Naturalistic Stimuli.
    arXiv:2509.21671 [cs.LG]. https://arxiv.org/abs/2509.2167
================================================================================
"""

import neuroprobe.config as neuroprobe_config

from sklearn.metrics import roc_auc_score
import torch, numpy as np
import time, psutil
import math

verbose = True  # print logs

############## LOGGING ###############


def model_name_from_classifier_type(classifier_type):
    if classifier_type == "linear":
        return "Logistic Regression"
    elif classifier_type == "cnn":
        return "CNN"
    elif classifier_type == "transformer":
        return "Transformer"
    elif classifier_type == "mlp":
        return "MLP"
    else:
        raise ValueError(f"Invalid classifier type: {classifier_type}")


def log(message, priority=0, indent=0):
    max_log_priority = -1 if not verbose else 4
    if priority > max_log_priority:
        return

    current_time = time.strftime("%H:%M:%S")
    gpu_memory_reserved = (
        torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    )
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    print(
        f"[{current_time} gpu {gpu_memory_reserved:04.1f}G ram {ram_usage:05.1f}G] {' '*4*indent}{message}"
    )


############## ELECTRODE SUBSET ###############


def subset_electrodes(subject, lite=False, nano=False):
    all_electrode_labels = subject.electrode_labels
    if lite:
        all_electrode_labels = neuroprobe_config.NEUROPROBE_LITE_ELECTRODES[
            subject.subject_identifier
        ]
    elif nano:
        all_electrode_labels = neuroprobe_config.NEUROPROBE_NANO_ELECTRODES[
            subject.subject_identifier
        ]
    subject.set_electrode_subset(all_electrode_labels)  # Use all electrodes
    return all_electrode_labels


############## DATA PREPROCESSING ###############

from scipy import signal
import numpy as np


def preprocess_stft(
    data,
    sampling_rate=2048,
    preprocess="stft_abs",
    preprocess_parameters={
        "stft": {
            "nperseg": 512,
            "poverlap": 0.75,
            "window": "hann",
            "max_frequency": 150,
            "min_frequency": 0,
        }
    },
):
    was_tensor = isinstance(data, torch.Tensor)
    x = torch.from_numpy(data) if not was_tensor else data

    if len(x.shape) == 2:  # if it is only (n_electrodes, n_samples)
        x = x.unsqueeze(0)
    # data is of shape (batch_size, n_electrodes, n_samples)
    batch_size, n_electrodes, n_samples = x.shape

    # convert to float32 and reshape for STFT
    x = x.to(dtype=torch.float32)
    x = x.reshape(batch_size * n_electrodes, -1)

    # STFT parameters
    nperseg = preprocess_parameters["stft"]["nperseg"]
    poverlap = preprocess_parameters["stft"]["poverlap"]
    noverlap = int(nperseg * poverlap)
    hop_length = nperseg - noverlap

    if preprocess_parameters["stft"]["window"] == "hann":
        window = torch.hann_window(nperseg, device=x.device)
    elif preprocess_parameters["stft"]["window"] == "boxcar":
        window = torch.ones(nperseg, device=x.device)
    else:
        raise ValueError(
            f"Invalid window type: {preprocess_parameters['stft']['window']}"
        )

    max_frequency = preprocess_parameters["stft"]["max_frequency"]
    min_frequency = preprocess_parameters["stft"]["min_frequency"]

    # Compute STFT
    x = torch.stft(
        x,
        n_fft=nperseg,
        hop_length=hop_length,
        win_length=nperseg,
        window=window,
        return_complex=True,
        normalized=False,
        center=True,
    )
    # Get frequency bins
    freqs = torch.fft.rfftfreq(nperseg, d=1.0 / sampling_rate)  # 2048Hz sampling rate
    x = x[:, (freqs >= min_frequency) & (freqs <= max_frequency)]

    if preprocess == "stft_absangle":
        # Split complex values into magnitude and phase
        magnitude = torch.abs(x)
        phase = torch.angle(x)
        # Stack magnitude and phase along a new axis
        x = torch.stack([magnitude, phase], dim=-2)
    elif preprocess == "stft_realimag":
        real = torch.real(x)
        imag = torch.imag(x)
        x = torch.stack([real, imag], dim=-2)
    elif preprocess == "stft_abs":
        x = torch.abs(x)
    else:
        raise ValueError(f"Invalid preprocess type: {preprocess}")

    # Reshape back
    _, n_freqs, n_times = x.shape
    x = x.reshape(batch_size, n_electrodes, n_freqs, n_times)
    x = x.transpose(2, 3)  # (batch_size, n_electrodes, n_timebins, n_freqs)

    # Z-score normalization
    # NOTE: skipping batch norm here because in the regression pipeline, StandardScaler is used anyway,
    # and we would like to avoid batch effects in case input items are processed one by one. TODO: find a better idea here
    # x = x - x.mean(dim=[0, 2], keepdim=True)
    # x = x / (x.std(dim=[0, 2], keepdim=True) + 1e-5)

    return x.numpy() if not was_tensor else x


def downsample(data, fs=2048, downsample_rate=200):
    # Handle both numpy arrays and torch tensors
    was_tensor = isinstance(data, torch.Tensor)
    if was_tensor:
        device = data.device
        data_np = data.cpu().numpy()
    else:
        data_np = data

    # Apply downsampling
    result = signal.resample_poly(data_np, up=downsample_rate, down=fs, axis=-1)

    # Convert back to tensor if input was a tensor
    if was_tensor:
        result = torch.from_numpy(result).to(device)

    return result


def remove_line_noise(data, fs=2048, line_freq=60):
    """Remove line noise (60 Hz and harmonics) from neural data.

    Args:
        data (numpy.ndarray or torch.Tensor): Input voltage data of shape (n_channels, n_samples) or (batch_size, n_channels, n_samples)
        fs (int): Sampling frequency in Hz
        line_freq (int): Fundamental line frequency in Hz (typically 60 Hz in the US)

    Returns:
        numpy.ndarray or torch.Tensor: Filtered data with the same shape as input (same type as input)
    """
    # Handle both numpy arrays and torch tensors
    was_tensor = isinstance(data, torch.Tensor)
    if was_tensor:
        device = data.device
        filtered_data = data.cpu().numpy().copy()
    else:
        filtered_data = data.copy()

    # Define the width of the notch filter (5 Hz on each side)
    bandwidth = 5.0

    # Calculate the quality factor Q
    Q = line_freq / bandwidth

    # Apply notch filters for the fundamental frequency and harmonics
    # We'll filter up to the 5th harmonic (60, 120, 180, 240, 300 Hz)
    for harmonic in range(1, 6):
        harmonic_freq = line_freq * harmonic

        # Skip if the harmonic frequency is above the Nyquist frequency
        if harmonic_freq > fs / 2:
            break

        # Create and apply a notch filter
        b, a = signal.iirnotch(harmonic_freq, Q, fs)

        # Apply the filter along the time dimension
        if filtered_data.ndim == 2:  # (n_channels, n_samples)
            filtered_data = signal.filtfilt(b, a, filtered_data, axis=1)
        elif filtered_data.ndim == 3:  # (batch_size, n_channels, n_samples)
            for i in range(filtered_data.shape[0]):
                filtered_data[i] = signal.filtfilt(b, a, filtered_data[i], axis=1)

    # Convert back to tensor if input was a tensor
    if was_tensor:
        filtered_data = torch.from_numpy(filtered_data).to(device)

    return filtered_data


def laplacian_rereference_neural_data(
    electrode_data, electrode_labels, remove_non_laplacian=True
):
    """
    Rereference the neural data using the laplacian method (subtract the mean of the neighbors, as determined by the electrode labels)
    inputs:
        electrode_data: torch tensor of shape (batch_size, n_electrodes, n_samples) or (n_electrodes, n_samples)
        electrode_labels: list of electrode labels
        remove_non_laplacian: boolean, if True, remove the non-laplacian electrodes from the data; if false, keep them without rereferencing
    outputs:
        rereferenced_data: torch tensor of shape (batch_size, n_electrodes_rereferenced, n_samples) or (n_electrodes_rereferenced, n_samples)
        rereferenced_labels: list of electrode labels of length n_electrodes_rereferenced (n_electrodes_rereferenced could be different from n_electrodes if remove_non_laplacian is True)
    """

    def get_all_laplacian_electrodes(electrode_labels):
        """
        Get all laplacian electrodes for a given subject. This function is originally from
        https://github.com/czlwang/BrainBERT repository (Wang et al., 2023)
        """

        def stem_electrode_name(name):
            # names look like 'O1aIb4', 'O1aIb5', 'O1aIb6', 'O1aIb7'
            # names look like 'T1b2
            found_stem_end = False
            stem, num = [], []
            for c in reversed(name):
                if c.isalpha():
                    found_stem_end = True
                if found_stem_end:
                    stem.append(c)
                else:
                    num.append(c)
            return "".join(reversed(stem)), int("".join(reversed(num)))

        def has_neighbors(stem, stems):
            (x, y) = stem
            return ((x, y + 1) in stems) or ((x, y - 1) in stems)

        def get_neighbors(stem, stems):
            (x, y) = stem
            return [f"{x}{y}" for (x, y) in [(x, y + 1), (x, y - 1)] if (x, y) in stems]

        stems = [stem_electrode_name(e) for e in electrode_labels]
        laplacian_stems = [x for x in stems if has_neighbors(x, stems)]
        electrodes = [f"{x}{y}" for (x, y) in laplacian_stems]
        neighbors = {
            e: get_neighbors(stem_electrode_name(e), stems) for e in electrodes
        }
        return electrodes, neighbors

    # Handle both numpy arrays and torch tensors
    was_tensor = isinstance(electrode_data, torch.Tensor)

    batch_unsqueeze = False
    if len(electrode_data.shape) == 2:
        batch_unsqueeze = True
        if was_tensor:
            electrode_data = electrode_data.unsqueeze(0)
        else:
            electrode_data = electrode_data[np.newaxis, :, :]

    laplacian_electrodes, laplacian_neighbors = get_all_laplacian_electrodes(
        electrode_labels
    )
    laplacian_neighbor_indices = {
        laplacian_electrode_label: [
            electrode_labels.index(neighbor_label) for neighbor_label in neighbors
        ]
        for laplacian_electrode_label, neighbors in laplacian_neighbors.items()
    }

    batch_size, n_electrodes, n_samples = electrode_data.shape
    rereferenced_n_electrodes = (
        len(laplacian_electrodes) if remove_non_laplacian else n_electrodes
    )
    if was_tensor:
        rereferenced_data = torch.zeros(
            (batch_size, rereferenced_n_electrodes, n_samples),
            dtype=electrode_data.dtype,
            device=electrode_data.device,
        )
    else:
        rereferenced_data = np.zeros(
            (batch_size, rereferenced_n_electrodes, n_samples),
            dtype=electrode_data.dtype,
        )

    electrode_i = 0
    original_electrode_indices = []
    for original_electrode_index, electrode_label in enumerate(electrode_labels):
        if electrode_label in laplacian_electrodes:
            rereferenced_data[:, electrode_i] = electrode_data[
                :, electrode_i
            ] - electrode_data[:, laplacian_neighbor_indices[electrode_label]].mean(
                axis=1
            )
            original_electrode_indices.append(original_electrode_index)
            electrode_i += 1
        else:
            if remove_non_laplacian:
                continue  # just skip the non-laplacian electrodes
            else:
                rereferenced_data[:, electrode_i] = electrode_data[:, electrode_i]
                original_electrode_indices.append(original_electrode_index)
                electrode_i += 1

    if batch_unsqueeze:
        if was_tensor:
            rereferenced_data = rereferenced_data.squeeze(0)
        else:
            rereferenced_data = rereferenced_data.squeeze(0)

    return (
        rereferenced_data,
        laplacian_electrodes if remove_non_laplacian else electrode_labels,
        original_electrode_indices,
    )


def preprocess_data(data, electrode_labels, preprocess, preprocess_parameters):
    for preprocess_option in preprocess.split("-"):
        if preprocess_option.lower() in ["stft_absangle", "stft_realimag", "stft_abs"]:
            data = preprocess_stft(
                data,
                preprocess=preprocess_option,
                preprocess_parameters=preprocess_parameters,
            )
        elif preprocess_option.lower() == "remove_line_noise":
            data = remove_line_noise(data)
        elif preprocess_option.lower() == "downsample_200":
            data = downsample(data, downsample_rate=200)
        elif preprocess_option.lower() == "downsample_500":
            data = downsample(data, downsample_rate=500)
        elif preprocess_option.lower() == "laplacian":
            data, electrode_labels, original_electrode_indices = (
                laplacian_rereference_neural_data(
                    data, electrode_labels, remove_non_laplacian=False
                )
            )
    return data


############## CLASSIFICATION ###############


class TransformerClassifier:
    def __init__(
        self,
        random_state=42,
        max_iter=100,
        batch_size=64,
        learning_rate=0.001,
        val_size=0.2,
        tol=1e-4,
        patience=10,
        d_model=64,
        nhead=8,
        dim_feedforward=256,
        dropout=0.1,
        num_layers=3,
    ):
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.tol = tol
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes_ = None
        self.best_val_auroc = 0.0
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_layers = num_layers

    def _create_model(self, input_shape, n_classes):
        class Transformer(torch.nn.Module):
            def __init__(
                self,
                input_shape,
                n_classes,
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                dropout=0.1,
                num_layers=3,
            ):
                super().__init__()
                self.d_model = d_model
                self.nhead = nhead
                self.dim_feedforward = dim_feedforward
                self.dropout = dropout
                self.num_layers = num_layers
                # Assuming input shape is (channels, time) or (channels, freq, time)
                if len(input_shape) == 2:
                    self.input_proj = torch.nn.Linear(
                        input_shape[0], self.d_model
                    )  # Project channels to embedding dim
                    self.pos_encoder = PositionalEncoding(
                        self.d_model, max_len=input_shape[1]
                    )
                    encoder_layer = torch.nn.TransformerEncoderLayer(
                        d_model=self.d_model,
                        nhead=self.nhead,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.dropout,
                        batch_first=True,
                    )
                    self.transformer_encoder = torch.nn.TransformerEncoder(
                        encoder_layer, num_layers=self.num_layers
                    )
                    self.fc = torch.nn.Linear(self.d_model, n_classes)
                else:  # 3D input (channels, freq, time)
                    self.input_proj = torch.nn.Linear(
                        input_shape[0] * input_shape[1], self.d_model
                    )  # Project channels*freq to embedding dim
                    self.pos_encoder = PositionalEncoding(
                        self.d_model, max_len=input_shape[2]
                    )
                    encoder_layer = torch.nn.TransformerEncoderLayer(
                        d_model=self.d_model,
                        nhead=self.nhead,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.dropout,
                        batch_first=True,
                    )
                    self.transformer_encoder = torch.nn.TransformerEncoder(
                        encoder_layer, num_layers=self.num_layers
                    )
                    self.fc = torch.nn.Linear(self.d_model, n_classes)

            def forward(self, x):
                # Reshape input for transformer
                if len(x.shape) == 3:  # (batch, channels, time)
                    x = x.transpose(1, 2)  # (batch, time, channels)
                    x = self.input_proj(x)  # (batch, time, 64)
                else:  # (batch, channels, freq, time)
                    batch_size, channels, freq, time = x.shape
                    x = x.transpose(1, 3)  # (batch, time, channels, freq)
                    x = x.reshape(batch_size, time, channels * freq)
                    x = self.input_proj(x)  # (batch, time, 64)

                # Add positional encoding
                x = self.pos_encoder(x)

                # Apply transformer
                x = self.transformer_encoder(x)

                # Global average pooling over time dimension
                x = x.mean(dim=1)

                # Final classification layer
                x = self.fc(x)
                return x

        class PositionalEncoding(torch.nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer("pe", pe)

            def forward(self, x):
                return x + self.pe[:, : x.size(1)]

        return Transformer(
            input_shape,
            n_classes,
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            num_layers=self.num_layers,
        )

    def fit(self, X, y):
        # Convert to torch tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Create train/val split - take last portion for validation
        val_size = int(self.val_size * len(X))
        train_indices = np.arange(len(X) - val_size)
        val_indices = np.arange(len(X) - val_size, len(X))

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        log(
            f"Training with {len(X_train)} samples, validating with {len(X_val)} samples",
            priority=3,
            indent=2,
        )

        # Create model
        input_shape = X.shape[1:]
        self.model = self._create_model(input_shape, n_classes)
        self.model = self.model.to(self.device)

        # Setup training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_auroc = 0.0
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.max_iter):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i : i + self.batch_size].to(self.device)
                batch_y = y_train[i : i + self.batch_size].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Validation
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                val_outputs = self.model(X_val.to(self.device))
                val_loss_value = criterion(val_outputs, y_val.to(self.device))
                val_loss = val_loss_value.item()

                # Calculate validation AUROC
                val_probs = (
                    torch.nn.functional.softmax(val_outputs, dim=1).cpu().numpy()
                )
                y_val_np = y_val.numpy()

                # Convert to one-hot encoding for AUROC calculation
                y_val_onehot = np.zeros((len(y_val_np), n_classes))
                for i, label in enumerate(y_val_np):
                    y_val_onehot[i, label] = 1

                if n_classes > 2:
                    val_auroc = roc_auc_score(
                        y_val_onehot, val_probs, multi_class="ovr", average="macro"
                    )
                else:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs)

                log(
                    f"Epoch {epoch+1}/{self.max_iter}: Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}",
                    priority=3,
                    indent=2,
                )

                # Check if validation AUROC improved
                if val_auroc > best_val_auroc + self.tol:
                    best_val_auroc = val_auroc
                    best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    patience_counter = 0
                    log(
                        f"New best model saved with val AUROC: {best_val_auroc:.4f}",
                        priority=3,
                        indent=2,
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        log(
                            f"Early stopping triggered after {epoch+1} epochs",
                            priority=3,
                            indent=2,
                        )
                        break

        # Load best model
        self.model.load_state_dict(best_model_state)
        log(
            f"Training complete. Best validation AUROC: {best_val_auroc:.4f}",
            priority=3,
            indent=2,
        )
        return self

    def predict_proba(self, X):
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            X = torch.FloatTensor(X)
            # Process in batches
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i : i + self.batch_size].to(self.device)
                outputs = self.model(batch_X)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


class CNNClassifier:
    def __init__(
        self,
        random_state=42,
        max_iter=100,
        batch_size=128,
        learning_rate=0.0001,
        val_size=0.2,
        tol=1e-4,
        patience=10,
    ):
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.tol = tol
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes_ = None
        self.best_val_auroc = 0.0

    def _create_model(self, input_shape, n_classes):
        class CNN(torch.nn.Module):
            def __init__(self, input_shape, n_classes):
                super().__init__()
                # Assuming input shape is (channels, time) or (channels, freq, time)
                if len(input_shape) == 2:
                    self.conv1 = torch.nn.Conv1d(
                        input_shape[0], 32, kernel_size=3, padding=1
                    )
                    self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1)
                    self.conv3 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1)
                    self.pool = torch.nn.MaxPool1d(2)
                    self.dropout = torch.nn.Dropout(0.5)

                    # Calculate the size after convolutions and pooling
                    conv_output_size = input_shape[1] // 8 * 128

                    self.fc1 = torch.nn.Linear(conv_output_size, 256)
                    self.fc2 = torch.nn.Linear(256, n_classes)

                else:  # 3D input (channels, freq, time)
                    self.conv1 = torch.nn.Conv2d(
                        input_shape[0], 32, kernel_size=3, padding=1
                    )
                    self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
                    self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
                    self.pool = torch.nn.MaxPool2d(2)
                    self.dropout = torch.nn.Dropout(0.5)

                    # Calculate the size after convolutions and pooling
                    conv_output_size = (
                        (input_shape[1] // 8) * (input_shape[2] // 8) * 128
                    )

                    self.fc1 = torch.nn.Linear(conv_output_size, 256)
                    self.fc2 = torch.nn.Linear(256, n_classes)

                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.pool(x)
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = self.relu(self.conv3(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.dropout(x)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x

        return CNN(input_shape, n_classes)

    def fit(self, X, y):
        # Convert to torch tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Create train/val split - take last portion for validation
        val_size = int(self.val_size * len(X))
        train_indices = np.arange(len(X) - val_size)
        val_indices = np.arange(len(X) - val_size, len(X))

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        log(
            f"Training with {len(X_train)} samples, validating with {len(X_val)} samples",
            priority=3,
            indent=2,
        )

        # Create model
        input_shape = X.shape[1:]
        self.model = self._create_model(input_shape, n_classes)
        self.model = self.model.to(self.device)

        # Setup training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_auroc = 0.0
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.max_iter):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i : i + self.batch_size].to(self.device)
                batch_y = y_train[i : i + self.batch_size].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Validation
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                val_outputs = self.model(X_val.to(self.device))
                val_loss_value = criterion(val_outputs, y_val.to(self.device))
                val_loss = val_loss_value.item()

                # Calculate validation AUROC
                val_probs = (
                    torch.nn.functional.softmax(val_outputs, dim=1).cpu().numpy()
                )
                y_val_np = y_val.numpy()

                # Convert to one-hot encoding for AUROC calculation
                y_val_onehot = np.zeros((len(y_val_np), n_classes))
                for i, label in enumerate(y_val_np):
                    y_val_onehot[i, label] = 1

                if n_classes > 2:
                    val_auroc = roc_auc_score(
                        y_val_onehot, val_probs, multi_class="ovr", average="macro"
                    )
                else:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs)

                log(
                    f"Epoch {epoch+1}/{self.max_iter}: Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}",
                    priority=3,
                    indent=2,
                )

                # Check if validation AUROC improved
                if val_auroc > best_val_auroc + self.tol:
                    best_val_auroc = val_auroc
                    best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    patience_counter = 0
                    log(
                        f"New best model saved with val AUROC: {best_val_auroc:.4f}",
                        priority=3,
                        indent=2,
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        log(
                            f"Early stopping triggered after {epoch+1} epochs",
                            priority=3,
                            indent=2,
                        )
                        break

        # Load best model
        self.model.load_state_dict(best_model_state)
        log(
            f"Training complete. Best validation AUROC: {best_val_auroc:.4f}",
            priority=3,
            indent=2,
        )
        return self

    def predict_proba(self, X):
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            X = torch.FloatTensor(X)
            # Process in batches
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i : i + self.batch_size].to(self.device)
                outputs = self.model(batch_X)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


class MLPClassifier:
    def __init__(
        self,
        random_state=42,
        max_iter=100,
        batch_size=200,
        learning_rate=0.00001,
        hidden_dims=[1024, 1024],
        tol=1e-8,
        patience=100,
    ):
        """
        MLP Classifier with configurable hidden layers.

        Args:
            hidden_dims: list of integers specifying hidden layer dimensions.
                        If None or empty list, creates a linear model (no hidden layers).
                        E.g., [128, 64] creates two hidden layers with 128 and 64 units.
        """
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tol = tol
        self.patience = patience
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes_ = None

    def _create_model(self, input_size, n_classes):
        class MLP(torch.nn.Module):
            def __init__(self, input_size, n_classes, hidden_dims):
                super().__init__()
                layers = []

                if len(hidden_dims) == 0:
                    # Linear model (logistic regression)
                    layers.append(torch.nn.Linear(input_size, n_classes))
                else:
                    # MLP with hidden layers
                    prev_dim = input_size
                    for hidden_dim in hidden_dims:
                        layers.append(torch.nn.Linear(prev_dim, hidden_dim))
                        layers.append(torch.nn.ReLU())
                        layers.append(torch.nn.Dropout(0.2))
                        prev_dim = hidden_dim

                    # Output layer
                    layers.append(torch.nn.Linear(prev_dim, n_classes))

                self.network = torch.nn.Sequential(*layers)

            def forward(self, x):
                # Flatten all dimensions except batch
                x = x.view(x.size(0), -1)
                return self.network(x)

        return MLP(input_size, n_classes, self.hidden_dims)

    def fit(self, X, y, X_val=None, y_val=None):
        # Convert to torch tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        if X_val is not None and y_val is not None:
            X_val = torch.FloatTensor(X_val)
            y_val = torch.LongTensor(y_val)

        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        model_type = (
            "Linear" if len(self.hidden_dims) == 0 else f"MLP{self.hidden_dims}"
        )
        log(
            f"Training {model_type} on full dataset with {len(X)} samples",
            priority=3,
            indent=2,
        )

        # Create model - flatten input to get total size
        input_size = np.prod(X.shape[1:])
        self.model = self._create_model(input_size, n_classes)
        self.model = self.model.to(self.device)

        # Setup training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_model_state = None
        patience_counter = 0
        best_val_auroc = 0.0

        for epoch in range(self.max_iter):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for i in range(0, len(X), self.batch_size):
                batch_X = X[i : i + self.batch_size].to(self.device)
                batch_y = y[i : i + self.batch_size].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            train_loss = train_loss / train_total
            train_probs = self.predict_proba(X)
            train_auroc = roc_auc_score(
                y, train_probs[:, 1], multi_class="ovr", average="macro"
            )

            # Calculate val AUROC if val data is provided
            if X_val is not None and y_val is not None:
                all_val_probs = []
                with torch.no_grad():
                    for i in range(0, len(X_val), self.batch_size):
                        batch_X = X_val[i : i + self.batch_size].to(self.device)
                        outputs = self.model(batch_X)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        all_val_probs.append(probs.cpu().numpy())
                val_probs = np.concatenate(all_val_probs, axis=0)
                val_auroc = roc_auc_score(
                    y_val, val_probs[:, 1], multi_class="ovr", average="macro"
                )

            log(
                f"Epoch {epoch+1}/{self.max_iter}: Train loss: {train_loss:.8f}, Train AUROC: {train_auroc:.4f}, Val AUROC: {val_auroc:.4f}",
                priority=3,
                indent=2,
            )

            # Early stopping based on validation AUROC
            if val_auroc > best_val_auroc + self.tol:
                best_val_auroc = val_auroc
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
                log(
                    f"New best model saved with val AUROC: {best_val_auroc:.4f}",
                    priority=3,
                    indent=2,
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log(
                        f"Early stopping triggered after {epoch+1} epochs",
                        priority=3,
                        indent=2,
                    )
                    break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        log(
            f"Training complete. Best validation AUROC: {best_val_auroc:.4f}",
            priority=3,
            indent=2,
        )
        return self

    def predict_proba(self, X):
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            X = torch.FloatTensor(X)
            # Process in batches
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i : i + self.batch_size].to(self.device)
                outputs = self.model(batch_X)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


############## REGION AVERAGING (FOR DS/DM SPLITS) ###############


def get_region_labels(subject):
    """
    subject: BrainTreebankSubject
    returns: np.ndarray of shape (n_channels,)
    """
    return subject.get_all_electrode_metadata()["DesikanKilliany"].to_numpy()


def combine_regions(X_train, X_test, regions_train, regions_test):
    """
    X_train: np.ndarray of shape (n_samples, n_channels_train, n_timebins, d_model) or (n_samples, n_channels_train, n_timesamples)
    X_test: np.ndarray of shape (n_samples, n_channels_test, n_timebins, d_model) or (n_samples, n_channels_test, n_timesamples)
    regions_train: np.ndarray of shape (n_channels_train,)
    regions_test: np.ndarray of shape (n_channels_test,)
    """
    # Find the intersection of regions between train and test
    unique_regions_train = np.unique(regions_train)
    unique_regions_test = np.unique(regions_test)
    common_regions = np.intersect1d(unique_regions_train, unique_regions_test)

    d_model_dimension_unsqueezed = False
    if X_train.ndim == 3:
        # Add a dummy dimension to X_train and X_test for d_model=1
        X_train = X_train[:, :, :, np.newaxis]
        X_test = X_test[:, :, :, np.newaxis]
        d_model_dimension_unsqueezed = True

    n_samples_train, _, n_timebins, d_model = X_train.shape
    n_samples_test = X_test.shape[0]
    n_regions_intersect = len(common_regions)

    # Create new arrays to store region-averaged data
    X_train_regions = np.zeros(
        (n_samples_train, n_regions_intersect, n_timebins, d_model), dtype=X_train.dtype
    )
    X_test_regions = np.zeros(
        (n_samples_test, n_regions_intersect, n_timebins, d_model), dtype=X_test.dtype
    )

    # For each common region, average across all channels with that region label
    for i, region in enumerate(common_regions):
        # Find channels corresponding to this region
        train_mask = regions_train == region
        test_mask = regions_test == region

        # Average across channels with the same region
        X_train_regions[:, i, :, :] = X_train[:, train_mask, :, :].mean(axis=1)
        X_test_regions[:, i, :, :] = X_test[:, test_mask, :, :].mean(axis=1)

    if d_model_dimension_unsqueezed:  # remove the dummy dimension
        X_train_regions = X_train_regions[:, :, :, 0]
        X_test_regions = X_test_regions[:, :, :, 0]

    return X_train_regions, X_test_regions, common_regions
