from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.signal as signal
import tqdm
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from brainsets_utils.taxonomy import writing


def process_single_letters(
    session_path: Path,
    processed_folder_path: Optional[Path] = None,
    straight_lines: bool = False,
):
    single_letters_data = loadmat(session_path)

    labels = []
    spike_cubes = []
    train_masks = []
    valid_masks = []
    test_masks = []

    for key in single_letters_data.keys():
        if not key.startswith("neuralActivityCube_"):
            continue

        letter = key[len("neuralActivityCube_") :]
        resolved = False
        found = False
        try:
            resolved = writing.Character[letter]
            found = True
        except:
            pass

        try:
            resolved = writing.Line[letter]
            found = True
        except:
            pass

        assert found

        data = single_letters_data[f"neuralActivityCube_{letter}"]

        spike_cubes.append(data)
        labels += [int(resolved)] * len(data)
        valid_mask = np.arange(len(data)) % 9 == 2
        test_mask = np.arange(len(data)) % 9 == 5
        train_mask = ~valid_mask & ~test_mask
        train_masks.append(train_mask)
        valid_masks.append(valid_mask)
        test_masks.append(test_mask)

    train_masks = np.concatenate(train_masks)
    valid_masks = np.concatenate(valid_masks)
    test_masks = np.concatenate(test_masks)

    assert len(test_masks) == len(labels)

    folds = np.where(train_masks, "train", np.where(valid_masks, "valid", "test"))

    spike_cubes = np.concatenate(spike_cubes, axis=0)

    assert spike_cubes.shape[0] == len(test_masks)
    return spike_cubes, train_masks, valid_masks, test_masks, labels, folds


def smooth_spikes(spike_cubes, sigma=3):
    # Define a convolution kernel
    # 30 ms is 3 time bins
    assert spike_cubes.shape[1] == 201
    kernel = signal.windows.gaussian(7 * sigma, sigma)
    kernel /= kernel.sum()
    smoothed_spikes = signal.convolve(spike_cubes, kernel.reshape((1, -1, 1)), "same")
    return smoothed_spikes


# We reshape the data to be (trials x timepoints) x neurons and perform PCA. We
# truncate the PCA to 15 dimensions.
def pca_spikes(avg_spikes, smoothed_spikes, n_components=15):
    pca = PCA(n_components=n_components)
    X = avg_spikes.reshape(-1, smoothed_spikes.shape[-1])
    pca.fit(X)
    Y = pca.transform(smoothed_spikes.reshape(-1, smoothed_spikes.shape[-1]))
    # Undo the reshaping
    Y = Y.reshape(smoothed_spikes.shape[0], smoothed_spikes.shape[1], -1)
    return Y


def knn_classification(X, labels, train_masks, valid_masks, test_masks, tuned=True):
    X_train = X[train_masks]
    y_train = labels[train_masks]

    X_valid = X[valid_masks]
    y_valid = labels[valid_masks]

    if tuned:
        ks = [2, 4, 6, 8, 10, 12, 14]
        scores = []
        for k in ks:
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(X_train, y_train)
            y_valid_pred = neigh.predict(X_valid)
            scores.append(np.mean(y_valid_pred == y_valid))

        k = ks[np.argmax(scores)]
    else:
        k = 10

    X_test = X[test_masks]
    y_test = labels[test_masks]
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    y_train_pred = neigh.predict(X_train)
    y_pred = neigh.predict(X_test)
    return np.mean(y_train_pred == y_train), np.mean(y_pred == y_test)


def resample_spikes(spike_cube, scale):
    x = np.arange(spike_cube.shape[1])
    x_new = x * scale
    x_new = x_new[x_new <= x.max()]
    f = interp1d(x, spike_cube, kind="linear", axis=1)
    return f(x_new)


def time_warp_knn_classification(
    D, labels, train_masks, valid_masks, test_masks, tuned=True
):
    X_train = D[train_masks, :][:, train_masks]
    y_train = labels[train_masks]

    X_valid = D[valid_masks, :][:, train_masks]
    y_valid = labels[valid_masks]

    if tuned:
        ks = [2, 4, 6, 8, 10, 12, 14]
        scores = []
        for k in ks:
            neigh = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
            neigh.fit(X_train, y_train)
            y_valid_pred = neigh.predict(X_valid)
            scores.append(np.mean(y_valid_pred == y_valid))

        k = ks[np.argmax(scores)]
    else:
        k = 10

    X_test = D[test_masks, :][:, train_masks]
    y_test = labels[test_masks]

    neigh = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
    neigh.fit(X_train, y_train)
    y_train_pred = neigh.predict(X_train)
    y_pred = neigh.predict(X_test)
    return np.mean(y_train_pred == y_train), np.mean(y_pred == y_test)


def main():
    raw_folder_path = Path(
        "/network/scratch/p/patrick.mineault/data/raw/willett_shenoy/handwritingBCIData"
    )
    files = sorted(
        list((Path(raw_folder_path) / "Datasets").glob("*/singleLetters.mat"))
        + list((Path(raw_folder_path) / "Datasets").glob("*/straightLines.mat"))
    )

    all_results = []

    for tuned in [True, False]:
        for file in tqdm.tqdm(files):
            (
                spike_cubes,
                train_masks,
                valid_masks,
                test_masks,
                labels,
                folds,
            ) = process_single_letters(file)

            # Here, spike_cubes has trials x timepoints x neurons as dimensions. The
            # time bins are 10 ms. We smooth the spikes temporally with a Gaussian
            # kernel of 30 ms
            smoothed_spikes = smooth_spikes(spike_cubes, 3)[:, 21:181, :]

            # Also get average spikes
            nlabels = len(np.unique(labels))
            smoothed_spikes_2 = smooth_spikes(spike_cubes, 5)[:, 21:181, :]
            avg_spikes = np.reshape(
                smoothed_spikes_2[train_masks, :],
                (
                    nlabels,
                    -1,
                    smoothed_spikes_2.shape[1],
                    smoothed_spikes_2.shape[2],
                ),
            ).mean(axis=1)

            X = pca_spikes(avg_spikes, smoothed_spikes)

            # Perform k-nearest neighbors classification on the PCA data. Train on
            # the train set and calculate the accuracy on the test set.

            train_acc, test_acc = knn_classification(
                X.reshape((X.shape[0], -1)),
                np.array(labels),
                np.array(train_masks),
                np.array(valid_masks),
                np.array(test_masks),
                tuned=tuned,
            )

            train_acc_no_cv, _ = knn_classification(
                X.reshape((X.shape[0], -1)),
                np.array(labels),
                np.array(labels) >= 0,
                np.array(valid_masks),
                np.array(test_masks),
                tuned=tuned,
            )

            # Take each trial, and resample it using scalar time warping with a scale
            # ranging from .7 to 1.42 with 15 different values.
            scales = np.linspace(0.7, 1.42, 15)

            dmats = []
            for scale in scales:
                Y_ = resample_spikes(X, scale)
                X_ = X[:, : Y_.shape[1], :]
                X_ = X_.reshape((X_.shape[0], -1))
                Y_ = Y_.reshape((Y_.shape[0], -1))
                dmats.append(distance.cdist(X_, Y_, "euclidean"))

            D = np.stack(dmats, axis=0)
            D = np.min(D, axis=0)

            train_acc_tw, test_acc_tw = time_warp_knn_classification(
                D,
                np.array(labels),
                np.array(train_masks),
                np.array(valid_masks),
                np.array(test_masks),
                tuned=tuned,
            )

            train_acc_tw_no_cv, _ = time_warp_knn_classification(
                D,
                np.array(labels),
                np.array(labels) >= 0,
                np.array(valid_masks),
                np.array(test_masks),
                tuned=tuned,
            )

            results = {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_acc_no_cv": train_acc_no_cv,
                "train_acc_tw": train_acc_tw,
                "test_acc_tw": test_acc_tw,
                "train_acc_tw_no_cv": train_acc_tw_no_cv,
                "name": file.parent.parts[-1],
            }

            all_results.append(results)

        # Save the results to disk as csvs using pandas
        df = pd.DataFrame(all_results)
        if tuned:
            df.to_csv("willett_shenoy_baseline_tuned.csv")
        else:
            df.to_csv("willett_shenoy_baseline_untuned.csv")


if __name__ == "__main__":
    main()
