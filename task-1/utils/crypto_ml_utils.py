import os
import requests
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import zlib
import math
from scipy import stats
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import random
import itertools
import seaborn as sns


def get_data(
    owner: str = "pytorch",
    repo: str = "pytorch",
    branch: str = "main",
    num_files: int = 20,
) -> list[bytes]:
    """
    Fetches raw file data chunks from a GitHub repository for specified file types.
    """
    file_chunks = []
    CHUNK_SIZE = 32 * 1024

    api_url = (
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    )
    response = requests.get(api_url)
    response.raise_for_status()
    files = response.json()["tree"]
    target_files = [f for f in files if f["path"].endswith((".txt", ".md", ".py"))]
    target_files.sort(key=lambda x: x["size"], reverse=True)
    target_files = target_files[:num_files]
    for file in target_files:
        file_path = file["path"]
        url = f"https://github.com/{owner}/{repo}/raw/refs/heads/{branch}/{file_path}"
        raw_file = requests.get(url)
        data = raw_file.content
        for i in range(0, len(data), CHUNK_SIZE):
            chunk = data[i : i + CHUNK_SIZE]
            if len(chunk) == CHUNK_SIZE:
                file_chunks.append(chunk)
    return file_chunks


def apply_AES(
    data: list[bytes], mode: str = "ECB", key: bytes = b"Sixteen byte key"
) -> list[bytes]:
    """
    Applies AES encryption to a list of byte chunks using the specified mode and key.
    """
    encypted_data = []
    if mode == "ECB":
        cipher = AES.new(key, AES.MODE_ECB)
    elif mode == "CBC":
        cipher = AES.new(key, AES.MODE_CBC)
    elif mode == "CTR":
        cipher = AES.new(key, AES.MODE_CTR)
    for chunk in data:
        encypted_chunk = cipher.encrypt(chunk)
        encypted_data.append(encypted_chunk)
    return encypted_data


def get_random_compressed_data(target_length: int) -> list[bytes]:
    """
    Generates a list of byte chunks filled with compressed random data.
    """
    TARGET_SIZE = 32 * 1024
    random_chunks = []
    compressor = zlib.compressobj(level=6)
    while len(random_chunks) < target_length:
        data = b""
        compressed = b""
        while len(compressed) < TARGET_SIZE:
            block = os.urandom(1024)
            data += block
            compressed = compressor.compress(data) + compressor.flush(zlib.Z_SYNC_FLUSH)
        random_chunks.append(compressed[:TARGET_SIZE])
    return random_chunks


def entropy_bits(val: bytes) -> float:
    """Calculate Shannon entropy of bits (in bits, not nats)"""
    bits = "".join(f"{byte:08b}" for byte in val)
    ones_p = bits.count("1") / len(bits)
    zeros_p = bits.count("0") / len(bits)

    # Handle edge cases where all bits are 0 or 1
    if ones_p == 0 or zeros_p == 0:
        return 0.0

    # Use log base 2 for bits
    return -ones_p * math.log2(ones_p) - zeros_p * math.log2(zeros_p)


def entropy_ideal_bits(val: None) -> float:
    """Ideal entropy for random bits is 1.0 bit per bit"""
    return 1.0


def entropy_bytes(val: bytes) -> float:
    """Calculate Shannon entropy of bytes (in bits per byte)"""
    if len(val) == 0:
        return 0.0

    counts = Counter(val)
    total = len(val)
    entropy = 0.0

    for count in counts.values():
        if count > 0:
            prob = count / total
            entropy -= prob * math.log2(prob)

    return entropy


def entropy_ideal_bytes(val: None) -> float:
    """Ideal entropy for random bytes is 8.0 bits per byte"""
    return 8.0


def chi_squared_bits(bin_val: bytes) -> tuple[float, float]:
    """Chi-squared test for bit distribution"""
    bits = "".join(f"{byte:08b}" for byte in bin_val)
    ones_p = bits.count("1")
    zeros_p = bits.count("0")
    bits_freq = [ones_p, zeros_p]

    # Use float division for expected frequency
    uniform_freq = [len(bits) / 2, len(bits) / 2]
    chi2, p = stats.chisquare(bits_freq, f_exp=uniform_freq)
    return chi2, p


def chi_squared_bytes(bin_val: bytes) -> float:
    """Chi-squared test for byte distribution"""
    counts = Counter(bin_val)
    total = len(bin_val)
    obs = [counts.get(i, 0) for i in range(256)]
    exp = [total / 256] * 256
    chi2, p = stats.chisquare(obs, f_exp=exp)
    return p


def compressibility(bin_val: bytes) -> float:
    """Measure compressibility (lower is more random)"""
    if len(bin_val) == 0:
        return 0.0
    return len(zlib.compress(bin_val)) / len(bin_val)


def serial_correlation(data: bytes) -> float:
    """Measure correlation between adjacent bytes"""
    if len(data) < 2:
        return 0.0

    x = np.frombuffer(data, dtype=np.uint8)
    x_lagged, x_self = x[:-1], x[1:]
    num = np.sum((x_lagged - x_lagged.mean()) * (x_self - x_self.mean()))
    den = np.sqrt(
        np.sum((x_lagged - x_lagged.mean()) ** 2)
        * np.sum((x_self - x_self.mean()) ** 2)
    )
    return num / den if den != 0 else 0


def fft_flatness(data: bytes) -> float:
    """Measure spectral flatness (closer to 1 is more random)"""
    if len(data) == 0:
        return 0.0

    x = np.frombuffer(data, dtype=np.uint8)
    spectrum = np.abs(np.fft.fft(x - np.mean(x)))
    power = spectrum**2
    geometric_mean = np.exp(np.mean(np.log(power + 1e-12)))
    arithmetic_mean = np.mean(power)
    return geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0


def ecb_block_duplicates(data: bytes, block_size: int = 16) -> float:
    """Measure block duplication (higher indicates patterns)"""
    if len(data) < block_size:
        return 0.0

    blocks = [data[i : i + block_size] for i in range(0, len(data), block_size)]
    total_blocks = len(blocks)
    unique_blocks = len(set(blocks))
    return 1 - (unique_blocks / total_blocks) if total_blocks else 0


def get_features(data: list[bytes]) -> np.ndarray:
    """
    Calculates various randomness features for a list of data chunks.
    """
    features = []
    entropy_bits_features = [entropy_bits(data_val) for data_val in data]
    chisqr_features = [chi_squared_bytes(data_val) for data_val in data]
    compresibility_features = [compressibility(data_val) for data_val in data]
    corr_features = [serial_correlation(data_val) for data_val in data]
    fft_features = [fft_flatness(data_val) for data_val in data]
    ecb_features = [ecb_block_duplicates(data_val) for data_val in data]

    entropy_bits_features = np.array(entropy_bits_features, dtype=np.float32)
    chisqr_features = np.array(chisqr_features, dtype=np.float32)
    compresibility_features = np.array(compresibility_features, dtype=np.float32)
    corr_features = np.array(corr_features, dtype=np.float32)
    fft_features = np.array(fft_features, dtype=np.float32)
    ecb_features = np.array(ecb_features, dtype=np.float32)
    features = np.array(
        [
            entropy_bits_features,
            chisqr_features,
            compresibility_features,
            corr_features,
            fft_features,
            ecb_features,
        ],
        dtype=np.float32,
    )
    return np.swapaxes(features, 0, 1)


def plot_entropy_histogram(
    complete_data: np.ndarray, n_bins: int = 10, figsize: tuple[int, int] = (10, 8)
) -> None:
    """
    Plots a histogram of the entropy values.
    """
    plt.hist(complete_data, bins=n_bins)
    plt.title("Entropy Histogram")
    plt.ylabel("Entropy (bits)")
    plt.figure(figsize=figsize)
    plt.show()


def calculate_stats(data: list[bytes], label: str = "random") -> None:
    """Calculate and print randomness statistics"""
    print(f"\n*********{label}*********")
    print("Stats Mean: ")
    print(
        f"Mean Entropy Bits: {sum([entropy_bits(data_val) for data_val in data])/len(data):.6f}"
    )
    print(f"Mean Entropy Bits (Ideal): {entropy_ideal_bits(None):.6f}")
    print(
        f"Mean Entropy Bytes: {sum([entropy_bytes(data_val) for data_val in data])/len(data):.6f}"
    )
    print(f"Mean Entropy Bytes (Ideal): {entropy_ideal_bytes(None):.6f}")
    print(
        f"Mean Chi-Squared P-val: {sum([chi_squared_bytes(data_val) for data_val in data])/len(data):.6f}"
    )
    print(
        f"Mean Compressibility: {sum([compressibility(data_val) for data_val in data])/len(data):.6f}"
    )
    print(
        f"Mean Serial Correlation: {sum([serial_correlation(data_val) for data_val in data])/len(data):.6f}"
    )
    print(
        f"Mean FFT: {sum([fft_flatness(data_val) for data_val in data])/len(data):.6f}"
    )
    print(
        f"Mean ECB Indicator: {sum([ecb_block_duplicates(data_val) for data_val in data])/len(data):.6f}"
    )

    print("\nStats, Complete Dataset")
    complete_data = b"".join(data)
    print(f"Entropy Bits: {entropy_bits(complete_data):.6f}")
    print(f"Entropy Bits (Ideal): {entropy_ideal_bits(None):.6f}")
    print(f"Entropy Bytes: {entropy_bytes(complete_data):.6f}")
    print(f"Entropy Bytes (Ideal): {entropy_ideal_bytes(None):.6f}")
    print(f"Chi-Squared P-val: {chi_squared_bytes(complete_data):.6f}")
    print(f"Compressibility: {compressibility(complete_data):.6f}")
    print(f"Serial Correlation: {serial_correlation(complete_data):.6f}")
    print(f"FFT: {fft_flatness(complete_data):.6f}")
    print(f"ECB Indicator: {ecb_block_duplicates(complete_data):.6f}")


def fit_describe(
    model,
    encrypted_data: np.ndarray,
    unencrypted_data: np.ndarray,
    label: str = "Random Forest",
    verbose: int = 1,
) -> tuple:
    """
    Fits the model, calculates metrics (accuracy, ROC AUC, TPR@FPR=1%), and prints results.
    """
    accuracy, roc_auc, tpr_fpr, model = fit_data(
        model, encrypted_data, unencrypted_data
    )
    if verbose:
        print("*****************************")
        print(f"Results {label}")
        print(f"Accuracy: {accuracy}")
        print(f"ROC AUC: {roc_auc}")
        print(f"TPR@FPR = 1%: {tpr_fpr}")
    return model, accuracy, roc_auc, tpr_fpr


def tpr_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01
) -> float:
    """
    Calculates the True Positive Rate (TPR) at a specific False Positive Rate (FPR) using ROC analysis.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    tpr_at_target = np.interp(target_fpr, fpr, tpr)
    return tpr_at_target


def fit_data(model, encrypted_data: np.ndarray, unencrypted_data: np.ndarray) -> tuple:
    """
    Trains the model on the data and computes performance metrics.
    """
    encrypted_y = np.ones(shape=(len(encrypted_data),))
    unencypted_y = np.zeros(
        shape=(
            len(
                unencrypted_data,
            )
        )
    )
    X = np.concatenate([encrypted_data, unencrypted_data], axis=0)
    y = np.concatenate([encrypted_y, unencypted_y])
    indexes = np.array(range(len(X)))
    np.random.shuffle(indexes)
    X, y = X[indexes], y[indexes]
    model.fit(X, y)
    y_out = model.predict(X)
    return (
        accuracy_score(y, y_out),
        roc_auc_score(y, y_out),
        tpr_at_fpr(y, y_out),
        model,
    )


def compressed_or_binary_classification_pipeline(
    models: list[tuple],
    owner: str = "pytorch",
    repo: str = "pytorch",
    branch: str = "main",
    num_files: int = 20,
    n_bins: int = 10,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Runs a classification pipeline to distinguish raw data from encrypted/random data.
    Includes data fetching, encryption, feature extraction, training, and testing.
    """
    data = get_data(owner=owner, repo=repo, branch=branch, num_files=num_files)
    ECB_data = apply_AES(data, mode="ECB", key=get_random_bytes(16))
    CBC_data = apply_AES(data, mode="CBC", key=get_random_bytes(16))
    CTR_data = apply_AES(data, mode="CTR", key=get_random_bytes(16))
    random_data = get_random_compressed_data(len(data))

    print("=========Stats==========")
    calculate_stats(data, label="Raw Data")
    calculate_stats(ECB_data, label="AES ECB")
    calculate_stats(CBC_data, label="AES CBC")
    calculate_stats(CTR_data, label="AES CTR")
    calculate_stats(random_data, label="Random Data")

    unencrypted = data
    n_unenc = len(unencrypted)
    encrypted_sources = [ECB_data, CBC_data, CTR_data, random_data]
    encrypted_pool = sum(encrypted_sources, [])
    encrypted = random.sample(encrypted_pool, n_unenc)

    print("\n=========Merged Dataset Stats=========")
    calculate_stats(encrypted, label="Encrypted Data")
    encrypted_features = get_features(encrypted)
    unencrypted_features = get_features(unencrypted)
    entropy_encrypted = encrypted_features[:, 0]
    entropy_unencrypted = unencrypted_features[:, 0]
    plot_entropy_histogram(
        np.concatenate([entropy_encrypted, entropy_unencrypted], axis=0),
        n_bins=n_bins,
        figsize=figsize,
    )

    for model_name, model in models:
        model, _, _, _ = fit_describe(
            model, encrypted_features, unencrypted_features, label=model_name, verbose=1
        )
        test_repo = "tensorflow" if repo == "pytorch" else "pytorch"
        test_branch = "master" if repo == "pytorch" else "main"
        test_data = get_data(
            owner=test_repo, repo=test_repo, branch=test_branch, num_files=num_files
        )
        test_random = get_random_compressed_data(len(test_data))

        test_encrypted_features = get_features(test_random)
        test_unencrypted_features = get_features(test_data)
        accuracy, roc_auc, tpr_fpr, _ = fit_data(
            model, test_encrypted_features, test_unencrypted_features
        )

        print("-----------------------------")
        print(f"Results (TESTING) - {model_name}")
        print(f"Accuracy: {accuracy}")
        print(f"ROC AUC: {roc_auc}")
        print(f"TPR@FPR = 1%: {tpr_fpr}")


def complete_classification_pipeline(
    models: list[tuple],
    owner: str = "pytorch",
    repo: str = "pytorch",
    branch: str = "main",
    num_files: int = 20,
) -> None:
    """
    Runs a multi-class-like classification comparison between different data sources (Raw, AES modes, Random).
    Generates and displays accuracy and ROC AUC heatmaps for all pairwise comparisons.
    """
    data = get_data(owner=owner, repo=repo, branch=branch, num_files=num_files)
    ECB_data = apply_AES(data, mode="ECB", key=get_random_bytes(16))
    CBC_data = apply_AES(data, mode="CBC", key=get_random_bytes(16))
    CTR_data = apply_AES(data, mode="CTR", key=get_random_bytes(16))
    random_data = get_random_compressed_data(len(data))

    datasets = {
        "Raw": data,
        "ECB": ECB_data,
        "CBC": CBC_data,
        "CTR": CTR_data,
        "Random": random_data,
    }

    features = {name: get_features(ds) for name, ds in datasets.items()}
    dataset_names = list(datasets.keys())
    dataset_pairs = list(itertools.combinations(dataset_names, 2))
    model_names = [m[0] for m in models]
    acc_matrix = np.zeros((len(model_names), len(dataset_pairs)))
    auc_matrix = np.zeros((len(model_names), len(dataset_pairs)))

    for j, (ds1_name, ds2_name) in enumerate(dataset_pairs):
        X1 = features[ds1_name]
        X2 = features[ds2_name]

        for i, (model_name, model) in enumerate(models):
            _, acc, roc_auc, _ = fit_describe(
                model,
                X1,
                X2,
                label=f"{model_name}: {ds1_name} vs {ds2_name}",
                verbose=0,
            )
            acc_matrix[i, j] = acc
            auc_matrix[i, j] = roc_auc

    pair_labels = [f"{a} vs {b}" for a, b in dataset_pairs]
    fig, axes = plt.subplots(
        1, 2, figsize=(len(dataset_pairs) * 3, len(model_names) * 2.5)
    )
    sns.heatmap(
        acc_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=pair_labels,
        yticklabels=model_names,
        cmap="Blues",
        ax=axes[0],
    )
    axes[0].set_title("Accuracy Comparison")
    sns.heatmap(
        auc_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=pair_labels,
        yticklabels=model_names,
        cmap="Greens",
        ax=axes[1],
    )
    axes[1].set_title("ROC AUC Comparison")
    plt.tight_layout()
    plt.show()
