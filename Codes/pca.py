import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# Your preprocessing functions (unchanged)
def decode_class_id(class_id, num_encoders=3, num_modulations=3, num_interleavers=2):
    total_per_encoder = num_modulations * num_interleavers
    class_id -= 1
    encoder_id = class_id // total_per_encoder
    rem = class_id % total_per_encoder
    modulation_id = rem // num_interleavers
    interleaver_id = rem % num_interleavers
    return encoder_id, modulation_id, interleaver_id


def data_preprocessing(path):
    data = np.load(path)
    # real = data['real_parts']  # shape: (samples, 4080)
    # imag = data['imag_parts']  # shape: (samples, 4080)
    # label = data['labels']
    real = data['signals_real']
    imag = data['signals_imag']
    label = data['labels_class']
    data.close()
    # real = np.transpose(real)
    # imag = np.transpose(imag)
    inputs = np.stack((real, imag), axis=-1).astype(np.float32)  # shape: (samples, 4080, 2)
    label = np.transpose(label)
    return inputs, label


def data_loader(path):
    inputs, label = data_preprocessing(path)
    x_train, x_val, y_train, y_val = train_test_split(
        inputs, label, test_size=0.2, stratify=label, random_state=42
    )
    return x_train, y_train, x_val, y_val


def analyze_signal_pca_tsne(npz_path, n_pca=200, n_tsne=2, variance_analysis=True):
    """
    Optimized PCA and t-SNE for 4080x2 signal data

    Parameters:
    n_pca: Number of PCA components (recommended 50-400 for 8160 features)
    n_tsne: Number of t-SNE components (recommended 2-3)
    variance_analysis: Whether to analyze variance explained
    """

    print("Loading 4080×2 signal data...")
    x_train, y_train, x_val, y_val = data_loader(npz_path)

    # Flatten the 4080×2 signal data to 8160 features
    print(f"Original shape: {x_train.shape}")  # Should be (samples, 4080, 2)
    X_train_flat = x_train.reshape(x_train.shape[0], -1)  # (samples, 8160)
    X_val_flat = x_val.reshape(x_val.shape[0], -1)
    y_train_flat = y_train.flatten()
    y_val_flat = y_val.flatten()

    print(f"Flattened shape: {X_train_flat.shape}")  # Should be (samples, 8160)
    print(f"Unique classes: {np.unique(y_train_flat)}")

    # Standardize features (crucial for 8160-dimensional data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)

    # Variance analysis to determine optimal PCA components
    if variance_analysis:
        print("\nAnalyzing optimal PCA components...")
        pca_analysis = PCA()
        pca_analysis.fit(X_train_scaled)

        cumsum_var = np.cumsum(pca_analysis.explained_variance_ratio_)

        # Find components for different variance thresholds
        for threshold in [0.80, 0.90, 0.95, 0.99]:
            n_comp = np.argmax(cumsum_var >= threshold) + 1
            print(f"{threshold * 100}% variance: {n_comp} components")

    # Apply PCA with specified components
    print(f"\nApplying PCA with {n_pca} components...")
    pca = PCA(n_components=n_pca, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"PCA output shape: {X_train_pca.shape}")

    # Apply t-SNE (use PCA result as input for efficiency)
    print(f"\nApplying t-SNE with {n_tsne} components...")
    # For large datasets, use subset for t-SNE computation
    max_samples_tsne = min(5000, X_train_pca.shape[0])
    if X_train_pca.shape[0] > max_samples_tsne:
        indices = np.random.choice(X_train_pca.shape[0], max_samples_tsne, replace=False)
        X_tsne_input = X_train_pca[indices]
        y_tsne_input = y_train_flat[indices]
        print(f"Using subset of {max_samples_tsne} samples for t-SNE")
    else:
        X_tsne_input = X_train_pca
        y_tsne_input = y_train_flat

    # FIXED: Updated t-SNE parameters for newer scikit-learn versions
    tsne = TSNE(
        n_components=n_tsne,
        perplexity=min(30, len(X_tsne_input) // 4),
        learning_rate=200,
        max_iter=1000,  # CHANGED: n_iter -> max_iter
        random_state=42,
        init='pca'
    )

    X_tsne = tsne.fit_transform(X_tsne_input)
    print(f"t-SNE output shape: {X_tsne.shape}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # PCA Plot (using first 2 components)
    scatter1 = axes[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                               c=y_train_flat, cmap='tab10', s=20, alpha=0.7)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    axes[0].set_title(f'PCA: First 2 of {n_pca} Components\n(4080×2 → {n_pca} features)')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Class')

    # t-SNE Plot
    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                               c=y_tsne_input, cmap='tab10', s=20, alpha=0.7)
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    axes[1].set_title(f't-SNE: {n_tsne}D Embedding\n(4080×2 → PCA{n_pca} → t-SNE{n_tsne})')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Class')

    plt.tight_layout()
    plt.show()

    return {
        'X_train_pca': X_train_pca,
        'X_val_pca': X_val_pca,
        'X_tsne': X_tsne,
        'y_train': y_train_flat,
        'y_val': y_val_flat,
        'pca_model': pca,
        'scaler': scaler,
        'explained_variance': pca.explained_variance_ratio_.sum()
    }


def run_multiple_analyses(npz_path):
    """Run analysis with different parameter combinations"""

    # Configuration 1: Conservative (fast, good for initial exploration)
    print("=" * 60)
    print("CONFIGURATION 1: Conservative (Fast)")
    print("=" * 60)
    results_conservative = analyze_signal_pca_tsne(
        npz_path, n_pca=50, n_tsne=2, variance_analysis=True
    )

    # Configuration 2: Moderate (balanced performance/detail)
    print("\n" + "=" * 60)
    print("CONFIGURATION 2: Moderate (Balanced)")
    print("=" * 60)
    results_moderate = analyze_signal_pca_tsne(
        npz_path, n_pca=200, n_tsne=2, variance_analysis=False
    )

    return results_conservative, results_moderate


# Example usage:
if __name__ == "__main__":
    results = analyze_signal_pca_tsne('Matlab/training.npz', n_pca=4080, n_tsne=2)
    print(f"\nAnalysis complete!")
    print(f"PCA explained variance: {results['explained_variance']:.4f}")
    print(f"PCA output shape: {results['X_train_pca'].shape}")
    print(f"t-SNE output shape: {results['X_tsne'].shape}")
