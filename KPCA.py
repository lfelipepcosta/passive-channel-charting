# KPCA.py
import numpy as np
from sklearn.decomposition import KernelPCA

class KPCADenoiser:
    """
    A class to apply Kernel PCA (KPCA) as a denoising technique.

    This class wraps the scikit-learn KernelPCA to facilitate its use in a
    preprocessing pipeline. Denoising is achieved by projecting the data 
    onto the principal components in the feature space and then reconstructing 
    it back into the original space.

    Parameters
    ----------
    n_components : int
        The number of principal components to keep. This is the most critical
        parameter for denoising. A smaller number results in more denoising,
        but can also remove useful information from the signal.
    
    kernel : str, default='rbf'
        The kernel function to be used. Commonly 'rbf', 'poly', or 'sigmoid'.
        'rbf' (Radial Basis Function) is a robust default choice.

    gamma : float, default=None
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If None,
        1/n_features will be used. It controls the influence of a single
        training point.
    
    fit_inverse_transform : bool, default=True
        Enables the ability to reconstruct the data (`inverse_transform`),
        which is essential for denoising.
    """
    def __init__(self, n_components, kernel='rbf', gamma=None):
        # The `fit_inverse_transform=True` flag allows reconstructing the signal in the original space.
        self.kpca = KernelPCA(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            fit_inverse_transform=True,
            remove_zero_eig=True # Removes eigenvectors with a zero eigenvalue
        )
        # print(f"KPCADenoiser initialized with {n_components} components and kernel '{kernel}'.")

    def fit(self, X):
        """
        Fits the KPCA model to the input data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training data. Ideally, this should be a dataset
            representative of the signal you want to preserve.
        """
        # print("Fitting the KPCA model...")
        self.kpca.fit(X)
        # print("Fit complete.")
        return self

    def transform(self, X):
        """
        Applies the denoising process to data X.

        This involves two steps:
        1. Projecting the data into the feature space (transform).
        2. Reconstructing the data back into the original space (inverse_transform).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The noisy data to be cleaned.

        Returns
        -------
        X_denoised : array-like, shape (n_samples, n_features)
            The data after the denoising has been applied.
        """
        # Project the data into the KPCA component space
        X_kpca = self.kpca.transform(X)
        
        # Reconstruct the data back to the original space. This is the denoising step.
        X_denoised = self.kpca.inverse_transform(X_kpca)
        
        return X_denoised

    def fit_transform(self, X):
        """
        Combines fitting and applying the denoising in a single step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to fit the model on and then denoise.

        Returns
        -------
        X_denoised : array-like, shape (n_samples, n_features)
            The data after fitting and denoising.
        """
        return self.fit(X).transform(X)


# --- Usage Example ---
if __name__ == '__main__':
    # This block will only run when you execute `python KPCA.py` directly.
    # It's useful for testing the module.

    print("--- Testing KPCADenoiser Module ---")
    
    # 1. Generate synthetic noisy data
    np.random.seed(0)
    t = np.linspace(0, 5, 100)
    pure_signal = np.sin(t * 3) + np.cos(t * 5) * 0.5
    noise = np.random.normal(0, 0.3, pure_signal.shape)
    noisy_signal = pure_signal + noise
    
    # Scikit-learn expects 2D data (n_samples, n_features)
    X_noisy = noisy_signal.reshape(-1, 1)
    
    # 2. Initialize and use the denoiser
    # Let's keep 2 principal components for demonstration.
    kpca_denoiser = KPCADenoiser(n_components=2, kernel='rbf', gamma=1.0)
    
    # 3. Apply the denoising
    X_denoised = kpca_denoiser.fit_transform(X_noisy)
    
    print("\nOriginal shape:", X_noisy.shape)
    print("Denoised shape:", X_denoised.shape)

    # 4. (Optional) Visualize the results
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(t, pure_signal, 'g-', label='Pure Signal', linewidth=2)
        plt.plot(t, X_noisy.flatten(), 'b.', label='Noisy Signal', alpha=0.5)
        plt.plot(t, X_denoised.flatten(), 'r-', label='Denoised Signal (KPCA)', linewidth=2)
        plt.title('Signal Denoising with KPCA')
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Skipping visualization.")
        print("To view the plot, install matplotlib: pip install matplotlib")