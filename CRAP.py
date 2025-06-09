import numpy as np

# Implementation of the CRAP Algorithm as presented in
# M. Henninger, S. Mandelli, A. Grudnitsky, T. Wild, S. ten Brink: "CRAP: Clutter Removal with Acquisitions Under Phase Noise"
#
# Note that the original CRAP algorithm actually assumes empty channels, but we only have measurements with a target in the channel.

def acquire_clutter(csi_dataset, order = 5):
    """
    Learns the clutter subspace from a CSI dataset for a single transmitter.
    'order' is the assumed clutter order K.
    """

    # Reshape the multi-dimensional CSI data into a 2D matrix C of shape (n_samples, n_features)
    C = np.reshape(csi_dataset, (csi_dataset.shape[0], -1))
    # Compute the (n_features, n_features) autocovariance matrix R, equivalent to C^H * C
    R = np.einsum("na,nb->ab", C, np.conj(C), optimize = True)
    # Perform eigen-decomposition of R (eigenvectors, eigenvalues)
    w, v = np.linalg.eigh(R)

    # Reverse the eigenvectors to sort by descending eigenvalue, then select the top 'order' vectors
    # These vectors form the basis of the clutter subspace
    return v[:,::-1][:,:order]

def remove_clutter(csi_dataset, clutter_subspace):
    """
    Remove clutter from channel measurements (for a single transmitter).
    """

    # Reshape the multi-dimensional CSI data into a 2D matrix h of shape (n_samples, n_features)
    h = np.reshape(csi_dataset, (csi_dataset.shape[0], -1))
    # Project each data vector h onto the clutter subspace to find its clutter component.
    # This implements the mathematical projection: C_hat * (C_hat^H * h).
    clutter = clutter_subspace @ np.einsum("sl,ds->ld", np.conj(clutter_subspace), h)
    # Subtract the projected clutter component from the original signal
    h_noclutter = h - np.transpose(clutter)

    # Reshape the cleaned data back to the original multi-dimensional CSI shape
    return np.reshape(h_noclutter, csi_dataset.shape)