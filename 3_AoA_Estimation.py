from tqdm.auto import tqdm
import espargos_0007
import cluster_utils
import numpy as np
import CRAP
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# Data Loading and Preprocessing

# Loading all the datasets can take some time...
training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)

all_datasets = training_set_robot + test_set_robot + test_set_human

# Load the pre-computed clutter signatures for each dataset
for dataset in all_datasets:
    dataset['clutter_acquisitions'] = np.load(os.path.join("clutter_channel_estimates", os.path.basename(dataset['filename']) + ".npy"))

# Create a directory to store the output AoA estimates
os.makedirs("aoa_estimates", exist_ok=True)

# Group the data into temporal clusters
for dataset in all_datasets:
    cluster_utils.cluster_dataset(dataset)

# Unitary Root-MUSIC Algorithm Definition
def get_unitary_rootmusic_estimator(chunksize = 32, shed_coeff_ratio = 0):  # `chunksize` is the number of antennas in the array
    
    # The Q matrix is a special Unitary Transformation Matrix. Its purpose is to convert the complex-valued covariance matrix R into a new, real-valued
    # matrix C via the operation: C = real(Q.H @ R @ Q).
    #
    # This transformation has two main benefits for Angle of Arrival (AoA) estimation:
    # 1. It improves performance when dealing with correlated signals, which are common in environments with multipath reflections.
    # 2. It effectively doubles the amount of data ("forward-backward averaging"), leading to a more robust estimate with fewer samples.

    I = np.eye(chunksize // 2)
    J = np.flip(np.eye(chunksize // 2), axis = -1)
    Q = np.asmatrix(np.block([[I, 1.0j * I], [J, -1.0j * J]]) / np.sqrt(2))
    
    def unitary_rootmusic(R):
"""
The actual MUSIC estimator function.
"""
        assert(len(R) == chunksize)
        # Apply the unitary transformation to the covariance matrix R
        C = np.real(Q.H @ R @ Q)

        # Perform eigen-decomposition on the transformed matrix
        eig_val, eig_vec = np.linalg.eigh(C)
        eig_val = eig_val[::-1]
        eig_vec = eig_vec[:,::-1]

        # Separate the noise subspace (En), which is formed by the eigenvectors corresponding to the smallest eigenvalues. We assume 1 signal source
        source_count = 1
        En = eig_vec[:,source_count:]
        # Construct the polynomial matrix from the noise subspace and unitary matrix
        ENSQ = Q @ En @ En.T @ Q.H

        # Calculate the polynomial coefficients by summing the diagonals of the ENSQ matrix.
        # np.trace with an offset sums a specific diagonal
        coeffs = np.asarray([np.trace(ENSQ, offset = diag) for diag in range(1, len(R))])
        coeffs = coeffs[:int(len(coeffs) * (1 - shed_coeff_ratio))]
        
        # Assemble the full, conjugate-symmetric polynomial from the coefficients
        coeffs = np.hstack((coeffs[::-1], np.trace(ENSQ), coeffs.conj()))
        # Find the roots of the polynomial
        roots = np.roots(coeffs)
        # Keep only the roots inside the unit circle
        roots = roots[abs(roots) < 1.0]
        # Find the root closest to the unit circle, which corresponds to the AoA
        largest_root = np.argmax(1 / (1.0 - np.abs(roots)))
        
        # Return the angle of the signal root (the AoA) and its magnitude (confidence)
        return np.angle(roots[largest_root]), np.abs(roots[largest_root])

    return unitary_rootmusic

# Create a MUSIC estimator for a 4-antenna array
umusic = get_unitary_rootmusic_estimator(4)

# Main AoA Estimation Loop
for dataset in tqdm(all_datasets):
    print(f"AoA estimation for dataset: {dataset['filename']}")

    # Initialize lists to store results for each cluster
    dataset['cluster_aoa_angles'] = []
    dataset['cluster_aoa_powers'] = []

    # Iterate through each cluster
    for cluster in tqdm(dataset['clusters']):
        # First, remove clutter from the CSI data for this cluster
        csi_by_transmitter_noclutter = []
        for tx_idx, csi in enumerate(cluster['csi_freq_domain']):
            csi_by_transmitter_noclutter.append(CRAP.remove_clutter(csi, dataset['clutter_acquisitions'][tx_idx]))

        # Calculate the spatial covariance matrix R for each of the 4 receiver arrays
        R = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype = np.complex64)

        for tx_csi in csi_by_transmitter_noclutter:
            # Average the covariance matrices from all transmitters
            R = R + np.einsum("dbrms,dbrns->bmn", tx_csi, np.conj(tx_csi)) / tx_csi.shape[0]

        # Apply the MUSIC algorithm to each of the 4 receiver arrays' covariance matrices
        music_results = [umusic(R[array]) for array in range(R.shape[0])]

        # Convert the electrical angle from MUSIC to a physical AoA in radians and store it
        dataset['cluster_aoa_angles'].append(np.asarray([np.arcsin(angle_power[0] / np.pi) for angle_power in music_results]))
        dataset['cluster_aoa_powers'].append(np.asarray([angle_power[1] for angle_power in music_results]))

    # Convert result lists to NumPy arrays
    dataset['cluster_aoa_angles'] = np.asarray(dataset['cluster_aoa_angles'])
    dataset['cluster_aoa_powers'] = np.asarray(dataset['cluster_aoa_powers'])

# Save AoA Estimation Results
for dataset in all_datasets:
    # Save the computed angles and powers to .npy files for the next script
    dataset_name = os.path.basename(dataset['filename'])
    np.save(os.path.join("aoa_estimates", dataset_name + ".aoa_angles.npy"), np.asarray(dataset["cluster_aoa_angles"]))
    np.save(os.path.join("aoa_estimates", dataset_name + ".aoa_powers.npy"), np.asarray(dataset["cluster_aoa_powers"]))

# Evaluation and Visualizatio
plots_output_dir = "plots_3_AoA_Estimation" 
os.makedirs(plots_output_dir, exist_ok=True)

# Loop through the test sets to create plots
for dataset in tqdm(test_set_robot + test_set_human):

    # Calculate the ideal, ground-truth AoA using geometry
    relative_pos = dataset['cluster_positions'][:,np.newaxis,:] - espargos_0007.array_positions
    normal = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_normalvectors)
    right = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_rightvectors)
    up = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_upvectors)
    # The ideal AoA is the arctangent of the rightward vs. forward distance
    ideal_aoas = np.arctan2(right, normal)
    ideal_eles = -np.arctan2(up, normal)
    dataset['cluster_groundtruth_aoas'] = ideal_aoas

    # Calculate the estimation error
    dataset['cluster_aoa_estimation_errors'] = dataset['cluster_aoa_angles'] - dataset['cluster_groundtruth_aoas']

    Generate plots for each receiver array
    norm = mcolors.Normalize(vmin=-45, vmax=45)
    for b in range(dataset["cluster_aoa_angles"].shape[1]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Map colored by the estimated AoA
        im1 = axes[0].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(dataset["cluster_aoa_angles"][:,b]), norm = norm)
        axes[0].set_title(f"AoA Estimates seen from Array {b}")
        axes[0].set_xlabel("x coordinate in m")
        axes[0].set_ylabel("y coordinate in m")

        # Plot 2: Map colored by the ideal (ground-truth) AoA
        im2 = axes[1].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(dataset["cluster_groundtruth_aoas"][:,b]), norm = norm)
        axes[1].set_title(f"Ideal AoAs seen from Array {b}")
        axes[1].set_xlabel("x coordinate in m")
        axes[1].set_ylabel("y coordinate in m")

        # Plot 3: Map colored by the estimation error
        im2 = axes[2].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(dataset["cluster_aoa_estimation_errors"][:,b]), norm = norm)
        axes[2].set_title(f"AoA Estimation Error seen from Array {b}, MAE = {np.mean(np.abs(np.rad2deg(dataset['cluster_aoa_estimation_errors'][:,b]))):.2f}°")
        axes[2].set_xlabel("x coordinate in m")
        axes[2].set_ylabel("y coordinate in m")

        # Add a colorbar to the figure
        cbar_ax = fig.add_axes([1.00, 0.2, 0.02, 0.6])
        fig.colorbar(im1, cax=cbar_ax)
        
        plt.tight_layout()
        
        safe_dataset_basename = os.path.basename(dataset['filename']).replace(".tfrecords", "")
        
        # Save the figure to a file
        plot_filename = f"aoa_array{b}_{safe_dataset_basename}.png"
        full_plot_path = os.path.join(plots_output_dir, plot_filename)
        
        plt.savefig(full_plot_path)
        plt.close(fig)
        
print(f"Plots for AoA Estimation saved to: {os.path.abspath(plots_output_dir)}")