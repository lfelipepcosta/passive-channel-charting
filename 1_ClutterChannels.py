from tqdm.auto import tqdm
import espargos_0007
import numpy as np
import CRAP
import os
import matplotlib
matplotlib.use('Agg')               # Use a non-interactive backend for saving plots to files
import matplotlib.pyplot as plt 

# Loading all the datasets can take some time...
# Load the training and test sets from the espargos-0007 dataset
training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)

# Combine all loaded datasets into a single list for unified processing
all_datasets = training_set_robot + test_set_robot + test_set_human

# Create a directory to store the computed clutter estimates
os.makedirs("clutter_channel_estimates", exist_ok=True)

clutter_acquisitions_by_dataset = dict()

# Loop through each dataset file
for dataset in tqdm(all_datasets):
    print(f"Computing clutter channel for dataset {dataset['filename']}")

    # This list will store the clutter subspace for each transmitter in this dataset
    clutter_acquisitions = []

    # Loop through each unique transmitter MAC address found in the dataset
    # Clutter must be estimated separately for each transmitter
    for tx_idx, mac in enumerate(tqdm(dataset["unique_macs"])):

        # Filter CSI by transmitter index (= transmitter MAC address)
        csi_filtered = []
        for i in range(dataset["csi_freq_domain"].shape[0]):
            if dataset["source_macs"][i] == mac:
                csi_filtered.append(dataset["csi_freq_domain"][i])
    
        # Learn the clutter subspace for this transmitter's data using CRAP
        csi_filtered = np.asarray(csi_filtered)
        clutter_acquisitions.append(CRAP.acquire_clutter(csi_filtered, order = 2))

    # Store the list of clutter subspaces (one per TX) for the current dataset
    clutter_acquisitions_by_dataset[dataset['filename']] = np.asarray(clutter_acquisitions)

    # Save the computed clutter subspaces to a .npy file for caching
    npy_output_dir = "clutter_channel_estimates"
    os.makedirs(npy_output_dir, exist_ok=True) # Adicionado para garantir que o diretório original exista
    for filename, clutter_acquisitions_by_tx in clutter_acquisitions_by_dataset.items():
        np.save(os.path.join(npy_output_dir, os.path.basename(filename)), np.asarray(clutter_acquisitions_by_tx))

# Loop through each dataset file again to apply the removal
for dataset in tqdm(all_datasets):
    print(f"Estimating clutter channel power for dataset {dataset['filename']}")

    # Loop through each unique transmitter
    for tx_idx, mac in enumerate(tqdm(dataset['unique_macs'])):

        # Filter CSI by transmitter index (= transmitter MAC address)
        csi_filtered = []
        for i in range(dataset['csi_freq_domain'].shape[0]):
            if dataset["source_macs"][i] == mac:
                csi_filtered.append(dataset['csi_freq_domain'][i])


        csi_cluttered = np.asarray(csi_filtered)
        # Apply CRAP Phase 2 to remove the clutter
        csi_noclutter = CRAP.remove_clutter(csi_cluttered, clutter_acquisitions_by_dataset[dataset['filename']][tx_idx])

        # Power is proportional to the squared absolute value
        csi_cluttered_power = np.mean(np.sum(np.abs(csi_cluttered)**2, axis = (1, 2, 3, 4)), axis = 0)
        csi_noclutter_power = np.mean(np.sum(np.abs(csi_noclutter)**2, axis = (1, 2, 3, 4)), axis = 0)
        clutter_only_power = csi_cluttered_power - csi_noclutter_power

        print(f"======== Transmitter {tx_idx}: {mac} ========")
        print(f"                Datapoint mean power: {10 * np.log10(csi_cluttered_power):.2f} dB")
        print(f"Datapoint mean power without clutter: {10 * np.log10(csi_noclutter_power):.2f} dB")
        print(f"                       Clutter power: {10 * np.log10(clutter_only_power):.2f} dB")
        print(f" Share of non-clutter power of total: {10 * np.log10(csi_noclutter_power / csi_cluttered_power):.2f} dB")


def channel_plot(csi_datapoint, suptitle = None, title = None, outfile_path = None):
    plt.figure(figsize = (8, 6))
    # Lógica original para suptitle e title
    plt.suptitle("Channel State Information" if suptitle is None else suptitle)
    # Subplot 1: Plot the magnitude of the CSI coefficients
    plt.subplot(211)
    if title is not None:
        plt.title(title)
    plt.xlabel("Subcarrier $i$")
    plt.ylabel("Referenced channel coeff.,\n abs. value $|h_i|_{dB}$ [dB]")
    for b in range(csi_datapoint.shape[0]):
        for r in range(csi_datapoint.shape[1]):
            for c in range(csi_datapoint.shape[2]):
                plt.plot(20 * np.log10(np.abs(csi_datapoint[b,r,c,:])))
    # Subplot 2: Plot the phase of the CSI coefficients
    plt.subplot(212)
    plt.xlabel("Subcarrier $i$")
    plt.ylabel("Referenced channel coeff.,\nphase shift $arg(h_i)$")
    for b in range(csi_datapoint.shape[0]):
        for r in range(csi_datapoint.shape[1]):
            for c in range(csi_datapoint.shape[2]):
                plt.plot(np.unwrap(np.angle(csi_datapoint[b,r,c,:])))

    plt.tight_layout()
    

    if outfile_path:
        plt.savefig(outfile_path)
    plt.close()

# Create a directory to store the output plots
plots_dir = "plots_1_ClutterChannels" 
os.makedirs(plots_dir, exist_ok=True)

# Loop through the computed clutter subspaces to plot them
for filename, clutter_acquisitions_by_tx in clutter_acquisitions_by_dataset.items():
    # Loop through each transmitter's learned clutter subspace
    for tx_idx in range(clutter_acquisitions_by_tx.shape[0]):
        # The clutter subspace is (Q, order). We take the first principal component ([:, 0])
        clutter_basis = np.reshape(clutter_acquisitions_by_tx[tx_idx], (espargos_0007.ARRAY_COUNT, espargos_0007.ROW_COUNT, espargos_0007.COL_COUNT, espargos_0007.SUBCARRIER_COUNT, -1))
        
        plot_suptitle = "Principal Component of Clutter Channel Estimate"
        plot_title = f"Transmitter {tx_idx} in {os.path.basename(filename)}"
        
        # Construção do nome do arquivo de plot
        base_plot_filename = os.path.basename(filename).replace(".tfrecords", "")
        output_image_filename = f"clutter_pc_{base_plot_filename}_tx{tx_idx}.png"
        full_output_plot_path = os.path.join(plots_dir, output_image_filename)
        
        # Call the plotting function to create and save the image
        channel_plot(clutter_basis[:,:,:,:,0], 
                     suptitle=plot_suptitle, 
                     title=plot_title, 
                     outfile_path=full_output_plot_path)

print(f"Plots saved in: {os.path.abspath(plots_dir)}")