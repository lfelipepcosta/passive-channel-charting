#!/usr/bin/env python3

from tqdm.auto import tqdm
import espargos_0007
import numpy as np
import CRAP
import os
import matplotlib # Adicionado
matplotlib.use('Agg') # Adicionado - backend não interativo
import matplotlib.pyplot as plt 

# Loading all the datasets can take some time...
training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)

all_datasets = training_set_robot + test_set_robot + test_set_human

os.makedirs("clutter_channel_estimates", exist_ok=True)

clutter_acquisitions_by_dataset = dict()

for dataset in tqdm(all_datasets):
    print(f"Computing clutter channel for dataset {dataset['filename']}")
    clutter_acquisitions = []
    for tx_idx, mac in enumerate(tqdm(dataset["unique_macs"])): # tqdm original
        # Filter CSI by transmitter index (= transmitter MAC address)
        csi_filtered = []
        for i in range(dataset["csi_freq_domain"].shape[0]):
            if dataset["source_macs"][i] == mac:
                csi_filtered.append(dataset["csi_freq_domain"][i])
    
        csi_filtered = np.asarray(csi_filtered)
        clutter_acquisitions.append(CRAP.acquire_clutter(csi_filtered, order = 2))

    clutter_acquisitions_by_dataset[dataset['filename']] = np.asarray(clutter_acquisitions)


    # Lógica de salvamento original para .npy (dentro do loop do dataset, iterando sobre todo o dicionário)
    # Mantendo o diretório original "clutter_channel_estimates"
    npy_output_dir = "clutter_channel_estimates"
    os.makedirs(npy_output_dir, exist_ok=True) # Adicionado para garantir que o diretório original exista
    for filename, clutter_acquisitions_by_tx in clutter_acquisitions_by_dataset.items():
        np.save(os.path.join(npy_output_dir, os.path.basename(filename)), np.asarray(clutter_acquisitions_by_tx))


for dataset in tqdm(all_datasets):
    print(f"Estimating clutter channel power for dataset {dataset['filename']}")
    for tx_idx, mac in enumerate(tqdm(dataset['unique_macs'])): # tqdm original
        # Filter CSI by transmitter index (= transmitter MAC address)
        csi_filtered = []
        for i in range(dataset['csi_freq_domain'].shape[0]):
            if dataset["source_macs"][i] == mac:
                csi_filtered.append(dataset['csi_freq_domain'][i])
    
        csi_cluttered = np.asarray(csi_filtered)
        csi_noclutter = CRAP.remove_clutter(csi_cluttered, clutter_acquisitions_by_dataset[dataset['filename']][tx_idx])

        csi_cluttered_power = np.mean(np.sum(np.abs(csi_cluttered)**2, axis = (1, 2, 3, 4)), axis = 0)
        csi_noclutter_power = np.mean(np.sum(np.abs(csi_noclutter)**2, axis = (1, 2, 3, 4)), axis = 0)
        clutter_only_power = csi_cluttered_power - csi_noclutter_power

        print(f"======== Transmitter {tx_idx}: {mac} ========")
        print(f"                Datapoint mean power: {10 * np.log10(csi_cluttered_power):.2f} dB")
        print(f"Datapoint mean power without clutter: {10 * np.log10(csi_noclutter_power):.2f} dB")
        print(f"                       Clutter power: {10 * np.log10(clutter_only_power):.2f} dB")
        print(f" Share of non-clutter power of total: {10 * np.log10(csi_noclutter_power / csi_cluttered_power):.2f} dB")


# --- MODIFICAÇÃO da função channel_plot ---
# Adicionado argumento outfile_path
def channel_plot(csi_datapoint, suptitle = None, title = None, outfile_path = None):  # Modificado
    plt.figure(figsize = (8, 6))
    # Lógica original para suptitle e title
    plt.suptitle("Channel State Information" if suptitle is None else suptitle)
    plt.subplot(211)
    if title is not None:
        plt.title(title)
    plt.xlabel("Subcarrier $i$")
    plt.ylabel("Referenced channel coeff.,\n abs. value $|h_i|_{dB}$ [dB]")
    for b in range(csi_datapoint.shape[0]):
        for r in range(csi_datapoint.shape[1]):
            for c in range(csi_datapoint.shape[2]):
                plt.plot(20 * np.log10(np.abs(csi_datapoint[b,r,c,:])))

    plt.subplot(212)
    plt.xlabel("Subcarrier $i$")
    plt.ylabel("Referenced channel coeff.,\nphase shift $arg(h_i)$")
    for b in range(csi_datapoint.shape[0]):
        for r in range(csi_datapoint.shape[1]):
            for c in range(csi_datapoint.shape[2]):
                plt.plot(np.unwrap(np.angle(csi_datapoint[b,r,c,:])))

    plt.tight_layout() # Mantido como original (pode precisar de ajuste para suptitle se sobrepor)
    
    # Salvar em arquivo se o caminho for fornecido
    if outfile_path: # Adicionado
        plt.savefig(outfile_path) # Adicionado
    plt.close() # Adicionado: Fechar a figura para liberar memória
# --- FIM MODIFICAÇÃO da função channel_plot ---


# --- MODIFICAÇÃO: Definir diretório para salvar os plots e garantir que ele exista ---
# Sugestão de um diretório relativo simples para os plots.
# Você pode ajustar "clutter_plots_output" conforme necessário.
plots_dir = "plots_1_ClutterChannels" 
os.makedirs(plots_dir, exist_ok=True) # Adicionado
# --- FIM MODIFICAÇÃO ---


for filename, clutter_acquisitions_by_tx in clutter_acquisitions_by_dataset.items():
    for tx_idx in range(clutter_acquisitions_by_tx.shape[0]):
        clutter_basis = np.reshape(clutter_acquisitions_by_tx[tx_idx], (espargos_0007.ARRAY_COUNT, espargos_0007.ROW_COUNT, espargos_0007.COL_COUNT, espargos_0007.SUBCARRIER_COUNT, -1))
        
        # --- MODIFICAÇÃO: Construir caminho do arquivo de output e chamar channel_plot modificado ---
        # Mantendo a forma original de obter suptitle e title
        plot_suptitle = "Principal Component of Clutter Channel Estimate"
        plot_title = f"Transmitter {tx_idx} in {os.path.basename(filename)}"
        
        # Construção do nome do arquivo de plot
        base_plot_filename = os.path.basename(filename).replace(".tfrecords", "") # Remover extensão se houver
        output_image_filename = f"clutter_pc_{base_plot_filename}_tx{tx_idx}.png"
        full_output_plot_path = os.path.join(plots_dir, output_image_filename)
        
        channel_plot(clutter_basis[:,:,:,:,0], 
                     suptitle=plot_suptitle, 
                     title=plot_title, 
                     outfile_path=full_output_plot_path) # Passar o caminho para salvar
        # --- FIM MODIFICAÇÃO ---

# Adicionado para informar onde os plots foram salvos
print(f"Plots de clutter salvos em: {os.path.abspath(plots_dir)}")