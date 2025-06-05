#!/usr/bin/env python3

# Pre-compute features for Neural Networks in NumPy
# Used for both supervised baseline and Channel Charting

import multiprocessing as mp
from tqdm.auto import tqdm
import espargos_0007
import numpy as np
import CRAP

PROCESSES = min(8, mp.cpu_count() // 2) # Mantido como no original

TAP_START = espargos_0007.SUBCARRIER_COUNT // 2 - 4
TAP_STOP = espargos_0007.SUBCARRIER_COUNT // 2 + 8
cov_matrix_shape = espargos_0007.ROW_COUNT * espargos_0007.COL_COUNT
FEATURE_SHAPE = (espargos_0007.TX_COUNT, espargos_0007.ARRAY_COUNT, TAP_STOP - TAP_START, cov_matrix_shape, cov_matrix_shape)

# CHANGED: feature_engineering_worker movida para o nível superior do módulo
# para permitir o "pickling" pelo multiprocessing.
# Adicionamos 'global_shared_data' como um exemplo se você precisar inicializar
# workers com dados grandes/compartilhados, mas a abordagem atual passa dados via fila.
def feature_engineering_worker(todo_queue, output_queue):
    # Esta função agora é de nível superior.
    # O acesso a 'all_datasets' foi removido daqui, pois os dados relevantes
    # serão passados através da 'todo_queue'.
    while True:
        # CHANGED: Espera uma tupla contendo os dados necessários para a tarefa, ou None para sair.
        task_details = todo_queue.get()

        if task_details is None: # Sinal para o worker terminar
            # CHANGED: Coloca None na fila de saída para sinalizar ao processo pai que este worker terminou de processar
            # Este sinal específico de término de worker pode não ser estritamente necessário se o pai conta as tarefas.
            # No entanto, é uma boa prática para saber quando os workers estão inativos.
            # output_queue.put(None) # Removido para simplificar a contagem de tarefas no pai
            break

        # CHANGED: Desempacota os dados da tarefa
        dataset_idx, cluster_idx, tx_idx, csi_fdomain_data, clutter_acquisitions_data = task_details

        csi_fdomain_noclutter = CRAP.remove_clutter(csi_fdomain_data, clutter_acquisitions_data)

        csi_tdomain = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(csi_fdomain_noclutter, axes=-1), axis=-1), axes=-1)[..., TAP_START:TAP_STOP]
        csi_tdomain_flat = np.reshape(csi_tdomain, (csi_tdomain.shape[0], espargos_0007.ARRAY_COUNT, cov_matrix_shape, TAP_STOP - TAP_START))
        result_data = np.einsum("davt,dawt->atvw", np.conj(csi_tdomain_flat), csi_tdomain_flat)

        # CHANGED: Envia o resultado de volta com os identificadores originais
        output_queue.put((dataset_idx, cluster_idx, tx_idx, result_data))


def precompute_features(all_datasets):
    todo_queue = mp.Queue()
    output_queue = mp.Queue()

    # CHANGED: Lógica de inicialização e gerenciamento de processos
    processes = []
    for _ in range(PROCESSES):
        # A função 'feature_engineering_worker' agora é de nível superior.
        # 'args' agora só precisa das filas.
        p = mp.Process(target=feature_engineering_worker, args=(todo_queue, output_queue))
        p.start()
        processes.append(p)

    total_tasks = 0
    for dataset_idx, dataset in enumerate(all_datasets):
        dataset["cluster_features"] = np.zeros((len(dataset["clusters"]),) + FEATURE_SHAPE, dtype=np.complex64)
        for cluster_idx, cluster in enumerate(dataset["clusters"]):
            for tx_idx in range(len(cluster["csi_freq_domain"])):
                total_tasks = total_tasks + 1
                # CHANGED: Prepara os dados específicos para esta tarefa a serem enviados pela fila
                csi_fdomain_task_data = all_datasets[dataset_idx]["clusters"][cluster_idx]["csi_freq_domain"][tx_idx]
                clutter_acquisitions_task_data = all_datasets[dataset_idx]["clutter_acquisitions"][tx_idx]
                task_data_tuple = (dataset_idx, cluster_idx, tx_idx, csi_fdomain_task_data, clutter_acquisitions_task_data)
                todo_queue.put(task_data_tuple)

    print(f"Pre-computing training features for {total_tasks} datapoints in total")
    with tqdm(total=total_tasks) as pbar:
        # CHANGED: Lógica de coleta de resultados
        finished_tasks_count = 0
        while finished_tasks_count < total_tasks:
            # Espera pelo resultado de qualquer worker
            dataset_idx_res, cluster_idx_res, tx_idx_res, res_data = output_queue.get()

            # Armazena o resultado
            all_datasets[dataset_idx_res]["cluster_features"][cluster_idx_res, tx_idx_res] = res_data
            pbar.update(1)
            finished_tasks_count += 1

    # CHANGED: Envia sinais de término para todos os workers
    for _ in range(PROCESSES):
        todo_queue.put(None)

    # CHANGED: Espera que todos os processos workers terminem
    for p in processes:
        p.join()