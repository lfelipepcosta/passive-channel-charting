import multiprocessing as mp
import os
import espargos_0007
import cluster_utils
import CRAP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np
from scipy.sparse.csgraph import dijkstra 
from sklearn.neighbors import NearestNeighbors, kneighbors_graph 
from tqdm.auto import tqdm
import scipy.special 

os.makedirs("dissimilarity_matrices", exist_ok=True)

def aoa_power_to_kappa(aoa_power):
    return 5 * np.array(aoa_power)**6

def get_likelihood_function(aoa_datapoint, aoa_power_datapoint):
    kappas = aoa_power_to_kappa(aoa_power_datapoint)
    def likelihood_func(pos):
        relative_pos = pos[:,np.newaxis,:] - espargos_0007.array_positions
        normal = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_normalvectors)
        right = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_rightvectors)
        ideal_aoas = np.arctan2(right, normal)
        
        i0e_kappas = scipy.special.i0e(kappas)
        safe_i0e_kappas = np.where(i0e_kappas == 0, np.finfo(float).eps, i0e_kappas)
        log_exp_term = kappas * (np.cos(ideal_aoas - aoa_datapoint) - 1) 
        log_exp_term = np.maximum(log_exp_term, -700)
        aoa_likelihoods_stable = (1.0 / (2 * np.pi * safe_i0e_kappas)) * np.exp(log_exp_term)
        
        return np.prod(aoa_likelihoods_stable, axis = -1)
    return likelihood_func

def adp_dissimilarities_worker(todo_queue, output_queue, training_csi_reflected_data):
    def adp_dissimilarities(index):
        h = training_csi_reflected_data[index,:,:,:]
        w = training_csi_reflected_data[index:,:,:,:]
        dotproducts = np.abs(np.einsum("brm,lbrm->lb", np.conj(h), w, optimize = "optimal"))**2
        h_norm_sq_per_array = np.real(np.einsum("brm,brm->b", h, np.conj(h), optimize = "optimal"))
        w_norm_sq_per_array = np.real(np.einsum("lbrm,lbrm->lb", w, np.conj(w), optimize = "optimal"))
        norms = h_norm_sq_per_array[np.newaxis, :] * w_norm_sq_per_array
        norms_safe = norms + 1e-12 
        dissimilarity_values = 1 - dotproducts / norms_safe
        return np.sum(dissimilarity_values, axis = (1))

    while True:
        index = todo_queue.get()
        if index == -1:
            break       
        output_queue.put((index, adp_dissimilarities(index)))

def shortest_path_worker(todo_queue, output_queue, nbg_graph_data):
    while True:
        index = todo_queue.get()
        if index == -1:
            break       
        d = dijkstra(nbg_graph_data, directed = False, indices = index)
        output_queue.put((index, d))


def plot_dissimilarity_over_euclidean_distance(input_dissimilarity_matrix, input_distance_matrix, label=None):
    nth_reduction = 10
    
    dissimilarities_flat = input_dissimilarity_matrix[::nth_reduction, ::nth_reduction].flatten()
    distances_flat = input_distance_matrix[::nth_reduction, ::nth_reduction].flatten()

    valid_mask = np.isfinite(dissimilarities_flat) & np.isfinite(distances_flat)
    dissimilarities_flat = dissimilarities_flat[valid_mask]
    distances_flat = distances_flat[valid_mask]

    if distances_flat.size == 0:
        print(f"WARN: No valid points to plot (plot_dissimilarity_over_euclidean_distance) for label: {label}")
        return

    max_distance = np.max(distances_flat) 
    bins = np.linspace(0, max_distance, 200) 
    if len(bins) < 2: return

    bin_indices = np.digitize(distances_flat, bins)

    bin_medians = np.zeros(len(bins) - 1) 
    bin_25_perc = np.zeros(len(bins) - 1)
    bin_75_perc = np.zeros(len(bins) - 1)
    
    for i in range(1, len(bins)): 
        bin_values = dissimilarities_flat[bin_indices == i]
        if bin_values.size > 0: 
            bin_25_perc[i - 1], bin_medians[i - 1], bin_75_perc[i - 1] = np.percentile(bin_values, [25, 50, 75])
        else:
            bin_25_perc[i-1], bin_medians[i-1], bin_75_perc[i-1] = np.nan, np.nan, np.nan
            
    valid_plot_mask = ~np.isnan(bin_medians)
    plot_bins = bins[:-1][valid_plot_mask]
    
    if plot_bins.size > 0:
        plt.plot(plot_bins, bin_medians[valid_plot_mask], label=label)
        plt.fill_between(plot_bins, bin_25_perc[valid_plot_mask], bin_75_perc[valid_plot_mask], alpha=0.5)


if __name__ == '__main__':
    mp.freeze_support()

    plots_output_dir = "plots_5_DissimilarityMatrix" 
    os.makedirs(plots_output_dir, exist_ok=True)

    print("Loading training dataset...")
    training_set = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
    print("Training dataset loaded.")

    os.makedirs("clutter_channel_estimates", exist_ok=True)
    os.makedirs("triangulation_estimates", exist_ok=True)
    os.makedirs("dissimilarity_matrices", exist_ok=True)

    print("Loading precomputed estimates (clutter, triangulation)...")
    for dataset in training_set:
        dataset_name = os.path.basename(dataset['filename'])
        clutter_path = os.path.join("clutter_channel_estimates", dataset_name + ".npy")
        triangulation_path = os.path.join("triangulation_estimates", dataset_name + ".npy")
        if not os.path.exists(clutter_path) or not os.path.exists(triangulation_path):
            print(f"ERROR: Missing precomputed files for {dataset_name}. Exiting.")
            exit()
        dataset['clutter_acquisitions'] = np.load(clutter_path)
        dataset['triangulation_position_estimates'] = np.load(triangulation_path)
    print("Precomputed estimates loaded.")

    print("Clustering datasets...")
    for dataset in training_set:
        cluster_utils.cluster_dataset(dataset)
    print("Datasets clustered.")

    print("Calculating reflected CSI...")
    training_csi_reflected = []
    for dataset in tqdm(training_set, desc="Processing Datasets for Reflected CSI"):
        for cluster in tqdm(dataset['clusters'], desc=f"  Clusters in {os.path.basename(dataset['filename'])}", leave=False):
            antennas_per_array = espargos_0007.ROW_COUNT * espargos_0007.COL_COUNT
            R = np.zeros((espargos_0007.ARRAY_COUNT, antennas_per_array, antennas_per_array), dtype = np.complex64)
            num_tx_with_data = 0 
            for tx_idx, tx_csi_list_for_tx in enumerate(cluster['csi_freq_domain']):
                if tx_csi_list_for_tx.shape[0] == 0: continue
                num_tx_with_data += 1
                tx_csi_clutter_removed = CRAP.remove_clutter(tx_csi_list_for_tx, dataset['clutter_acquisitions'][tx_idx])
                tx_csi_flat = np.reshape(tx_csi_clutter_removed, (
                    tx_csi_clutter_removed.shape[0], tx_csi_clutter_removed.shape[1],
                    tx_csi_clutter_removed.shape[2] * tx_csi_clutter_removed.shape[3], tx_csi_clutter_removed.shape[4]))
                R_tx_contribution = np.einsum("dbms,dbns->bmn", tx_csi_flat, np.conj(tx_csi_flat)) / tx_csi_clutter_removed.shape[0]
                R += R_tx_contribution 
            
            if num_tx_with_data > 0: 
                R_final = R / num_tx_with_data
            else: 
                R_final = R 

            eig_val, eig_vec = np.linalg.eigh(R_final)
            eig_val = eig_val[:,::-1]; eig_vec = eig_vec[:,:,::-1]
            principal_eigenvectors = np.sqrt(eig_val[:,0])[:,np.newaxis] * eig_vec[:,:,0] 
            training_csi_reflected.append(np.reshape(principal_eigenvectors, (espargos_0007.ARRAY_COUNT, espargos_0007.ROW_COUNT, espargos_0007.COL_COUNT)))

    training_csi_reflected = np.asarray(training_csi_reflected)
    sample_count = training_csi_reflected.shape[0]
    print(f"Reflected CSI calculated. Shape: {training_csi_reflected.shape}")

    if sample_count == 0: print("ERROR: No training_csi_reflected samples. Exiting."); exit()

    print("Calculating ADP dissimilarity matrix...")
    adp_dissimilarity_matrix = np.zeros((sample_count, sample_count), dtype = np.float32)
    todo_queue_adp = mp.Queue(); output_queue_adp = mp.Queue(); processes_adp = []
    num_cpus = mp.cpu_count()

    print(f"  Spawning {num_cpus} workers for ADP dissimilarity...")
    for _ in range(num_cpus):
        p = mp.Process(target=adp_dissimilarities_worker, args=(todo_queue_adp, output_queue_adp, training_csi_reflected))
        p.start(); processes_adp.append(p)

    print(f"  Adding {sample_count} tasks to ADP todo_queue...")
    for i in range(sample_count): todo_queue_adp.put(i)
    for _ in range(num_cpus): todo_queue_adp.put(-1) 

    print(f"  Collecting {sample_count} results from ADP output_queue...")
    with tqdm(total = sample_count**2, desc="  ADP Dissimilarity Calculation") as pbar_adp:
        results_collected_adp = 0
        while results_collected_adp < sample_count:
            try:
                i, d_row_upper_triangle = output_queue_adp.get(timeout=300)
                if d_row_upper_triangle is not None: 
                    end_slice = i + len(d_row_upper_triangle)
                    adp_dissimilarity_matrix[i, i:end_slice] = d_row_upper_triangle
                    adp_dissimilarity_matrix[i:end_slice, i] = d_row_upper_triangle
                    pbar_adp.update(2 * len(d_row_upper_triangle) -1)
                results_collected_adp += 1
            except Exception as e:
                print(f"Error or Timeout getting from ADP output_queue: {e}. Collected {results_collected_adp} rows. Matrix might be incomplete.")
                break
    
    print("  Waiting for ADP workers to join...")
    for p in processes_adp: p.join(timeout=10); p.terminate() if p.is_alive() else None
    print("ADP dissimilarity matrix calculated.")

    plt.figure() 
    plt.hist(adp_dissimilarity_matrix.flatten(), bins = 100)
    adp_min = np.quantile(adp_dissimilarity_matrix.flatten(), 0.002) 
    plt.title(f"Minimal Dissimilarity = {adp_min:.3f}")
    plt.savefig(os.path.join(plots_output_dir, "adp_histogram.png"))
    plt.close()

    adp_thresh = 0.2
    adp_dissimilarity_matrix_shifted = np.maximum(adp_dissimilarity_matrix - adp_min, adp_thresh) 

    plt.figure() 
    plt.imshow(adp_dissimilarity_matrix)
    plt.title("ADP Dissimilarity Matrix")
    plt.colorbar()
    plt.savefig(os.path.join(plots_output_dir, "adp_matrix_raw.png"))
    plt.close()

    training_groundtruth_positions = []
    for dataset in training_set:
        training_groundtruth_positions.append(dataset['cluster_positions'])
    training_groundtruth_positions = np.concatenate(training_groundtruth_positions)

    if training_groundtruth_positions.shape[0] != sample_count:
        print(f"WARN: Mismatch sample_count ({sample_count}) vs groundtruth_positions ({training_groundtruth_positions.shape[0]}). This may cause issues.")
        min_len = min(sample_count, training_groundtruth_positions.shape[0])
        if sample_count > min_len:
            adp_dissimilarity_matrix = adp_dissimilarity_matrix[:min_len, :min_len]
            adp_dissimilarity_matrix_shifted = adp_dissimilarity_matrix_shifted[:min_len, :min_len]
        if training_groundtruth_positions.shape[0] > min_len:
            training_groundtruth_positions = training_groundtruth_positions[:min_len]
        sample_count = min_len


    groundtruth_distance_matrix = np.sqrt(np.sum((training_groundtruth_positions[np.newaxis,:,:2] - training_groundtruth_positions[:,np.newaxis,:2])**2, axis = -1))
    plt.figure() 
    plt.imshow(groundtruth_distance_matrix)
    plt.title("Groundtruth Distance Matrix"); plt.colorbar();
    plt.savefig(os.path.join(plots_output_dir, "groundtruth_distance_matrix.png"))
    plt.close()

    if sample_count > 1:
        step_plot = 500 if sample_count > 500 else max(1, sample_count // 4)
        for sample_index in range(1, min(2001, sample_count), step_plot):
            if sample_index >= adp_dissimilarity_matrix_shifted.shape[1]: continue 
            plt.figure() 
            plt.scatter(training_groundtruth_positions[:,0], training_groundtruth_positions[:,1], c = adp_dissimilarity_matrix_shifted[:,sample_index], s = 1, vmin = 0, vmax = 3) # Plot original
            plt.scatter([training_groundtruth_positions[sample_index,0]], [training_groundtruth_positions[sample_index,1]], c = "r", s = 2)
            plt.title("ADP Dissimilarities for Current Position");
            plt.savefig(os.path.join(plots_output_dir, f"adp_sample_scatter_{sample_index}.png"))
            plt.close()

    training_timestamps = []
    for dataset in training_set:
        training_timestamps.append(dataset['cluster_timestamps'])
    training_timestamps = np.concatenate(training_timestamps)
    if training_timestamps.shape[0] != sample_count: training_timestamps = training_timestamps[:sample_count]

    timestamp_dissimilarity_matrix = np.abs(np.subtract.outer(training_timestamps, training_timestamps))
    scaling_factor = 0.02
    dissimilarity_matrix_fused = np.minimum(adp_dissimilarity_matrix_shifted, timestamp_dissimilarity_matrix * scaling_factor)

    print("Calculating geodesic dissimilarity matrix...")
    n_neighbors = 20
    if sample_count < n_neighbors and sample_count > 0: n_neighbors = max(1, sample_count -1 if sample_count > 1 else 0)
    
    nbg = None
    if sample_count > 1 and n_neighbors > 0 :
        np.fill_diagonal(dissimilarity_matrix_fused, 0)
        nbrs_alg = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed", n_jobs=-1)
        nbrs = nbrs_alg.fit(dissimilarity_matrix_fused)
        nbg = kneighbors_graph(nbrs, n_neighbors, metric="precomputed", mode="distance") 
    else: print("WARN: Skipping k-NN graph construction.")

    dissimilarity_matrix_geodesic = np.zeros((sample_count, sample_count), dtype = np.float32)
    if nbg is not None and hasattr(nbg, 'shape') and nbg.shape[0] > 0 :
        todo_queue_geo = mp.Queue(); output_queue_geo = mp.Queue(); processes_geo = []
        print(f"  Spawning {num_cpus} workers for geodesic dissimilarity...")
        for _ in range(num_cpus):
            p = mp.Process(target=shortest_path_worker, args=(todo_queue_geo, output_queue_geo, nbg)); p.start(); processes_geo.append(p)
        print(f"  Adding {sample_count} tasks to geodesic todo_queue...")
        for i in range(sample_count): todo_queue_geo.put(i)
        for _ in range(num_cpus): todo_queue_geo.put(-1)
        
        print(f"  Collecting {sample_count} results from geodesic output_queue...")
        with tqdm(total = sample_count**2, desc="  Geodesic Distances Calculation") as pbar_geo:
            results_collected_geo = 0
            while results_collected_geo < sample_count:
                try:
                    i, d_geodesic_row = output_queue_geo.get(timeout=300) 
                    if d_geodesic_row is not None: 
                        dissimilarity_matrix_geodesic[i,:] = d_geodesic_row
                        pbar_geo.update(len(d_geodesic_row)) 
                    results_collected_geo += 1
                except Exception as e: print(f"Error/Timeout getting from Geodesic output_queue: {e}. Collected {results_collected_geo} rows."); break
        
        print("  Waiting for geodesic workers to join...");
        for p in processes_geo: p.join(timeout=10); p.terminate() if p.is_alive() else None 
        print("Geodesic dissimilarity matrix calculated.")
    else: print("Skipping geodesic dissimilarity calculation.")

    dissimilarity_matrix_geodesic_shifted = np.maximum(dissimilarity_matrix_geodesic - adp_thresh, 0.0) 

    print("Computing scaling factor to meters...")
    scaling_factor_meters = 1.0 
    if sample_count > 1: 
        scaling_nth_reduction_orig = 10
        classical_positions_for_scaling = np.concatenate([dataset['triangulation_position_estimates'] for dataset in training_set])

        if classical_positions_for_scaling.shape[0] != sample_count:
            print(f"WARN: Mismatch in scaling: classical_positions ({classical_positions_for_scaling.shape[0]}) vs geodesic_matrix_dim ({sample_count}). Aligning to min.")
            min_len_scale = min(classical_positions_for_scaling.shape[0], sample_count)
            classical_positions_for_scaling = classical_positions_for_scaling[:min_len_scale]
            temp_dissim_matrix_for_scaling = dissimilarity_matrix_geodesic_shifted[:min_len_scale, :min_len_scale]
        else:
            temp_dissim_matrix_for_scaling = dissimilarity_matrix_geodesic_shifted

        if temp_dissim_matrix_for_scaling.shape[0] < scaling_nth_reduction_orig or temp_dissim_matrix_for_scaling.shape[0] < 2:
            print(f"WARN: Not enough data for scaling factor after alignment ({temp_dissim_matrix_for_scaling.shape[0]} points). Using default 1.0.")
        else:
            classical_positions_reduced = classical_positions_for_scaling[::scaling_nth_reduction_orig]
            dissimilarity_matrix_reduced = temp_dissim_matrix_for_scaling[::scaling_nth_reduction_orig, ::scaling_nth_reduction_orig]

            if classical_positions_reduced.shape[0] < 2:
                print("WARN: Less than 2 points after reduction for scaling factor. Using default 1.0.")
            else:
                classical_distance_matrix = np.sqrt(np.sum((classical_positions_reduced[np.newaxis,:,:2] - classical_positions_reduced[:,np.newaxis,:2])**2, axis = -1)) # Nome original da variável
                dissimilarity_unit_meters = np.full_like(dissimilarity_matrix_reduced, np.nan)
                np.divide(dissimilarity_matrix_reduced, classical_distance_matrix, out = dissimilarity_unit_meters, where = classical_distance_matrix != 0)
                ratios = dissimilarity_unit_meters.flatten()
                ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
                
                if ratios.size > 20:
                    q_low, q_high = np.quantile(ratios, 0.01), np.quantile(ratios, 0.95)
                    if q_low >= q_high: q_low = np.min(ratios); q_high = np.max(ratios)
                    if q_low < q_high:
                        occurences, edges = np.histogram(ratios, bins=100, range=(q_low, q_high))
                        bin_centers = edges[:-1] + np.diff(edges) / 2.
                        if occurences.size > 0 and np.any(occurences > 0):
                            max_bin = np.argmax(occurences)
                            scaling_factor_meters = bin_centers[max_bin] 

                            plt.figure() 
                            plt.hist(ratios, bins = 100, range = (q_low, q_high)) 
                            plt.vlines(scaling_factor_meters, 0, 5000, "r")
                            plt.title(f"Dissimilarity Scaling Factor = {scaling_factor_meters:.3f}")
                            plt.savefig(os.path.join(plots_output_dir, "scaling_factor_histogram.png"))
                            plt.close()

                        else: scaling_factor_meters = np.median(ratios) if ratios.size > 0 else 1.0
                    else: scaling_factor_meters = np.median(ratios) if ratios.size > 0 else 1.0
                elif ratios.size > 0 : scaling_factor_meters = np.median(ratios)
                else: print("WARN: No valid ratios to compute scaling factor. Using default 1.0.")
    
    if not (np.isfinite(scaling_factor_meters) and scaling_factor_meters > 0):
        print(f"WARN: Computed scaling_factor_meters is invalid ({scaling_factor_meters}). Defaulting to 1.0.")
        scaling_factor_meters = 1.0
    dissimilarity_matrix_geodesic_meters = dissimilarity_matrix_geodesic_shifted / scaling_factor_meters
    print(f"Scaling factor to meters: {scaling_factor_meters}")

    output_filename = os.path.join("dissimilarity_matrices", espargos_0007.hash_dataset_names(training_set) + ".geodesic_meters.npy")
    np.save(output_filename, dissimilarity_matrix_geodesic_meters)
    print(f"Geodesic dissimilarity matrix (in meters) saved to: {output_filename}")

    plt.figure() 
    plt.imshow(dissimilarity_matrix_geodesic_meters)
    plt.title("Geodesic Dissimilarity Matrix"); plt.colorbar();
    plt.savefig(os.path.join(plots_output_dir, "geodesic_dissimilarity_matrix_meters.png"))
    plt.close()

    plt.figure() 
    plt.imshow(groundtruth_distance_matrix)
    plt.title("Groundtruth Distance Matrix"); plt.colorbar();
    plt.savefig(os.path.join(plots_output_dir, "groundtruth_distance_matrix_replot.png")) 
    plt.close()

    if sample_count > 1:
        step_plot_geo = 500 if sample_count > 500 else max(1, sample_count // 4) 
        for sample_index in range(1, min(2001, sample_count), step_plot_geo):
            if sample_index >= dissimilarity_matrix_geodesic_meters.shape[1]: continue
            plt.figure() 
            plt.scatter(training_groundtruth_positions[:,0], training_groundtruth_positions[:,1], c = dissimilarity_matrix_geodesic_meters[:,sample_index], s = 1, vmin = 0, vmax = 3) # Plot original
            plt.scatter([training_groundtruth_positions[sample_index,0]], [training_groundtruth_positions[sample_index,1]], c = "r", s = 2)
            plt.title("Geodesic Dissimilarities for Current Position");
            plt.savefig(os.path.join(plots_output_dir, f"geodesic_sample_scatter_{sample_index}.png"))
            plt.close()

    plt.figure(figsize = (8,4))
    plot_dissimilarity_over_euclidean_distance(dissimilarity_matrix_geodesic_meters, groundtruth_distance_matrix, "Geodesic fused")
    plot_dissimilarity_over_euclidean_distance(adp_dissimilarity_matrix, groundtruth_distance_matrix, "ADP") 
    plot_dissimilarity_over_euclidean_distance(adp_dissimilarity_matrix_shifted, groundtruth_distance_matrix, "ADP shifted")
    plt.legend(); plt.xlabel("Euclidean Distance [m]"); plt.ylabel("Dissimilarity");
    plt.savefig(os.path.join(plots_output_dir, "all_dissimilarities_vs_euclidean.png"))
    plt.close()

    print(f"Plots for Dissimilarity Matrix saved to: {os.path.abspath(plots_output_dir)}")
    print("Script finished.")