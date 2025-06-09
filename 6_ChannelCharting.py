# CORREÇÃO: Definir o backend do Matplotlib ANTES de importar pyplot
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import espargos_0007
import cluster_utils
import CCEvaluation # Certifique-se que esta é a sua versão atual (robusta com NaNs)
import FeatureEngineering
import neural_network_utils
import numpy as np
import tensorflow as tf

# Heuristic: Uncertainty about angle of arrival as a function of root-MUSIC AoA estimate power
def aoa_power_to_kappa(aoa_power_tensor):
    aoa_power_tensor_float = tf.cast(aoa_power_tensor, dtype=tf.float32)
    return 5.0 * tf.pow(aoa_power_tensor_float, 6.0)

# Aproximação de Bessel I0 (versão do seu script, que é robusta)
def bessel_i0_approx(x):
    x_f32 = tf.cast(x, tf.float32)
    x_f32_safe = tf.maximum(x_f32, 0.0) 
    term1_den = tf.maximum((1.0 + x_f32_safe**2 / 4.0), 1e-9) 
    term1 = tf.math.cosh(x_f32_safe) / tf.pow(term1_den, 0.25)
    term2_num = 1.0 + 0.24273 * x_f32_safe**2
    term2_den = tf.maximum(1.0 + 0.43023 * x_f32_safe**2, 1e-9)
    term2 = term2_num / term2_den
    return term1 * term2

class ChannelChartingLoss(tf.keras.losses.Loss):
    def __init__(self, classical_weight, aoa_angles, aoa_powers, height, dissimilarity_matrix, dissimilarity_margin = 1.0, name = "CCLoss"):
        super().__init__(name = name)
        self.classical_weight = tf.constant(classical_weight, dtype=tf.float32)
        self.height = tf.constant(height, dtype=tf.float32)
        self.dissimilarity_matrix_tensor = tf.constant(dissimilarity_matrix, dtype = tf.float32)
        self.dissimilarity_margin = tf.constant(dissimilarity_margin, dtype=tf.float32)
        
        self.use_classical_loss = classical_weight > 1e-6 
        if self.use_classical_loss:
            if aoa_angles is None or aoa_powers is None:
                raise ValueError("aoa_angles and aoa_powers must be provided if classical_weight > 0")
            self.estimated_aoas_tensor = tf.constant(aoa_angles, dtype = tf.float32)
            self.aoa_powers_tensor = tf.constant(aoa_powers, dtype = tf.float32)
            self.centers_tensor = tf.constant(espargos_0007.array_positions, dtype = tf.float32)
            self.normalvectors_tensor = tf.constant(espargos_0007.array_normalvectors, dtype = tf.float32)
            self.rightvectors_tensor = tf.constant(espargos_0007.array_rightvectors, dtype = tf.float32)
        
        self.centroid_tensor = tf.constant(espargos_0007.centroid, dtype = tf.float32)[tf.newaxis, :2]

    def classical(self, pos_relative_to_centroid, aoas_tensor, aoa_powers_tensor):
        pos_global_xy = pos_relative_to_centroid + self.centroid_tensor
        batch_size = tf.shape(pos_global_xy)[0]
        height_column = tf.ones((batch_size, 1), dtype=self.height.dtype) * self.height
        pos_with_height = tf.concat([pos_global_xy, height_column], axis=1)

        relative_pos_to_arrays = pos_with_height[:,tf.newaxis,:] - self.centers_tensor
        normal = tf.einsum("dax,ax->da", relative_pos_to_arrays, self.normalvectors_tensor)
        right = tf.einsum("dax,ax->da", relative_pos_to_arrays, self.rightvectors_tensor)
        ideal_aoas = tf.math.atan2(right, normal)

        kappas_tf = aoa_power_to_kappa(aoa_powers_tensor) 
        kappas_safe = tf.maximum(kappas_tf, 0.0) 
        i0_kappas = bessel_i0_approx(kappas_safe)
        denominator = (2.0 * np.pi * tf.maximum(i0_kappas, 1e-9))
        aoa_likelihoods = tf.exp(kappas_safe * tf.cos(ideal_aoas - aoas_tensor)) / denominator
        return tf.math.reduce_prod(aoa_likelihoods, axis = -1)

    def siamese(self, pos_A, pos_B, dissimilarities):
        distances_pred = tf.math.sqrt(tf.math.reduce_sum(tf.square(pos_A - pos_B), axis = 1) + 1e-9)
        return tf.reduce_mean(tf.square(distances_pred - dissimilarities) / (dissimilarities + self.dissimilarity_margin + 1e-9))

    def call(self, y_true, y_pred):
        index_A = tf.cast(y_true[:,0], tf.int64)
        index_B = tf.cast(y_true[:,1], tf.int64)
        pos_A, pos_B = (y_pred[:,:2], y_pred[:,2:])
        
        indices_for_gather = tf.stack([index_A, index_B], axis = -1)
        dissimilarities = tf.gather_nd(self.dissimilarity_matrix_tensor, indices_for_gather)
        siamese_loss_val = self.siamese(pos_A, pos_B, dissimilarities)

        total_loss = (1.0 - self.classical_weight) * siamese_loss_val

        if self.use_classical_loss:
            aoa_A_tensor = tf.gather(self.estimated_aoas_tensor, index_A) 
            aoa_B_tensor = tf.gather(self.estimated_aoas_tensor, index_B) 
            aoa_power_A_tensor = tf.gather(self.aoa_powers_tensor, index_A)
            aoa_power_B_tensor = tf.gather(self.aoa_powers_tensor, index_B)

            classical_loss_A = self.classical(pos_A, aoa_A_tensor, aoa_power_A_tensor)
            classical_loss_B = self.classical(pos_B, aoa_B_tensor, aoa_power_B_tensor)
            
            neg_sum_classical_likelihoods = -tf.reduce_sum(classical_loss_A + classical_loss_B)
            total_loss += self.classical_weight * neg_sum_classical_likelihoods
        return total_loss

def combine_datasets(datasets_list_of_dicts, for_training = False):
    combined = dict()
    combined["cluster_features"] = []
    combined["cluster_positions"] = [] 
    heights = []
    if for_training:
        combined["cluster_aoa_angles"] = []
        combined["cluster_aoa_powers"] = []

    for dataset_dict in datasets_list_of_dicts: 
        combined["cluster_features"].append(dataset_dict["cluster_features"])
        combined["cluster_positions"].append(dataset_dict["cluster_positions"][:,:2]) 
        heights.append(dataset_dict["cluster_positions"][:,2]) 
        if for_training:
            if "cluster_aoa_angles" not in dataset_dict or "cluster_aoa_powers" not in dataset_dict:
                raise ValueError(f"Dados AOA faltando para {dataset_dict['filename']} em modo de treinamento.")
            combined["cluster_aoa_angles"].append(dataset_dict["cluster_aoa_angles"])
            combined["cluster_aoa_powers"].append(dataset_dict["cluster_aoa_powers"])

    combined["cluster_features"] = np.concatenate(combined["cluster_features"])
    combined["cluster_positions"] = np.concatenate(combined["cluster_positions"]) 
    combined["mean_height"] = np.mean(np.concatenate(heights)) if heights and np.concatenate(heights).size > 0 else 0.0
    if for_training:
        combined["cluster_aoa_angles"] = np.concatenate(combined["cluster_aoa_angles"])
        combined["cluster_aoa_powers"] = np.concatenate(combined["cluster_aoa_powers"])
        
        training_set_name_hash = espargos_0007.hash_dataset_names(datasets_list_of_dicts) 
        dissimilarity_matrix_path = os.path.join("dissimilarity_matrices", training_set_name_hash + ".geodesic_meters.npy")
        if not os.path.exists(dissimilarity_matrix_path):
            raise FileNotFoundError(f"Dissimilarity matrix file not found: {dissimilarity_matrix_path}")
        combined["dissimilarity_matrix"] = np.load(dissimilarity_matrix_path)
        
        num_features = combined["cluster_features"].shape[0]
        num_dissim_matrix_dim = combined["dissimilarity_matrix"].shape[0]
        if num_features != num_dissim_matrix_dim:
            raise ValueError(f"Mismatch feature count ({num_features}) and dissimilarity matrix dim ({num_dissim_matrix_dim}).")
    return combined

def affine_transform_channel_chart(groundtruth_pos_xy, channel_chart_pos_xy): 
    if groundtruth_pos_xy.shape[0] == 0 or channel_chart_pos_xy.shape[0] == 0:
        return channel_chart_pos_xy
    min_len = min(groundtruth_pos_xy.shape[0], channel_chart_pos_xy.shape[0])
    if min_len < groundtruth_pos_xy.shape[0] or min_len < channel_chart_pos_xy.shape[0]:
         print(f"WARN: Affine transform using {min_len} points due to mismatched input lengths.")
    groundtruth_pos_xy = groundtruth_pos_xy[:min_len]
    channel_chart_pos_xy = channel_chart_pos_xy[:min_len]
    if min_len < 3 : 
        print("WARN: Too few points for affine transform (need at least 3), returning untransformed positions.")
        return channel_chart_pos_xy

    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    A, res, rank, s = np.linalg.lstsq(pad(channel_chart_pos_xy), pad(groundtruth_pos_xy), rcond = None)
    transform = lambda x: unpad(np.dot(pad(x), A))
    return transform(channel_chart_pos_xy)

class ChartPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, featprov, position_labels_xy, fcf, augmented = True, plot_prefix="epoch_plot"):
        self.featprov = featprov
        self.position_labels_xy = position_labels_xy 
        self.fcf = fcf
        self.augmented = augmented
        self.plot_prefix = plot_prefix
        self.epoch_charts_dir = "epoch_charts" 
        os.makedirs(self.epoch_charts_dir, exist_ok=True)
   
    def on_epoch_end(self, epoch, logs = None):
        if self.position_labels_xy.shape[0] == 0: return

        indices_to_predict = np.arange(self.position_labels_xy.shape[0]).astype(np.int64)
        features_for_prediction = self.featprov(indices_to_predict)
        predicted_chart_positions_relative = self.fcf.predict(features_for_prediction, verbose=0)
        predicted_chart_positions_abs_chart = predicted_chart_positions_relative + espargos_0007.centroid[np.newaxis, :2]

        suptitle = f"Epoch {epoch+1}" 
        filename = os.path.join(self.epoch_charts_dir, f"{self.plot_prefix}_epoch_{epoch+1:03d}.png")

        position_predictions_for_eval = None
        if self.augmented: 
            position_predictions_for_eval = predicted_chart_positions_abs_chart
        else: 
            suptitle = suptitle + " (after opt. aff. trans.)"
            transformed_relative_chart = affine_transform_channel_chart(self.position_labels_xy, predicted_chart_positions_relative)
            position_predictions_for_eval = transformed_relative_chart + espargos_0007.centroid[np.newaxis, :2]

        if position_predictions_for_eval is not None and position_predictions_for_eval.shape[0] > 0: 
            errorvectors, errors, mae, cep = CCEvaluation.compute_localization_metrics(position_predictions_for_eval, self.position_labels_xy)
            title = f"CC Epoch {epoch+1}: MAE = {mae:.3f}m, CEP = {cep:.3f}m"
            print(f"  Callback: {title} for {suptitle}. Saving plot to {filename}")
            CCEvaluation.plot_colorized(position_predictions_for_eval, self.position_labels_xy, 
                                        suptitle = suptitle, title = title, 
                                        show = False, outfile = filename) 
        else:
            print(f"  Callback Epoch {epoch+1}: No valid points for evaluation. Skipping plot.")
        plt.close('all') 

def train_model(training_data, augmented = True, plot_prefix_cb="train_chart", num_epochs=100):
    STEPS_PER_EPOCH = 200
    EPOCHS = 10
    LEARNING_RATE_INITIAL = 1e-2
    LEARNING_RATE_FINAL = 1e-5
    BATCH_SIZES = [64, 128, 256, 512, 1024, 2048, 4096]
    
    training_features = training_data["cluster_features"]
    sample_count = training_features.shape[0]

    fcf_model = neural_network_utils.construct_model(input_shape = FeatureEngineering.FEATURE_SHAPE, name = "ChannelChartingModel")

    input_A = tf.keras.layers.Input(shape = (), dtype = tf.int64)
    input_B = tf.keras.layers.Input(shape = (), dtype = tf.int64)
    featprov = neural_network_utils.FeatureProviderLayer(dtype = tf.int64)
    featprov.set_features(training_features)
    csi_A = featprov(input_A)
    csi_B = featprov(input_B)
    embedding_A = fcf_model(csi_A)
    embedding_B = fcf_model(csi_B)
    output = tf.keras.layers.concatenate([embedding_A, embedding_B], axis = 1)
    siamese_model = tf.keras.models.Model([input_A, input_B], output, name = "SiameseNeuralNetwork")

    margin_val = 0.01 # Valor fixo do GitHub para dissimilarity_margin

    loss = ChannelChartingLoss(
        classical_weight = 0.05 if augmented else 0.0,
        aoa_angles = training_data["cluster_aoa_angles"],
        aoa_powers = training_data["cluster_aoa_powers"],
        height = training_data["mean_height"],
        dissimilarity_matrix = training_data["dissimilarity_matrix"],
        dissimilarity_margin = np.quantile(training_data["dissimilarity_matrix"], 0.01),
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate = LEARNING_RATE_INITIAL,
                    decay_steps = EPOCHS * STEPS_PER_EPOCH,
                    decay_rate = LEARNING_RATE_FINAL / LEARNING_RATE_INITIAL,
                    staircase = False)
    
    siamese_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule), loss = loss)

    def random_index_batch_generator():
         batch_count = 0
        while True:
            #print(batch_count, int(np.floor(batch_count / (TRAINING_BATCHES + 1) * len(BATCH_SIZES))))
            batch_size = BATCH_SIZES[min(int(np.floor(batch_count / (EPOCHS * STEPS_PER_EPOCH + 1) * len(BATCH_SIZES))), len(BATCH_SIZES) - 1)]
            batch_count = batch_count + 1
            #print("batch_size =", batch_size)
            indices_A = np.random.randint(sample_count, size = 256)
            indices_B = np.random.randint(sample_count, size = 256)
            yield (indices_A, indices_B), tf.stack([indices_A, indices_B], axis = 1)

    training_dataset_tf = tf.data.Dataset.from_generator(random_index_batch_generator,
        output_signature = (
            (tf.TensorSpec(shape = (None,), dtype = tf.int64), tf.TensorSpec(shape = (None,), dtype = tf.int64)), 
            tf.TensorSpec(shape = (None, 2), dtype = tf.int64)  
        )).prefetch(tf.data.AUTOTUNE) 
    
    plot_callback = ChartPlotCallback(featprov, training_data["cluster_positions"], fcf_model, augmented = augmented, plot_prefix=plot_prefix_cb)
    
    print(f"Starting training for {'augmented' if augmented else 'un-augmented'} model ({num_epochs} epochs, {STEPS_PER_EPOCH} steps/epoch, batch_size={FIXED_BATCH_SIZE})...")
    siamese_model.fit(training_dataset_tf, steps_per_epoch = STEPS_PER_EPOCH, epochs = num_epochs, callbacks = [plot_callback])
    print(f"Training finished for {'augmented' if augmented else 'un-augmented'} model.")
    return fcf_model

if __name__ == '__main__':
    mp.freeze_support()
    os.makedirs("clutter_channel_estimates", exist_ok=True)
    os.makedirs("aoa_estimates", exist_ok=True)
    os.makedirs("dissimilarity_matrices", exist_ok=True)
    os.makedirs("epoch_charts", exist_ok=True)
    os.makedirs("evaluation_charts", exist_ok=True) 

    print("Loading datasets...")
    training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
    test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
    test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)
    print("Datasets loaded.")

    all_datasets = training_set_robot + test_set_robot + test_set_human

    print("Loading precomputed estimates...")
    for dataset in all_datasets:
        dataset_name = os.path.basename(dataset["filename"])
        clutter_path = os.path.join("clutter_channel_estimates", dataset_name + ".npy")
        if not os.path.exists(clutter_path): print(f"ERROR: Clutter file not found: {clutter_path}. Exiting."); exit()
        dataset["clutter_acquisitions"] = np.load(clutter_path)
        
        is_in_training_set_robot = any(dataset["filename"] == tr_ds["filename"] for tr_ds in training_set_robot)
        if is_in_training_set_robot:
            aoa_angles_path = os.path.join("aoa_estimates", dataset_name + ".aoa_angles.npy")
            aoa_powers_path = os.path.join("aoa_estimates", dataset_name + ".aoa_powers.npy")
            if not os.path.exists(aoa_angles_path) or not os.path.exists(aoa_powers_path):
                print(f"ERROR: AOA files for training dataset {dataset_name} not found. Exiting."); exit()
            dataset["cluster_aoa_angles"] = np.load(aoa_angles_path)
            dataset["cluster_aoa_powers"] = np.load(aoa_powers_path)
    print("Precomputed estimates loaded.")

    print("Clustering datasets...")
    for dataset in all_datasets: cluster_utils.cluster_dataset(dataset)
    print("Datasets clustered.")

    print("Precomputing features (this may take a while)...")
    FeatureEngineering.precompute_features(all_datasets)
    print("Features precomputed.")

    print("Combining datasets for training and testing...")
    robot_training_data = combine_datasets(training_set_robot, for_training = True)
    robot_test_data = combine_datasets(test_set_robot, for_training = False)
    human_test_data = combine_datasets(test_set_human, for_training = False)
    print("Datasets combined.")

    EPOCHS_FOR_TRAINING = 100

    print("\n--- Training Un-augmented FCF Model (PCC) ---")
    fcf_model = train_model(robot_training_data, augmented = False, plot_prefix_cb="pcc_unaugmented_train_chart", num_epochs=EPOCHS_FOR_TRAINING)

    if fcf_model:
        eval_plot_dir = "evaluation_charts"
        print(f"\n--- Evaluating Un-augmented FCF Model on Robot Test Set (Plots saved to {eval_plot_dir}) ---")
        test_set_robot_predictions_relative = fcf_model.predict(robot_test_data["cluster_features"], verbose=0)
        test_set_robot_predictions_transformed_relative = affine_transform_channel_chart(robot_test_data["cluster_positions"], test_set_robot_predictions_relative)
        test_set_robot_predictions_final = test_set_robot_predictions_transformed_relative + espargos_0007.centroid[np.newaxis, :2]
        
        errorvectors, errors, mae, cep = CCEvaluation.compute_localization_metrics(test_set_robot_predictions_final, robot_test_data["cluster_positions"])
        suptitle = f"PCC (Un-augmented) Evaluated on Test Set (Robot) - {EPOCHS_FOR_TRAINING} Epochs"
        title = f"Unsupervised: MAE = {mae:.3f}m, CEP = {cep:.3f}m"
        CCEvaluation.plot_colorized(test_set_robot_predictions_final, robot_test_data["cluster_positions"], suptitle=suptitle, title=title, show=False, outfile=os.path.join(eval_plot_dir, f"pcc_unaug_eval_robot_scatter_{EPOCHS_FOR_TRAINING}e.png"))
        plt.close('all') 
        metrics = CCEvaluation.compute_all_performance_metrics(test_set_robot_predictions_final, robot_test_data["cluster_positions"])
        CCEvaluation.plot_error_ecdf(test_set_robot_predictions_final, robot_test_data["cluster_positions"], outfile=os.path.join(eval_plot_dir, f"pcc_unaug_eval_robot_ecdf_{EPOCHS_FOR_TRAINING}e.jpg"))
        plt.close('all')
        print(f"Metrics for Un-augmented PCC on Robot Test Set ({EPOCHS_FOR_TRAINING} Epochs):")
        for metric_name, metric_value in metrics.items(): print(f"  {metric_name.upper().rjust(6, ' ')}: {metric_value:.3f}")

        print(f"\n--- Evaluating Un-augmented FCF Model on Human Test Set (Plots saved to {eval_plot_dir}) ---")
        test_set_human_predictions_relative = fcf_model.predict(human_test_data["cluster_features"], verbose=0)
        test_set_human_predictions_transformed_relative = affine_transform_channel_chart(human_test_data["cluster_positions"], test_set_human_predictions_relative)
        test_set_human_predictions_final = test_set_human_predictions_transformed_relative + espargos_0007.centroid[np.newaxis, :2]
        
        errorvectors, errors, mae, cep = CCEvaluation.compute_localization_metrics(test_set_human_predictions_final, human_test_data["cluster_positions"])
        suptitle = f"PCC (Un-augmented) Evaluated on Test Set (Human) - {EPOCHS_FOR_TRAINING} Epochs"
        title = f"Unsupervised: MAE = {mae:.3f}m, CEP = {cep:.3f}m"
        CCEvaluation.plot_colorized(test_set_human_predictions_final, human_test_data["cluster_positions"], suptitle=suptitle, title=title, show=False, outfile=os.path.join(eval_plot_dir, f"pcc_unaug_eval_human_scatter_{EPOCHS_FOR_TRAINING}e.png"))
        plt.close('all')
        metrics = CCEvaluation.compute_all_performance_metrics(test_set_human_predictions_final, human_test_data["cluster_positions"])
        CCEvaluation.plot_error_ecdf(test_set_human_predictions_final, human_test_data["cluster_positions"], outfile=os.path.join(eval_plot_dir, f"pcc_unaug_eval_human_ecdf_{EPOCHS_FOR_TRAINING}e.jpg"))
        plt.close('all')
        print(f"Metrics for Un-augmented PCC on Human Test Set ({EPOCHS_FOR_TRAINING} Epochs):")
        for metric_name, metric_value in metrics.items(): print(f"  {metric_name.upper().rjust(6, ' ')}: {metric_value:.3f}")
    else:
        print("Skipping evaluation for un-augmented model as training failed.")

    print("\n--- Training Augmented FCF Model (Augmented PCC) ---")
    augmented_fcf_model = train_model(robot_training_data, augmented = True, plot_prefix_cb="pcc_augmented_train_chart", num_epochs=EPOCHS_FOR_TRAINING)

    if augmented_fcf_model:
        eval_plot_dir = "evaluation_charts" 
        print(f"\n--- Evaluating Augmented FCF Model on Training Set (Robot) (Plots saved to {eval_plot_dir}) ---")
        training_set_robot_predictions_relative_aug = augmented_fcf_model.predict(robot_training_data["cluster_features"], verbose=0)
        training_set_robot_predictions_abs_aug = training_set_robot_predictions_relative_aug + espargos_0007.centroid[np.newaxis, :2]
        
        errorvectors, errors, mae, cep = CCEvaluation.compute_localization_metrics(training_set_robot_predictions_abs_aug, robot_training_data["cluster_positions"])
        suptitle = f"Augmented PCC Evaluated on Training Set (Robot) - {EPOCHS_FOR_TRAINING} Epochs"
        title = f"Unsupervised (Augmented): MAE = {mae:.3f}m, CEP = {cep:.3f}m"
        CCEvaluation.plot_colorized(training_set_robot_predictions_abs_aug, robot_training_data["cluster_positions"], suptitle=suptitle, title=title, show=False, outfile=os.path.join(eval_plot_dir, f"pcc_aug_eval_train_robot_scatter_{EPOCHS_FOR_TRAINING}e.png"))
        plt.close('all')

        print(f"\n--- Evaluating Augmented FCF Model on Robot Test Set (Plots saved to {eval_plot_dir}) ---")
        test_set_robot_predictions_relative_aug = augmented_fcf_model.predict(robot_test_data["cluster_features"], verbose=0)
        test_set_robot_predictions_abs_aug = test_set_robot_predictions_relative_aug + espargos_0007.centroid[np.newaxis, :2]
        
        errorvectors, errors, mae, cep = CCEvaluation.compute_localization_metrics(test_set_robot_predictions_abs_aug, robot_test_data["cluster_positions"])
        suptitle = f"Augmented PCC Evaluated on Test Set (Robot) - {EPOCHS_FOR_TRAINING} Epochs"
        title = f"Unsupervised (Augmented): MAE = {mae:.3f}m, CEP = {cep:.3f}m"
        CCEvaluation.plot_colorized(test_set_robot_predictions_abs_aug, robot_test_data["cluster_positions"], suptitle=suptitle, title=title, show=False, outfile=os.path.join(eval_plot_dir, f"pcc_aug_eval_robot_scatter_{EPOCHS_FOR_TRAINING}e.png"))
        plt.close('all')
        metrics = CCEvaluation.compute_all_performance_metrics(test_set_robot_predictions_abs_aug, robot_test_data["cluster_positions"])
        CCEvaluation.plot_error_ecdf(test_set_robot_predictions_abs_aug, robot_test_data["cluster_positions"], outfile=os.path.join(eval_plot_dir, f"pcc_aug_eval_robot_ecdf_{EPOCHS_FOR_TRAINING}e.jpg"))
        plt.close('all')
        print(f"Metrics for Augmented PCC on Robot Test Set ({EPOCHS_FOR_TRAINING} Epochs):")
        for metric_name, metric_value in metrics.items(): print(f"  {metric_name.upper().rjust(6, ' ')}: {metric_value:.3f}")

        print(f"\n--- Evaluating Augmented FCF Model on Human Test Set (Plots saved to {eval_plot_dir}) ---")
        test_set_human_predictions_relative_aug = augmented_fcf_model.predict(human_test_data["cluster_features"], verbose=0)
        test_set_human_predictions_abs_aug = test_set_human_predictions_relative_aug + espargos_0007.centroid[np.newaxis, :2]
        
        errorvectors, errors, mae, cep = CCEvaluation.compute_localization_metrics(test_set_human_predictions_abs_aug, human_test_data["cluster_positions"])
        suptitle = f"Augmented PCC Evaluated on Test Set (Human) - {EPOCHS_FOR_TRAINING} Epochs"
        title = f"Unsupervised (Augmented): MAE = {mae:.3f}m, CEP = {cep:.3f}m"
        CCEvaluation.plot_colorized(test_set_human_predictions_abs_aug, human_test_data["cluster_positions"], suptitle=suptitle, title=title, show=False, outfile=os.path.join(eval_plot_dir, f"pcc_aug_eval_human_scatter_{EPOCHS_FOR_TRAINING}e.png"))
        plt.close('all')
        metrics = CCEvaluation.compute_all_performance_metrics(test_set_human_predictions_abs_aug, human_test_data["cluster_positions"])
        CCEvaluation.plot_error_ecdf(test_set_human_predictions_abs_aug, human_test_data["cluster_positions"], outfile=os.path.join(eval_plot_dir, f"pcc_aug_eval_human_ecdf_{EPOCHS_FOR_TRAINING}e.jpg"))
        plt.close('all')
        print(f"Metrics for Augmented PCC on Human Test Set ({EPOCHS_FOR_TRAINING} Epochs):")
        for metric_name, metric_value in metrics.items(): print(f"  {metric_name.upper().rjust(6, ' ')}: {metric_value:.3f}")
    else:
        print("Skipping evaluation for augmented model as training failed.")

    print(f"\nTodos os plots de época foram salvos em: {os.path.abspath('epoch_charts')}")
    print(f"Todos os plots de avaliação final foram salvos em: {os.path.abspath('evaluation_charts')}")
    print("Script 6_ChannelCharting.py finished.")