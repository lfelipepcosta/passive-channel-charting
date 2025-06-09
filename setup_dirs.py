import os

directories_list = [
    "clutter_channel_estimates",
    "plots_1_ClutterChannels",
    "plots_2_SupervisedBaseline",
    "aoa_estimates",
    "plots_3_AoA_Estimation",
    "triangulation_estimates",
    "plots_4_Triangulation",
    "dissimilarity_matrices",
    "plots_5_DissimilarityMatrix",
    "epoch_charts",
    "evaluation_charts"
]

for directory in directories_list
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory '{directory}': {e}")