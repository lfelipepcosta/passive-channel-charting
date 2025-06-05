# setup_dirs.py
import os

lista_diretorios = [
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

for diretorio in lista_diretorios:
    try:
        os.makedirs(diretorio, exist_ok=True)
        print(f"Diretório '{diretorio}' verificado/criado.")
    except Exception as e:
        print(f"Erro ao criar o diretório '{diretorio}': {e}")