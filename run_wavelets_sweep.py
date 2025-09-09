import os
import subprocess
from itertools import product

''' PARAM_GRID = {
    'wavelet_family': ['db2', 'db4', 'db8', 'sym4', 'sym8', 'coif2', 'coif5', 'bior2.2', 'bior3.7'],
    'decomposition_level': [1, 2, 3],
    'threshold_mode': ['soft', 'hard'],
    'threshold_scale': [0.6, 0.8, 1.0, 1.2, 1.4]
}'''

PARAM_GRID = {
    'wavelet_family': ['sym4'],
    'decomposition_level': [1, 2, 3],
    'threshold_mode': ['hard', 'soft'],
    'threshold_scale': [0.8, 1.0, 1.2, 1.4]
}

# Próximo conjunto de parâmetros a serem testados
'''PARAM_GRID = {
    'wavelet_family': ['coif5'],
    'decomposition_level': [1, 2, 3],
    'threshold_mode': ['hard', 'soft'],
    'threshold_scale': [0.8, 1.0, 1.2, 1.4]
}'''

def check_valid_level(wavelet, level, signal_length=53):
    """
    Checks if the decomposition level is valid for a given wavelet and signal length.
    Prevents UserWarning from PyWavelets.
    """
    import pywt
    try:
        w = pywt.Wavelet(wavelet)
        max_level = pywt.dwt_max_level(signal_length, w.dec_len)
        return level <= max_level
    except:
        return False

def run_experiment(params):
    """
    Constructs and runs the command for a single experiment.
    """
    wavelet = params['wavelet_family']
    level = params['decomposition_level']
    mode = params['threshold_mode']
    scale = params['threshold_scale']

    if not check_valid_level(wavelet, level):
        print(f"--- SKIPPING invalid combination: Wavelet '{wavelet}' with level {level} ---\n")
        return

    command = [
        "python",
        "3_AoA_Estimation_WAVELET_URM.py",
        str(wavelet),
        str(level),
        str(mode),
        str(scale)
    ]

    experiment_name = f"{wavelet}_level{level}_{mode}_scale{str(scale).replace('.', 'p')}"
    print(f"--- RUNNING EXPERIMENT: {experiment_name} ---")

    try:
        subprocess.run(command, check=True, text=True)
        print(f"--- COMPLETED: {experiment_name} ---\n")
    except subprocess.CalledProcessError as e:
        print(f"!!!!!! ERROR running experiment: {experiment_name} !!!!!!")
        print(f"Command failed with return code {e.returncode}")
        print(f"Output:\n{e.stdout}\n{e.stderr}")
        print("Continuing to the next experiment...\n")


if __name__ == "__main__":
    keys, values = zip(*PARAM_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    print(f"Starting hyperparameter sweep for {len(param_combinations)} total combinations.")

    for params in param_combinations:
        run_experiment(params)

    print("--- Hyperparameter sweep finished! ---")