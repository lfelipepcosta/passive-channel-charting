# WAVELET.py
import numpy as np
import pywt

def wavelet_denoise_csi(csi_data, wavelet='db4', level=2, mode='soft', threshold_scale=1.0):
    """
    Aplica denoising em um conjunto de dados CSI usando a Transformada Wavelet.

    O processo envolve transformar o sinal do domínio da frequência (CSI) para o
    domínio do tempo (Channel Impulse Response - CIR), aplicar o filtro wavelet em
    cada CIR individual, e então transformar o resultado de volta para o domínio
    da frequência.

    Args:
        csi_data (np.ndarray): O array de CSI a ser limpo. A última dimensão
                               deve ser a das subportadoras.
        wavelet (str): O nome da wavelet a ser usada (ex: 'db4', 'sym8').
        level (int): O nível de decomposição da wavelet.
        mode (str): O tipo de thresholding a ser aplicado ('soft' ou 'hard').

    Returns:
        np.ndarray: O array de CSI após a aplicação do denoising, com a mesma
                    forma do array de entrada.
    """
    # 1. Transformar o CSI (domínio da frequência) para o CIR (domínio do tempo)
    # A IFFT é aplicada ao longo do último eixo (subportadoras).
    cir_data = np.fft.ifft(csi_data, axis=-1)

    # 2. Inicializar um array para armazenar o CIR após o denoising
    cir_denoised = np.zeros_like(cir_data, dtype=np.complex64)

    # 3. Iterar por cada sinal CIR individual para aplicar o filtro
    # O nditer permite iterar de forma eficiente por todos os eixos, exceto o último.
    it = np.nditer(cir_data[..., 0], flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        signal = cir_data[idx]

        # Lidar separadamente com as partes real e imaginária do sinal complexo
        real_part = np.real(signal)
        imag_part = np.imag(signal)

        # --- Denoising da Parte Real ---
        coeffs_real = pywt.wavedec(real_part, wavelet, level=level)
        # Estimar o ruído usando o Desvio Padrão da Mediana Absoluta (MAD) no nível mais fino
        sigma_real = np.median(np.abs(coeffs_real[-1])) / 0.6745
        # Calcular o Universal Threshold
        threshold_real = sigma_real * np.sqrt(2 * np.log(len(real_part))) * threshold_scale
        # Aplicar o thresholding
        new_coeffs_real = [coeffs_real[0]] + [pywt.threshold(c, threshold_real, mode=mode) for c in coeffs_real[1:]]
        # Reconstruir o sinal
        denoised_real = pywt.waverec(new_coeffs_real, wavelet)

        # --- Denoising da Parte Imaginária ---
        coeffs_imag = pywt.wavedec(imag_part, wavelet, level=level)
        sigma_imag = np.median(np.abs(coeffs_imag[-1])) / 0.6745
        threshold_imag = sigma_imag * np.sqrt(2 * np.log(len(imag_part))) * threshold_scale
        new_coeffs_imag = [coeffs_imag[0]] + [pywt.threshold(c, threshold_imag, mode=mode) for c in coeffs_imag[1:]]
        denoised_imag = pywt.waverec(new_coeffs_imag, wavelet)
        
        # Garantir que o sinal reconstruído tenha o mesmo tamanho do original
        original_length = csi_data.shape[-1]
        denoised_signal = denoised_real[:original_length] + 1j * denoised_imag[:original_length]

        cir_denoised[idx] = denoised_signal
        
        it.iternext()

    # 4. Transformar o CIR limpo de volta para o domínio da frequência (CSI)
    csi_denoised = np.fft.fft(cir_denoised, axis=-1)

    return csi_denoised