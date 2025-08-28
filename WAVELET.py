import numpy as np
import pywt

def wavelet_denoise_csi(csi_data, wavelet='db4', level=2, mode='soft', threshold_scale=1.0):
    """
    Applies denoising to a CSI dataset using the Wavelet Transform.

    The process involves transforming the signal from the frequency domain (CSI) to
    the time domain (Channel Impulse Response - CIR), applying the wavelet filter
    to each individual CIR, and then transforming the result back to the
    frequency domain.

    Args:
        csi_data (np.ndarray): The CSI array to be cleaned. The last dimension
                               must be the subcarriers.
        wavelet (str): The name of the wavelet to be used (e.g., 'db4', 'sym8').
        level (int): The decomposition level of the wavelet.
        mode (str): The thresholding mode to be applied ('soft' or 'hard').
        threshold_scale (float): A scaling factor for the threshold value (1.0 = default).

    Returns:
        np.ndarray: The CSI array after applying the denoising, with the same
                    shape as the input array.
    """
    # 1. Transform the CSI (frequency domain) to the CIR (time domain)
    # The IFFT is applied along the last axis (subcarriers).
    cir_data = np.fft.ifft(csi_data, axis=-1)

    # 2. Initialize an array to store the CIR after denoising
    cir_denoised = np.zeros_like(cir_data, dtype=np.complex64)

    # 3. Iterate over each individual CIR signal to apply the filter
    # The nditer allows for efficient iteration over all axes except the last one.
    it = np.nditer(cir_data[..., 0], flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        signal = cir_data[idx]

        # Handle the real and imaginary parts of the complex signal separately
        real_part = np.real(signal)
        imag_part = np.imag(signal)

        # --- Denoising the Real Part ---
        coeffs_real = pywt.wavedec(real_part, wavelet, level=level)
        # Estimate noise using the Median Absolute Deviation (MAD) on the finest level
        sigma_real = np.median(np.abs(coeffs_real[-1])) / 0.6745
        # Calculate the Universal Threshold and apply the scaling factor
        threshold_real = (sigma_real * np.sqrt(2 * np.log(len(real_part)))) * threshold_scale
        # Apply thresholding
        new_coeffs_real = [coeffs_real[0]] + [pywt.threshold(c, threshold_real, mode=mode) for c in coeffs_real[1:]]
        # Reconstruct the signal
        denoised_real = pywt.waverec(new_coeffs_real, wavelet)

        # --- Denoising the Imaginary Part ---
        coeffs_imag = pywt.wavedec(imag_part, wavelet, level=level)
        sigma_imag = np.median(np.abs(coeffs_imag[-1])) / 0.6745
        threshold_imag = (sigma_imag * np.sqrt(2 * np.log(len(imag_part)))) * threshold_scale
        new_coeffs_imag = [coeffs_imag[0]] + [pywt.threshold(c, threshold_imag, mode=mode) for c in coeffs_imag[1:]]
        denoised_imag = pywt.waverec(new_coeffs_imag, wavelet)
        
        # Ensure the reconstructed signal has the same length as the original
        original_length = csi_data.shape[-1]
        denoised_signal = denoised_real[:original_length] + 1j * denoised_imag[:original_length]

        cir_denoised[idx] = denoised_signal
        
        it.iternext()

    # 4. Transform the cleaned CIR back to the frequency domain (CSI)
    csi_denoised = np.fft.fft(cir_denoised, axis=-1)

    return csi_denoised