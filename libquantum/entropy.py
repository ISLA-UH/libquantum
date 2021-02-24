import numpy as np
from libquantum import utils
from libquantum.scales import EPSILON

"""
Entropy
"""


# FOR TONES
def snr_mean_max(tfr_coeff_complex: np.ndarray):
    """
    Computes the snr lin energy, snr in bits, and snr entropy defined in Garces (2020)
    :param tfr_coeff_complex: Complex coefficients for time-frequency representation. Can be real.
    :return:
    """
    # Evaluate Log energy entropy (LEE) = log(p) and Shannon Entropy (SE) = -p*log(p)
    # Assumes linear spectral coefficien ts (not power), takes the square
    energy = np.abs(tfr_coeff_complex)**2
    energy_mean = np.mean(energy)
    snr_lin = energy/energy_mean
    # Surprisal = log(p)
    snr_bits = 0.5*utils.log2epsilon(snr_lin + EPSILON)

    # SNR entropy per pixel
    snr_entropy = snr_lin*snr_bits
    # Scale by max
    snr_entropy /= np.max(snr_lin)

    return snr_lin, snr_bits, snr_entropy


def snr_mean_max_baseline(tfr_coeff_complex: np.ndarray):
    """
    Computes the baseline mean energy and the snr max for a template signal
    :param tfr_coeff_complex: Complex coefficients for time-frequency representation. Can be real.
    :return:
    """
    # Evaluate Log energy entropy (LEE) = log(p) and Shannon Entropy (SE) = -p*log(p)
    # Assumes linear spectral coefficien ts (not power), takes the square
    energy = np.abs(tfr_coeff_complex)**2
    energy_lin_mean = np.mean(energy)
    energy_lin_max = np.max(energy)
    snr_lin = energy/energy_lin_mean
    snr_lin_max = np.max(snr_lin)

    return energy_lin_max, energy_lin_mean, snr_lin_max


def snr_ref_max(tfr_coeff_complex: np.ndarray, energy_mean: float, snr_max: float):
    """
    Computes the snr energy, snr bits and snr entropy from baseline mean energy and max snr
    :param tfr_coeff_complex: Complex coefficients for time-frequency representation. Can be real.
    :param energy_mean: baseline mean energy.
    :param snr_max: baseline max linear snr
    :return:
    """
    # Evaluate Log energy entropy (LEE) = log(p) and Shannon Entropy (SE) = -p*log(p)
    # Assumes linear spectral coefficien ts (not power), takes the square
    energy = np.abs(tfr_coeff_complex)**2
    # energy_mean = np.mean(energy)
    snr_lin = energy/energy_mean
    # Surprisal = log(p)
    snr_bits = 0.5*utils.log2epsilon(snr_lin)

    # SNR entropy per pixel
    snr_entropy = snr_lin*snr_bits
    # Scale by max
    snr_entropy /= snr_max

    return snr_lin, snr_bits, snr_entropy


# FOR TRANSIENTS
# TODO: GET EXISTING CODE TO PERFORM THIS
def snr_mean_max_profile(tfr_coeff_complex: np.ndarray):
    """
    Computes the snr lin energy, snr in bits, and snr entropy defined in Garces (2020)
    :param tfr_coeff_complex: Complex coefficients for time-frequency representation. Can be real.
    :return:
    """
    # Evaluate Log energy entropy (LEE) = log(p) and Shannon Entropy (SE) = -p*log(p)
    # Assumes linear spectral coefficien ts (not power), takes the square
    energy = np.abs(tfr_coeff_complex)**2
    energy_mean = np.mean(energy)
    print('Energy_mean = ', energy_mean)
    snr_lin = energy/energy_mean
    # Surprisal = log(p)
    snr_bits = 0.5*utils.log2epsilon(snr_lin)

    # SNR entropy per pixel
    snr_entropy = snr_lin*snr_bits
    # Scale by max
    snr_entropy /= np.max(snr_lin)

    return snr_lin, snr_bits, snr_entropy


def snr_ref_max_profile(tfr_coeff_complex: np.ndarray, energy_mean: float, snr_max: float):
    # TODO: BUILD THIS FOR TRANSIENTS
    """
    Computes the snr energy, snr bits and snr entropy from
    frequency-dependent noise model/profile mean energy and max snr per band
    :param tfr_coeff_complex: Complex coefficients for time-frequency representation. Can be real.
    :param energy_mean: baseline frequency-dependent mean energy.
    :param snr_max: baseline max frequency-dependent linear snr
    :return:
    """
    # Evaluate Log energy entropy (LEE) = log(p) and Shannon Entropy (SE) = -p*log(p)
    # Assumes linear spectral coefficien ts (not power), takes the square
    energy = np.abs(tfr_coeff_complex)**2
    # energy_mean = np.mean(energy)
    snr_lin = energy/energy_mean
    # Surprisal = log(p)
    snr_bits = 0.5*utils.log2epsilon(snr_lin + EPSILON)

    # SNR entropy per pixel
    snr_entropy = snr_lin*snr_bits
    # Scale by max
    snr_entropy /= snr_max

    return snr_lin, snr_bits, snr_entropy
