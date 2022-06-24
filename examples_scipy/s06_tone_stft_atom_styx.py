"""
libquantum example: s06_stft_tone_stft_atom_styx
Compute stft spectrogram with libquantum
TODO: Turn into functions with hard-coded defaults
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from libquantum import utils, synthetics, spectra, benchmark_signals, atoms, entropy
import libquantum.plot_templates.plot_time_frequency_reps_black as pltq
from libquantum.styx import tfr_stx_fft

print(__doc__)
EVENT_NAME = 'tone test'
station_id_str = 'synth'
# alpha: Shape parameter of the Tukey window, representing the fraction of the window inside the cosine tapered region.
# If zero, the Tukey window is equivalent to a rectangular window.
# If one, the Tukey window is equivalent to a Hann window.
alpha = 1

frequency_tone_hz = 60
order_number_input = 3

if __name__ == "__main__":
    """
    Compute the spectrogram over sliding windows.
    The Welch method is equivalent to averaging the spectrogram over the columns.
    """

    # Construct a tone of unit amplitude and fixed frequency with a constant sample rate
    [mic_sig, time_s, time_fft_nd,
     frequency_sample_rate_hz, frequency_center_fft_hz, frequency_resolution_fft_hz] = \
        benchmark_signals.well_tempered_tone(frequency_center_hz=frequency_tone_hz, add_noise_taper_aa=True)

    # Computed and nominal values
    mic_sig_rms = np.std(mic_sig)
    mic_sig_rms_nominal = 1/np.sqrt(2)

    # Compute the Welch PSD; averaged spectrum over sliding windows
    frequency_welch_hz, psd_welch_power = signal.welch(x=mic_sig,
                                                       fs=frequency_sample_rate_hz,
                                                       window=('tukey', alpha),
                                                       nperseg=time_fft_nd,
                                                       noverlap=time_fft_nd // 2,
                                                       nfft=time_fft_nd,
                                                       detrend='constant',
                                                       return_onesided=True,
                                                       axis=-1,
                                                       scaling='spectrum',
                                                       average='mean')

    # Compute the spectrogram with the stft option
    frequency_stft_hz, time_stft_s, stft_complex = \
        signal.stft(x=mic_sig,
                    fs=frequency_sample_rate_hz,
                    window=('tukey', alpha),
                    nperseg=time_fft_nd,
                    noverlap=time_fft_nd // 2,
                    nfft=time_fft_nd,
                    detrend='constant',
                    return_onesided=True,
                    axis=-1,
                    boundary='zeros',
                    padded=True)

    stft_power = 2 * np.abs(stft_complex) ** 2

    # Compute complex wavelet transform (cwt) from signal duration using the Gabor atoms
    cwt_complex, _, time_cwt_s, frequency_cwt_hz = \
        atoms.cwt_chirp_from_sig(sig_wf=mic_sig,
                                 frequency_sample_rate_hz=frequency_sample_rate_hz,
                                 band_order_Nth=order_number_input,
                                 dictionary_type="spect",
                                 frequency_ref=frequency_center_fft_hz)

    cwt_power = 2 * np.abs(cwt_complex) ** 2

    # Compute Stockwell transform
    [stx_complex, _, frequency_stx_hz, frequency_stx_fft_hz, W] = \
        tfr_stx_fft(sig_wf=mic_sig,
                    time_sample_interval=1/frequency_sample_rate_hz,
                    frequency_min=frequency_resolution_fft_hz,
                    frequency_max=frequency_sample_rate_hz/2,
                    scale_order_input=order_number_input,
                    scale_ref_input=1/59.66796875,
                    is_geometric=True,
                    is_inferno=False)

    stx_power = 2 * np.abs(stx_complex) ** 2

    # print("STX Frequency:", frequency_stx_fft_hz)
    # TODO: Reconcile STX frequency with STFT
    # Compute the 'equivalent' fft rms amplitude
    fft_rms_welch = np.sqrt(np.abs(psd_welch_power)) / mic_sig_rms
    fft_rms_stft = np.sqrt(np.average(stft_power, axis=1)) / mic_sig_rms
    fft_rms_cwt = np.sqrt(np.average(cwt_power, axis=1)) / mic_sig_rms
    fft_rms_stx = np.sqrt(np.average(stx_power, axis=1)) / mic_sig_rms

    # Express in bits; revisit
    # TODO: What units shall we use? Evaluate CWT and Stockwell first

    mic_stft_bits = utils.log2epsilon(np.sqrt(stft_power))
    mic_cwt_bits = utils.log2epsilon(np.sqrt(cwt_power))
    mic_stx_bits = utils.log2epsilon(np.sqrt(stx_power))

    print('Max stft bits:', np.max(mic_stft_bits))
    print('Max cwt bits:', np.max(mic_cwt_bits))
    print('Max stx bits:', np.max(mic_stx_bits))

    # Show the waveform and the averaged FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_s, mic_sig)
    ax1.set_title('Synthetic CW, with taper')
    ax1.set_xlabel('Time, s')
    ax1.set_ylabel('Norm')
    ax2.semilogx(frequency_welch_hz, fft_rms_welch, label='Welch')
    ax2.semilogx(frequency_stft_hz, fft_rms_stft, '.-', label="STFT")
    ax2.semilogx(frequency_cwt_hz, fft_rms_cwt, '-.', label='CWT')
    ax2.semilogx(frequency_stx_hz, fft_rms_stx, '--', label='STX')

    ax2.set_title('Welch and Spect FFT (RMS), f = ' + str(round(frequency_center_fft_hz*100)/100) + ' Hz')
    ax2.set_xlabel('Frequency, hz')
    ax2.set_ylabel('|FFT|/RMS(FFT)')
    ax2.grid(True)
    ax2.legend()

    plt.show()
    exit()
    # Select plot frequencies
    fmin = 2*frequency_resolution_fft_hz
    fmax = frequency_sample_rate_hz/2  # Nyquist

    pltq.plot_wf_mesh_vert(redvox_id='00',
                           wf_panel_a_sig=mic_sig,
                           wf_panel_a_time=time_s,
                           mesh_time=time_stft_s,
                           mesh_frequency=frequency_stft_hz,
                           mesh_panel_b_tfr=mic_stft_bits,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="stft for " + EVENT_NAME,
                           frequency_hz_ymin=fmin,
                           frequency_hz_ymax=fmax)

    pltq.plot_wf_mesh_vert(redvox_id='00',
                           wf_panel_a_sig=mic_sig,
                           wf_panel_a_time=time_s,
                           mesh_time=time_s,
                           mesh_frequency=frequency_stx_hz,
                           mesh_panel_b_tfr=mic_stx_bits,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="STX for " + EVENT_NAME,
                           frequency_hz_ymin=fmin,
                           frequency_hz_ymax=fmax)

    pltq.plot_wf_mesh_vert(redvox_id='00',
                           wf_panel_a_sig=mic_sig,
                           wf_panel_a_time=time_s,
                           mesh_time=time_cwt_s,
                           mesh_frequency=frequency_cwt_hz,
                           mesh_panel_b_tfr=mic_cwt_bits,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="cwt for " + EVENT_NAME,
                           frequency_hz_ymin=fmin,
                           frequency_hz_ymax=fmax)

    plt.show()

