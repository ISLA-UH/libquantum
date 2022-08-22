import math
from dataclasses import dataclass
from typing import Iterator

import matplotlib.pyplot as plt
from scipy.io import wavfile

import libquantum.styx_stx_loopy as stx

import numpy as np


@dataclass
class StockwellConfig:
    """
    Required Stockwell Transform inputs.
    """

    n_cols: int
    n_chunk: int
    band_type_lin: bool
    band_order_nth: float
    frequency_sample_rate_hz: float
    frequency_averaging_hz: float


def chunk_by(array: np.ndarray, chunk_size: int) -> Iterator[np.ndarray]:
    """
    Iterates over "chunks" of an array.
    :param array: The array to iterator over.
    :param chunk_size: The size of each chunk.
    :return: An iterator over "chunks" of the array.
    """
    start: int = 0
    while start + chunk_size <= len(array):
        yield array[start : start + chunk_size]
        start += chunk_size


def chunked_stockwell(config: StockwellConfig, samples: np.ndarray) -> None:
    """
    Computes the Stockwell Transform over chunks of data.
    :param config: The Stockwell config.
    :param samples: The samples to chunk and then transform.
    """

    # Compute times for the full output buffer and frequencies for the given input params.
    time_s: np.ndarray = np.arange(config.n_cols) / config.frequency_sample_rate_hz
    freqs: np.ndarray
    if config.band_type_lin:
        freqs = stx.stx_linear_frequencies(
            config.band_order_nth, config.frequency_sample_rate_hz, config.frequency_averaging_hz
        )
    else:
        freqs = stx.stx_octave_band_frequencies(
            config.band_order_nth, config.frequency_sample_rate_hz, config.frequency_averaging_hz
        )

    # Build full output buffer
    out: np.ndarray = np.zeros((len(freqs), config.n_cols))

    total_chunks: int = len(samples) // config.n_chunk

    # For each chunk, perform the transform and update the output
    i: int
    chunk: np.ndarray
    for i, chunk in enumerate(chunk_by(samples, config.n_chunk)):
        # Run stockwell
        # TODO: Incorrect -- what can be done to fix? Persistent Gaussian window for full output?
        # TODO: Don't recompute state on each chunk
        chunked_out: np.ndarray = stx.stx_power_any_scale_pow2(
            config.band_order_nth, chunk, config.frequency_sample_rate_hz, freqs
        )

        # Update output
        # TODO: Shift old data off the end
        for r in range(chunked_out.shape[0]):
            for c in range(chunked_out.shape[1]):
                out[r][c + (i * config.n_chunk)] = 10.0 * math.log10(chunked_out[r][c])

        # Show result
        scale: str = "linear" if config.band_type_lin else "log"
        plt.figure()
        plt.pcolormesh(time_s, freqs, out)
        plt.yscale(scale)
        plt.ylabel(f"{scale} Frequency, hz")
        plt.xlabel("Time, s")
        plt.title(f"{i + 1} / {total_chunks}")
        plt.colorbar()


def main() -> None:
    sr_hz: float
    samples: np.ndarray
    (sr_hz, samples) = wavfile.read("/Users/mgarces/Documents/DATA_2022/ROCKET_ML/Ukraine_ML/ukraine_48khz.wav")

    print(f"Loaded {len(samples)} samples @ {sr_hz}")

    # Let's only use the closest power of 2 samples
    n_cols: int = 2 ** math.floor(math.log2(len(samples)))

    print(f"Using output buf of size {2}^{math.floor(math.log2(len(samples)))} ({n_cols})")

    config: StockwellConfig = StockwellConfig(n_cols, 2**18, False, 3, sr_hz, 1)
    chunked_stockwell(config, samples[:n_cols])
    plt.show()


if __name__ == "__main__":
    main()
