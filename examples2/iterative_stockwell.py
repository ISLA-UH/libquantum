"""
A template for an iterative version of the Stockwell Transform.
"""

from dataclasses import dataclass

import redvox.cloud.subscription as sub
from redvox.api1000.wrapped_redvox_packet.sensors.audio import Audio
from redvox.api1000.wrapped_redvox_packet.wrapped_packet import WrappedRedvoxPacketM
from redvox.cloud.client import cloud_client, CloudClient
from scipy.io import wavfile

import libquantum.styx_stx_loopy as stx

import numpy as np


class FifoBuf:
    """
    A FIFO backed by a numpy array.

    This FIFO will fill up until the desired number of columns have been reached. Once the buffer is full, additional
    updates to the buffer will shift elements off the left end of the buffer to make room for the new data.

    This buffer is generalized to support 2D arrays (such as spectrograms), but can also work with 1D arrays by
    setting the number of rows to 1.
    """

    def __init__(self, n_cols: int, n_rows: int = 1) -> None:
        """
        Initializes a 2D FIFO buffer.
        :param n_cols: The number of columns in the buffer.
        :param n_rows: The number of rows in the buffer (defaults to 1 to support 1D buffers).
        """
        if n_cols <= 0:
            raise RuntimeError(f"n_cols={n_cols} must be >0")

        if n_rows <= 0:
            raise RuntimeError(f"n_rows={n_rows} must be >0")

        self.n_rows = n_rows
        self.n_cols = n_cols

        # Zero out the buffer
        self.buf: np.ndarray = np.zeros((n_rows, n_cols))

        # Keeps track of how "full" this buffer is
        self.len_cols: int = 0

    def update(self, vals: np.ndarray, cnt: int) -> None:
        """
        Updates this buffer with new data.
        :param vals: The data to add to the buffer.
        :param cnt: The number of columns to copy from vals.
        """
        if cnt <= 0:
            return

        shape = vals.shape
        if len(shape) != 2 or shape[0] != self.n_rows:
            raise RuntimeError(f"Invalid shape={shape}. Expected={self.buf.shape}")

        # Determine how much to shift the input buffer by
        count: int = cnt if cnt < self.n_cols else self.n_cols
        shift_by: int = self.len_cols + count - self.n_cols
        shift_by = max(shift_by, 0)
        shift_by = min(shift_by, self.n_cols)

        # Perform the shift
        if 0 < shift_by < self.n_cols:
            self.buf = np.roll(self.buf, -shift_by)

        # Copy the new elements
        offset: int = self.len_cols - shift_by
        self.buf[0 : self.buf.shape[0], offset : offset + count] = vals[0 : vals.shape[0], 0:count]

        # Update the state
        total: int = self.len_cols + count
        self.len_cols = total if total < self.n_cols else self.n_cols


@dataclass
class StockwellConfig:
    """
    Required Stockwell Transform inputs.
    """

    n_chunk: int
    n_reduce: int
    n_cols: int
    frequencies: np.ndarray
    band_type_lin: bool
    band_order_nth: float
    frequency_sample_rate_hz: float


class StockwellTransform:
    """
    A stateful Stockwell transform.
    """

    def __init__(self, config: StockwellConfig) -> None:
        """
        Initializes the transform.
        :param config: The configuration for the transform.
        """
        self.config: StockwellConfig = config

        # Setup input and output buffers
        # 1D input buffer.
        self.in_buf: FifoBuf = FifoBuf(config.n_chunk)

        # 2D output buffer.
        self.out_buf: FifoBuf = FifoBuf(config.n_cols, len(config.frequencies))

        # TODO: Init stockwell state -- all other "constants" that can be precomputed on first run
        # frequency_stx
        # sigma_stx
        # gaussian envelopes
        # frequency_fft
        # omega fft
        # and others that I may have missed

    # -------------------- Public API --------------------------------

    def update_input(self, vals: np.ndarray, cnt: int) -> None:
        """
        Updates the input buffer with the provided values.
        :param vals: Values to update input buffer with.
        :param cnt: Total number of items to copy from values into the buffer.
        """
        self.in_buf.update(vals.reshape(1, cnt), cnt)

    def run(self) -> None:
        """
        Run's the stockwell transform, reduces the result, scales the values, and updates the spectrogram output buffer.
        """
        print(f"TODO: Running stockwell on input of len={self.in_buf.len_cols}")
        res: np.ndarray = self.__run_stockwell()
        res = self.__reduce_res(res)
        res = self.__convert_to_db(res)
        self.__update_spect(res)

    def plot_res(self):
        # TODO: Continually update a single figure
        pass

    # -------------------- Private API -------------------------------

    def __run_stockwell(self) -> np.ndarray:
        # TODO: Run the current chunk returning the result chunk.
        return np.array([[]])

    def __reduce_res(self, res: np.ndarray) -> np.ndarray:
        # TODO: Reduce the number of columns in the result chunk
        return np.array([[]])

    @staticmethod
    def __convert_to_db(res: np.ndarray) -> np.ndarray:
        """
        Converts the result to dB.
        :param res: Result to convert.
        :return: Values in dB.
        """
        return 10.0 * np.log10(res)

    def __update_spect(self, res: np.ndarray) -> None:
        """
        Updates the output buffer with the passed in reduced and scaled data.
        :param res: Data to update the spectogram with.
        """
        self.out_buf.update(res, res.shape[1])


def wav_chunks(wav_path: str, stockwell_transform: StockwellTransform):
    sr_hz: float
    samples: np.ndarray
    (sr_hz, samples) = wavfile.read(wav_path)
    print(f"Loaded {len(samples)} samples @ {sr_hz}")

    start: int = 0
    chunk_size: int = stockwell_transform.config.n_chunk
    while start + chunk_size <= len(samples):
        chunk: np.ndarray = samples[start : start + chunk_size]
        stockwell_transform.update_input(chunk, len(chunk))
        stockwell_transform.run()
        stockwell_transform.plot_res()
        start += chunk_size


def extract_audio(packet: WrappedRedvoxPacketM) -> np.ndarray:
    audio: Audio = packet.get_sensors().get_audio()
    return audio.get_samples().get_values()


def real_time_chunks(stockwell_transform: StockwellTransform):
    client: CloudClient
    with cloud_client() as client:
        pub_msg: sub.PubMsg
        for pub_msg in sub.subscribe_packet("wss://redvox.io/subscribe", client, ["1637620002"]):
            print("Recv subscribed packet from cloud")
            packet: WrappedRedvoxPacketM = pub_msg.msg
            chunk: np.ndarray = extract_audio(packet)
            stockwell_transform.update_input(chunk, len(chunk))
            stockwell_transform.run()
            stockwell_transform.plot_res()


def main() -> None:
    # TODO: Pre-compute powers of twos based off of input reqs

    sr_hz: float = 48_000
    is_linear_band: bool = False
    band_order_nth: float = 12
    averaging_freq_hz: float = 50

    frequencies: np.ndarray
    if is_linear_band:
        frequencies = stx.stx_linear_frequencies(band_order_nth, sr_hz, averaging_freq_hz)
    else:
        frequencies = stx.stx_octave_band_frequencies(band_order_nth, sr_hz, averaging_freq_hz)

    # Configure the transform
    stockwell_config: StockwellConfig = StockwellConfig(
        2**16,  # Size per chunk
        4,  # TODO: Column reduction amount
        2**22,  # Final number of columns in output spect
        frequencies,  # Pre-computed frequencies (rows)
        is_linear_band,  # True for linear bands, False for Octave
        band_order_nth,  # nth band order
        sr_hz,  # Samples SR Hz
    )
    stockwell_transform: StockwellTransform = StockwellTransform(stockwell_config)

    # Choose one of the following two chunk methods and comment out the other
    wav_chunks("/home/opq/data/ml/ukraine_48khz.wav", stockwell_transform)
    # real_time_chunks(stockwell_transform)


if __name__ == "__main__":
    main()
