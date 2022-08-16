"""
A template for an iterative version of the Stockwell Transform.
"""
from dataclasses import dataclass

import libquantum.styx_stx_loopy as stx

import numpy as np


class ShiftingBuf:
    def __init__(self, n_cols: int, n_rows: int = 1) -> None:
        if n_cols <= 0:
            raise RuntimeError(f"n_cols={n_cols} must be >0")

        if n_rows <= 0:
            raise RuntimeError(f"n_rows={n_rows} must be >0")

        self.n_rows = n_rows
        self.n_cols = n_cols

        self.buf: np.ndarray = np.zeros((n_rows, n_cols))

        self.len_cols: int = 0

    def update(self, vals: np.ndarray, cnt: int) -> None:
        if cnt <= 0:
            raise RuntimeError(f"cnt={cnt} must be >0")

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

    n_total_cols: int
    n_reduced_cols: int
    n_chunk: int
    band_type_lin: bool
    band_order_nth: float
    frequency_sample_rate_hz: float
    frequency_averaging_hz: float


class StockwellTransform:
    # TODO: Narrow down params
    def __init__(self, config: StockwellConfig) -> None:
        self.config: StockwellConfig = config
        self.frequencies: np.ndarray = self.__frequencies()

        self.in_buf: ShiftingBuf = ShiftingBuf(config.n_chunk)
        self.out_buf: ShiftingBuf = ShiftingBuf(config.n_reduced_cols, len(self.frequencies))

        # TODO: Init stockwell state -- all other "constants"

    def update_input(self, vals: np.ndarray, cnt: int) -> None:
        self.in_buf.update(vals, cnt)

    def run(self) -> None:
        res: np.ndarray = self.__run_stockwell()
        res = self.__reduce_res(res)
        res = self.__convert_to_db(res)
        self.__update_spect(res)

    def plot_res(self):
        pass

    def __run_stockwell(self) -> np.ndarray:
        pass

    def __reduce_res(self, res: np.ndarray) -> np.ndarray:
        pass

    def __convert_to_db(self, res: np.ndarray) -> np.ndarray:
        pass

    def __update_spect(self, res: np.ndarray) -> None:
        pass

    def __frequencies(self) -> np.ndarray:
        if self.config.band_type_lin:
            return stx.stx_linear_frequencies(
                self.config.band_order_nth, self.config.frequency_sample_rate_hz, self.config.frequency_averaging_hz
            )

        return stx.stx_octave_band_frequencies(
            self.config.band_order_nth, self.config.frequency_sample_rate_hz, self.config.frequency_averaging_hz
        )
