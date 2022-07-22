"""
chirp_gamma.py
Introduction to Time-Frequency Representations (TFRs).
Compute Fast Fourier Transform (FFT) on simple tones to verify amplitudes
The foundation  of efficient TFR computation is the FFT.
For N = number of points, computation scales as N log N instead of N**2
Case study:
Sinusoid input with unit amplitude
Validate:
FFT power averaged over the signal duration is 1/2
RMS amplitude = 1/sqrt(2)

"""
import numpy as np
import matplotlib.pyplot as plt
from libquantum.utils import is_power_of_two
print(__doc__)


def main():
    

if __name__ == "__main__":
    main()