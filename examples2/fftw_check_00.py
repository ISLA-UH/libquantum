import pyfftw
import scipy.signal
import numpy
from timeit import Timer


if __name__ == "__main__":
    a = pyfftw.empty_aligned((128, 64), dtype='complex128')
    b = pyfftw.empty_aligned((128, 64), dtype='complex128')

    a[:] = numpy.random.randn(128, 64) + 1j*numpy.random.randn(128, 64)
    b[:] = numpy.random.randn(128, 64) + 1j*numpy.random.randn(128, 64)

    t = Timer(lambda: scipy.signal.fftconvolve(a, b))

    print('Time with scipy.fftpack: %1.3f seconds' % t.timeit(number=100))

    # Monkey patch fftpack with pyfftw.interfaces.scipy_fftpack
    scipy.fftpack = pyfftw.interfaces.scipy_fftpack
    scipy.signal.fftconvolve(a, b) # We cheat a bit by doing the planning first

    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()

    print('Time with monkey patched scipy_fftpack: %1.3f seconds' %
          t.timeit(number=100))

    

    # # Gabor atom specifications
    # order = 12
    # scale_multiplier = 3/4*np.pi * order
    # omega = np.pi/16  # Only matters for adequate sampling. Nyquist is at pi
    #
    # # Atom scale
    # scale = scale_multiplier/omega
    #
    # # Chirp index gamma, blueshift
    # gamma = 2
    # mu = np.sqrt(1 + gamma**2)
    # chirp_scale = scale * mu
    #
    # # scale multiplier Mc
    # window_support_points = 2*np.pi*chirp_scale
    # # scale up
    # window_support_pow2 = 2**int((np.ceil(np.log2(window_support_points))))
    #
    # time0 = np.arange(window_support_pow2)
    # dt = np.mean(np.diff(time0))
    # sample_rate = 1/dt
    # time = time0 - time0[-1]/2
    #
    # chirp_phase = omega * time + 0.5 * gamma * (time / chirp_scale) ** 2
    # chirp_wf = np.exp(-0.5 * (time / chirp_scale) ** 2 + 1j * chirp_phase)
    # chirp_sig = np.real(chirp_wf)



