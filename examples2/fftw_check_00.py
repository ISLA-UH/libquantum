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



