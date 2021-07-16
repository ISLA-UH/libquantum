import unittest
import numpy as np

start_time = 0
end_time = 10
sample_rate = 100
signal_time = np.arange(start_time, end_time, 1/sample_rate)
frequency = 3
amplitude = 1
theta = 0
sinewave = amplitude * np.sin(2 * np.pi * frequency * signal_time + theta)


class TestUnevenSensor(unittest.TestCase):
    def setUp(self) -> None:
        # Create signal
        self.start_time = 0
        self.end_time = 10
        self.sample_rate = 100
        self.signal_time = np.arange(self.start_time, self.end_time, 1/self.sample_rate)
        self.frequency = 3
        self.amplitude = 1
        self.theta = 0
        self.sinewave = self.amplitude * np.sin(2 * np.pi * self.frequency * self.signal_time + self.theta)


    def tearDown(self):
        self.example_station = None
        self.sensor_sample_rate_hz = None
        self.sensor_epoch_s = None
        self.sensor_raw = None
        self.sensor_nans = None



if __name__ == '__main__':
    unittest.main()