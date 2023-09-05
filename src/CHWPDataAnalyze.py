import numpy as np
import scipy.optimize as opt
from collections import deque
from spt3g import core  #pylint: disable=import-error
import os
import sys

class CHWPDataAnalyze(object):
    def __init__(self):
        # Bin size for velocity and acceleration vectors
        self._bin_size = 100
        # Polynomial degree for fitting for jitter
        self._jitter_poly_deg = 2
    
    def __call__(self, frame):
        """
        Input frame includes angle and time
        Output frame includes:
        -- angle vs time
        -- jitter vs time
        -- velocity vs time (smoothed)
        -- angle vs frequency
        -- power vs frequency
        """
        if not (frame.type == core.G3FrameType.Timepoint and
                'chwp_time' in frame.keys()):
            return [frame]

        # Save the input data to be processed
        self._encoder_clock = self._arr(frame['chwp_encoder_clock'])
        self._encoder_count = self._arr(frame['chwp_encoder_count'])
        self._irig_clock = self._arr(frame['chwp_irig_clock'])
        self._irig_time = self._arr(frame['chwp_irig_time'])
        self._time = self._arr(frame['chwp_time'])
        self._angle = self._arr(frame['chwp_angle'])

        # Calculate the time-ordered data
        self._calc_data()
        # Calculate the time-ordered jitter data
        self._calc_jitter()
        # Calculate PSDs
        self._calc_psds()

        # Populate passed frame with processed data
        frame['chwp_encoder_clock_jitter'] = self._g3(
            self._encoder_clock_jitter)
        frame['chwp_irig_clock_jitter'] = self._g3(self._irig_clock_jitter)
        frame['chwp_angle_jitter'] = self._g3(self._angle_jitter)

        frame['chwp_velocity'] = self._g3(self._velocity)
        frame['chwp_accel'] = self._g3(self._accel)
        frame['chwp_time_smooth'] = self._g3(self._time_smooth)
        frame['chwp_velocity_smooth'] = self._g3(self._velocity_smooth)
        frame['chwp_accel_smooth'] = self._g3(self._accel_smooth)

        frame['chwp_encoder_clock_jitter_freq'] = self._g3(
            self._encoder_clock_jitter_freq)
        frame['chwp_encoder_clock_jitter_psd'] = self._g3(
            self._encoder_clock_jitter_psd)
        frame['chwp_irig_clock_jitter_freq'] = self._g3(
            self._irig_clock_jitter_freq)
        frame['chwp_irig_clock_jitter_psd'] = self._g3(
            self._irig_clock_jitter_psd)
        frame['chwp_angle_jitter_freq'] = self._g3(self._angle_jitter_freq)
        frame['chwp_angle_jitter_psd'] = self._g3(self._angle_jitter_psd)

        # Return frames for further processing
        return [frame]

    # ***** Helper Methods *****
    def _calc_data(self):
        """ Calculate and smooth the TOD """
        # Smoothed time
        self._time_smooth = self._smooth(self._time, self._bin_size)
        # Velocity vs time
        self._velocity = np.diff(self._angle) / np.diff(self._time)
        self._velocity_smooth = self._smooth(self._velocity, self._bin_size)
        # Acceleration vs time
        self._accel = np.diff(self._velocity) / np.diff(self._time[:-1])
        self._accel_smooth = self._smooth(self._accel, self._bin_size)
        return

    def _calc_jitter(self):
        """ Subtract a poly fit from select TOD """
        # Encoder clock jitter
        self._encoder_clock_jitter = self._jitter(
            self._encoder_count, self._encoder_clock, self._jitter_poly_deg)
        # IRIG clock jitter
        self._irig_clock_jitter = self._jitter(
            self._irig_time, self._irig_clock, self._jitter_poly_deg)
        # Angle jitter
        self._angle_jitter = self._jitter(
            self._time, self._angle, self._jitter_poly_deg)
        return

    def _calc_psds(self):
        # Encoder clock jitter PSD
        self._encoder_clock_jitter_freq, self._encoder_clock_jitter_psd = (
            self._psd(self._encoder_count, self._encoder_clock_jitter))
        # IRIG clock jitter PSD
        self._irig_clock_jitter_freq, self._irig_clock_jitter_psd = (
            self._psd(self._irig_time, self._irig_clock_jitter))
        # Angle jitter PSD
        self._angle_jitter_freq, self._angle_jitter_psd = (
            self._psd(self._time, self._angle_jitter))
        return

    # ***** Static Helper Methods *****
    @staticmethod
    def _smooth(arr, bin_size):
        alen = len(arr)
        bin_trunc = alen % int(bin_size)
        arr_trunc = arr[:-bin_trunc]
        arr_binned = np.reshape(arr_trunc, (-1, bin_size))
        arr_smooth = np.mean(arr_binned, axis=-1).flatten()
        return arr_smooth

    @staticmethod
    def _jitter(xdata, ydata, deg):
        params = np.polyfit(xdata, ydata, deg=deg)
        yfit = np.polyval(params, xdata)
        return ydata - yfit

    @staticmethod
    def _psd(xarr, yarr):
        # Regularize x array
        xarri = np.linspace(xarr[0], xarr[-1], len(xarr))
        # Interpolate y array
        yarri = np.interp(xarri, xarr, yarr)
        # Calculate window function
        alen = len(xarri)
        window = np.hanning(len(xarri))
        # Calculate FFT normalization
        xspread = xarri[-1] - xarri[0]
        norm = (np.sqrt(xspread / (alen**2)) *
                (1. / (np.trapz(window) / alen)))
        # Calculate PSD x array
        xstep = xarri[1] - xarri[0]
        xpsd = np.fft.rfftfreq(alen, d=xstep)
        # Calculaate PSD y arr
        fft = norm * np.fft.rfft(yarri * window)
        ypsd = np.power(abs(fft), 2.)
        return xpsd, ypsd

    @staticmethod
    def _g3(arr):
        return core.G3VectorDouble(arr)

    @staticmethod
    def _arr(arr):
        return np.array(list(arr))
