import numpy as np
import scipy.optimize as opt
import scipy.ndimage as ndi
from collections import deque
from spt3g import core  #pylint: disable=import-error
import os
import sys


class CHWPDataProcess(object):
    def __init__(self, edge_scalar=1140):
        # Size of pakcets sent from the BBB
        self._pkt_size = 150
        # Maximum counter value from the BBB
        self._max_cnt = 0x5FFFFFFF
        # Number of encoder slits per HWP revolution
        self._num_edges = edge_scalar
        self._delta_angle = 2 * np.pi / self._num_edges
        self._ref_edges = 2
        self._edges_per_rev = self._num_edges - self._ref_edges
        self._filter_weights = np.full(self._edges_per_rev, 1/self._edges_per_rev)
        # Allowed jitter in slit width
        self._dev = 0.1  # 10%
        # Information about the bits that are saved
        self._nbits = 32
        # Arrays to be stuffed
        self._encd_quad = []
        self._encd_cnt = []
        self._encd_clk = []
        self._irig_tme = []
        self._irig_clk = []

    def __call__(self, frame):
        if (frame.type == core.G3FrameType.Timepoint and
           ('chwp_encoder_clock' not in frame.keys() and
           'chwp_irig_clock' not in frame.keys())):
            return [frame]
        elif (frame.type == core.G3FrameType.Timepoint and
             'chwp_encoder_clock' in frame.keys()):
            self._encd_quad.append(frame['chwp_encoder_quad'])
            self._encd_clk.append(frame['chwp_encoder_clock'])
            self._encd_cnt.append(frame['chwp_encoder_count'])
            return [frame]
        elif (frame.type == core.G3FrameType.Timepoint and
             'chwp_irig_clock' in frame.keys()):
            self._irig_clk.append(frame['chwp_irig_clock'])
            self._irig_tme.append(frame['chwp_irig_time'])
            return [frame]
        else:
            # End of scan -- process the data
            self._encd_quad = np.array(self._encd_quad).flatten()
            self._encd_clk = np.array(self._encd_clk).flatten()
            self._encd_cnt = np.array(self._encd_cnt).flatten()
            self._irig_clk = np.array(self._irig_clk).flatten()
            self._irig_tme = np.array(self._irig_tme).flatten()
            # Calculate angle vs time
            self._angle_time()
            
            # Return a frame with the angle and time
            out_frame = core.G3Frame(core.G3FrameType.Timepoint)
            out_frame['chwp_encoder_quad'] = core.G3VectorUInt(self._encd_quad)
            out_frame['chwp_encoder_clock'] = core.G3VectorUInt(self._encd_clk)
            out_frame['chwp_encoder_count'] = core.G3VectorUInt(self._encd_cnt)
            out_frame['chwp_irig_clock'] = core.G3VectorUInt(self._irig_clk)
            out_frame['chwp_irig_time'] = core.G3VectorUInt(self._irig_tme)
            out_frame['chwp_time'] = core.G3VectorDouble(self._time)
            out_frame['chwp_angle'] = core.G3VectorDouble(self._angle)
            out_frame['chwp_dropped_packets'] = core.G3UInt(int(self._num_dropped_pkts))
            return [out_frame, core.G3Frame(core.G3FrameType.EndProcessing)]

    def _angle_time(self):
        # Account for counter overflows
        self._flatten_counter()
        # Identify the reference slits
        self._find_refs()
        # "Fill" in these reference slits
        self._fill_refs()
        # Identifies and removes glitches
        self._fix_glitches()
        # Find any dropped packets
        self._find_dropped_packets()
        # Store the clock and count
        self._clock = self._encd_clk
        self._count = self._encd_cnt
        # Toss any encoder lines before the first IRIG clock value
        self._begin_trunc = len(self._encd_clk[self._encd_clk < self._irig_clk[0]])
        self._end_trunc = len(self._encd_clk[self._encd_clk > self._irig_clk[-1]]) + 1
        self._encd_clk = self._encd_clk[self._begin_trunc:-self._end_trunc]
        self._encd_cnt = self._encd_cnt[self._begin_trunc:-self._end_trunc]
        # Find the time
        self._time = np.interp(self._encd_clk, self._irig_clk, self._irig_tme)
        # Find the angle
        self._calc_angle_linear()
        return

    def _flatten_counter(self):
        cnt_diff = np.diff(self._encd_cnt)
        loop_indexes = np.argwhere(cnt_diff <= -(self._max_cnt-1)).flatten()
        for ind in loop_indexes:
            self._encd_cnt[(ind+1):] += -(cnt_diff[ind]-1)
        return

    def _find_refs(self):
        """ Find reference slits """
        # Calculate spacing between all clock values
        self._encd_diff = np.ediff1d(self._encd_clk, to_begin=2*self._encd_clk[0]-self._encd_clk[1])
        # Define median value as nominal slit distance
        self._slit_dist = ndi.convolve(self._encd_diff, self._filter_weights, mode='reflect')
        # Conditions for idenfitying the ref slit
        # Slit distance somewhere between 2 slits:
        # 2 slit distances (defined above) +/- 10%
        ref_hi_cond = ((self._ref_edges + 1) *
            self._slit_dist * (1 + self._dev))
        ref_lo_cond = ((self._ref_edges + 1) *
            self._slit_dist * (1 - self._dev))
        # Find the reference slit locations (indexes)
        self._ref_indexes = np.argwhere(np.logical_and(
            self._encd_diff < ref_hi_cond,
            self._encd_diff > ref_lo_cond)).flatten()
        # Define the reference slit line to be the line before
        # the two "missing" lines
        # Store the count and clock values of the reference lines
        self._ref_clk = np.take(self._encd_clk, self._ref_indexes)
        self._ref_cnt = np.take(self._encd_cnt, self._ref_indexes)
        return

    def _fill_refs(self, interp=False):
        """ Fill in the reference edges """
        # If no references, say that the first sample is theta = 0
        # This case comes up for testing with a function generator
        if len(self._ref_clk) == 0:
            self._ref_clk = [self._encd_clk[0]]
            self._ref_cnt = [self._encd_cnt[0]]
            return
        # Loop over all of the reference slits
        for ii in range(len(self._ref_indexes)):
            # Location of this slit
            ref_index = self._ref_indexes[ii]
            if interp:
                # Linearly interpolate the missing slits
                clks_to_add = np.linspace(
                    self._encd_clk[ref_index-1], self._encd_clk[ref_index],
                    self._ref_edges + 2)[1:-1]
                self._encd_clk = np.insert(self._encd_clk, ref_index, clks_to_add)
                # Adjust the encoder count values for the added lines
                # Add 2 to all future counts and interpolate the counts
                # for the two added slits
                self._encd_cnt[ref_index:] += self._ref_edges
                cnts_to_add = np.linspace(
                    self._encd_cnt[ref_index-1], self._encd_cnt[ref_index],
                    self._ref_edges + 2)[1:-1]
                self._encd_cnt = np.insert(self._encd_cnt, ref_index, cnts_to_add)
                # Also adjsut the reference count values in front of
                # this one for the added lines
                self._ref_cnt[ii+1:] += self._ref_edges
                # Adjust the reference index values in front of this one
                # for the added lines
                self._ref_indexes[ii+1:] += self._ref_edges
            else:
                self._encd_cnt[ref_index:] += self._ref_edges
                self._ref_cnt[ii+1:] += self._ref_edges
        return

    def _fix_glitches(self):
		def find_rot_start_type(diffs, high, low, loop=True):
			diff_sum = 0
			prev_res = {'value': 2**30}
			for ii, diff in enumerate(diffs[1:]):
				diff_sum += diff
				res_high = abs(high-diff_sum)
				res_low = abs(low-diff_sum)

				if prev_res['value'] < min(res_high, res_low):
					if prev_res['count'] < 10:
						prev_res['count'] += 1
					else:
						if loop:
							next_res = find_rot_start_type(diffs[1+prev_res['index']:], 
														   high, low, loop=False)
							if next_res['type'] != prev_res['type']:
								return True, prev_res['type']
							else:
								print('Warning: Could not determine duty cycle')
								return False, 'error'
						else:
							return prev_res

				else:
					prev_res = {'value': min(res_high, res_low),
								'type': 'high' if res_high < res_low else 'low',
								'count': 0,
								'index': ii}

			print('Warning: Unexpected glitch type')
			return False, 'error'

		def glitch_mask(diffs, high, low, start):
			return_mask = []
			toggle = not start
			diff_sum = 0
			prev_res = 0
			for ii, diff in enumerate(diffs[1:]):
				diff_sum += diff
				res = abs(high-diff_sum) if toggle else abs(low-diff_sum)

				if prev_res < res:
					return_mask.append(True)
					diff_sum = diff
					toggle = not toggle
					res = abs(high-diff_sum) if toggle else abs(low-diff_sum)
				else:
					return_mask.append(False)

				prev_res = res
			else:
				return_mask.append(True)

			if np.sum(return_mask) == self._edges_per_rev:
				return return_mask
			else:
				print('Warning: Could not remove glitches from rotation')
				return return_mask*False

		glitched_rots = np.ediff1d(self._ref_cnt, to_end=self._ref_cnt[-1]+self._num_edges) \
				!= self._num_edges
		for ii, ref_ind in enumerate(self._ref_indexes):
			if glitched_rots[ii]:
				rot_encd_clk = self._encd_clk[ref_ind:self._ref_indexes[ii+1]]
				rot_encd_diff = self._encd_diff[ref_ind:self._ref_indexes[ii+1]]
				rot_slit_dist = self._slit_dist[ref_ind:self._ref_indexes[ii+1]]

				ref_high_med = np.median(rot_encd_diff[np.where(rot_encd_diff > rot_slit_dist)])
				ref_low_med = np.median(rot_encd_diff[np.where(rot_encd_diff < rot_slit_dist)])

				response, start_type = find_rot_start_type(rot_encd_diff, ref_high_med, ref_low_med)

				if not response:
					rot_mask = np.full(len(rot_encd_diff), False)
				else:
					rot_mask = glitch_mask(rot_encd_diff, ref_high_med, ref_low_med, start_type)

				num_glitches = len(rot_mask) - np.sum(rot_mask)
				if num_glitches == 0:
					continue

				self._ref_indexes[ii+1:] -= num_glitches
				self._encd_clk[ref_ind:ref_indexes[ii+1]] = \
						rot_encd_clk[rot_mask]
				self._encd_cnt[]

    def _find_dropped_packets(self):
        """ Estimate the number of dropped packets """
        cnt_diff = np.diff(self._encd_cnt)
        dropped_samples = np.sum(cnt_diff[cnt_diff >= self._pkt_size])
        self._num_dropped_pkts = dropped_samples // (self._pkt_size - 1)
        return
    
    def _calc_angle_linear(self):
        self._angle = (self._encd_cnt - self._ref_cnt[0]) * self._delta_angle
        return
