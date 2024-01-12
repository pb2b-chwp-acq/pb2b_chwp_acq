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
        self._pkt_size = 120
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
        # Store the clock
        self._clock = self._encd_clk
        # Toss any encoder lines before the first IRIG clock value
        self._begin_trunc = len(self._encd_clk[self._encd_clk < self._irig_clk[0]])
        self._end_trunc = len(self._encd_clk[self._encd_clk > self._irig_clk[-1]]) + 1
        self._encd_clk = self._encd_clk[self._begin_trunc:-self._end_trunc]
		self._angle = self._angle[self._begin_trunc:-self._end_trunc]
        # Find the time
        self._time = np.interp(self._encd_clk, self._irig_clk, self._irig_tme)
        return

    def _find_refs(self):
        """ Find reference slits """
        # Calculate spacing between all clock values
        self._encd_diff = np.ediff1d(self._encd_clk, to_begin=self._encd_clk[0]-self._encd_clk[1])
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
        return

    def _fill_refs(self):
        """ Fill in the reference edges """
        # If no references, say that the first sample is theta = 0
        # This case comes up for testing with a function generator
        if len(self._ref_clk) == 0:
            self._ref_clk = [self._encd_clk[0]]
            return
        # Loop over all of the reference slits
        for ii in range(len(self._ref_indexes)):
            # Location of this slit
            ref_index = self._ref_indexes[ii]
			
            # Interpolate the missing slits
			try:
				rot_encd_diff = self._encd_diff[ref_index, self._ref_indexes[ii+1]]
				rot_slit_dist = self._slit_dist[ref_index, self._ref_indexes[ii+1]]
			except IndexError:
				rot_encd_diff = self._encd_diff[self._ref_indexes[ii-1], ref_index]
				rot_slit_dist = self._slit_dist[self._ref_indexes[ii-1], ref_index]

			high = np.median(rot_encd_diff[np.where(rot_encd_diff > rot_slit_dist)])
			low = np.median(rot_encd_diff[np.where(rot_encd_diff < rot_slit_dist)])

            clks_to_add = np.linspace(
                self._encd_clk[ref_index-1], self._encd_clk[ref_index],
                self._ref_edges + 2)

			res_high = abs(clks_to_add[-1] - clks_to_add[0] - 2*high - low)
			res_low = abs(clks_to_add[-1] - clks_to_add[0] - high - 2*low)

			first = high if res_high < res_low else low
			second = low if res_high < res_low else high

			ratio1 = first/(2*first + second)
			ratio2 = (first + second)/(2*first + second)

			clks_to_add[1] = clks_to_add[0] + (clks_to_add[3]-clks_to_add[0])*ratio1
			clks_to_add[2] = clks_to_add[0] + (clks_to_add[3]-clks_to_add[0])*ratio2
            self._encd_clk = np.insert(self._encd_clk, ref_index, clks_to_add[1:-1])

			# Adjust the clock difference array
			diffs_to_add = np.diff(clks_to_add)
			self._encd_diff = np.insert(np.delete(self._encd_diff, ref_index), 
										ref_index, diffs_to_add)

			# Adjust the slit distance array
			dists_to_add = np.ones(2)*(first+second)/2
			self._slit_dist = np.insert(self._slit_dist, ref_index, dists_to_add)
            # Adjust the reference index values in front of this one
            # for the added lines
            self._ref_indexes[ii+1:] += self._ref_edges
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
			for diff in diffs:
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
				return_mask = return_mask[1:]

			if np.sum(return_mask) == self._num_edges:
				return True, return_mask
			else:
				print('Warning: Could not remove glitches from rotation')
				return False, np.full(len(return_mask), False)

		def generate_angles(high, low, start):
			duty_cor = np.array([0,(1 if start == 'high' else -1)*(high-low)/(high+low)])
			angle = np.arange(self._num_edges).reshape(int(self._num_edges/2),2)
			angle = self._delta_angle*(angle + duty_cor).flatten()
			return angle

		glitched_rots = np.ediff1d(self._ref_indexes, to_end=self._num_edges) \
				!= self._num_edges
		dead_rots = []
		self._angle = np.zeros(len(self._encd_clk))
		for ii, ref_ind in enumerate(self._ref_indexes):
			if ref_ind == self._ref_indexes[-1]:
				rot_encd_clk = self._encd_clk[ref_ind:]
				rot_encd_diff = self._encd_diff[ref_ind:]
				rot_slit_dist = self._slit_dist[ref_ind:]
			else:
				rot_encd_clk = self._encd_clk[ref_ind:self._ref_indexes[ii+1]]
				rot_encd_diff = self._encd_diff[ref_ind:self._ref_indexes[ii+1]]
				rot_slit_dist = self._slit_dist[ref_ind:self._ref_indexes[ii+1]]

			high = np.median(rot_encd_diff[np.where(rot_encd_diff > rot_slit_dist)])
			low = np.median(rot_encd_diff[np.where(rot_encd_diff < rot_slit_dist)])

			resp1, start_type = find_rot_start_type(rot_encd_diff, high, low)

			if glitched_rots[ii]:
				if not resp1:
					rot_mask = np.full(len(rot_encd_diff), False)
					dead_rots.append(ii)
				else:
					resp2, rot_mask = glitch_mask(rot_encd_diff, high, low, start_type)
					if not resp2:
						dead_rots.append(ii)

				num_glitches = len(rot_mask) - np.sum(rot_mask)
				if num_glitches == 0:
					continue

				self._encd_clk = np.delete(self._encd_clk, ref_ind + np.arange(num_glitches))
				self._encd_clk[ref_ind:ref_ind + np.sum(rot_mask)] = \
						rot_encd_clk[rot_mask]

				self._encd_diff = np.delete(self._encd_diff, ref_ind + np.arange(num_glitches))
				self._encd_diff[ref_ind:ref_ind + np.sum(rot_mask)] = \
						np.ediff1d(rot_encd_clk[rot_mask], to_begin=rot_encd_diff[0])

				temp_angle = generate_angles(high, low, start_type)
				self._angle = np.delete(self._angle, ref_ind + np.arange(num_glitches))
				self._angle[ref_ind:ref_ind + np.sum(rot_mask)] = temp_angle

				self._ref_indexes[ii+1:] -= num_glitches
			else:
				temp_angle = generate_angles(high, low, start_type)
				self._angle[ref_ind:ref_ind + self._num_edges] = temp_angle
		
        self._slit_dist = ndi.convolve(self._encd_diff, self._filter_weights, mode='reflect')
		self._ref_indexes = np.delete(self._ref_indexes, dead_rots)

	def _flatten_counter(self):
		cnt_diff = np.diff(self._encd_cnt)
		loop_indexes = np.argwhere(cnt_diff <= -(self._max_cnt-1)).flatten()
		for ind in loop_indexes:
			self._encd_cnt[(ind+1):] += -(cnt_diff[ind]-1)
		return

    def _find_dropped_packets(self):
        """ Estimate the number of dropped packets """
		cnt_diff = np.diff(self._encd_cnt)
        dropped_samples = np.sum(cnt_diff[cnt_diff >= self._pkt_size])
        self._num_dropped_pkts = dropped_samples // (self._pkt_size - 1)
        return

    def _calc_angle_linear(self):
		# Needs to be fixed
		self._angle = np.zeros(len(self._encd_clk))
		for ii, ref_ind in enumerate(self._ref_indexes):
			rot_encd_diff = self._encd_diff[ref_ind:ref_ind+self._num_edges]
			rot_slit_dist = self._slit_dist[ref_ind:ref_ind+self._num_edges]

			high = np.median(rot_encd_diff[np.where(rot_encd_diff > rot_slit_dist)])
			low = np.median(rot_encd_diff[np.where(rot_encd_diff < rot_slit_dist)])

		self._angle = (self._encd_cnt - self._ref_cnt[0]) * self._delta_angle
        return
