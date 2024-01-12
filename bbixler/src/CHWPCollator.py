from spt3g import core  # pylint: disable=import-error


@core.indexmod
class CHWPCollator(object):
    """
    Move CHWP data to Scan frames, optionally deleting the
    original Timepoint frames where that data lives
    """
    def __init__(self, drop_timepoints=True, write_to_file=False):
        self.drop_timepoints = drop_timepoints
        # Buffer of frames that will be filled during
        # scans and emptied at a new scan start
        self.buffer = []
        # Whether or not we're buffering frames.
        # Most of the time will be True
        self.waiting = False
        # Generate the HWP data lists
        self._generate_g3_dicts()

    def __call__(self, frame):
        # frames to pass to the next module.  Should be either:
        # - just one frame when we haven't hit the first Scan frame yet
        # - no frames while we're processing a whole Scan's worth
        # - a whole Scan's worth of frames once we're done with a scan
        ret_frames = []

        if frame.type == core.G3FrameType.Timepoint:
            # Store encoder data
            if 'chwp_encoder_count' in frame.keys():
                # Store the data
                for i in range(len(frame['chwp_encoder_count'])):
                    self.encoder_count.append(frame['chwp_encoder_count'][i])
                    self.encoder_clock.append(frame['chwp_encoder_clock'][i])
                # Drop the original frames if requested
                if not self.drop_timepoints:
                    ret_frames.append(frame)
                return ret_frames
            # Store quadrature data
            elif 'chwp_encoder_quad' in frame.keys():
                # Store the data
                self.encoder_quad.append(frame['chwp_encoder_quad'])
                # Drop the original frames if requested
                if not self.drop_timepoints:
                    ret_frames.append(frame)
                return ret_frames
            # Store IRIG data
            elif 'chwp_irig_time' in frame.keys():
                # Store the data
                self.irig_time.append(frame['chwp_irig_time'])
                self.irig_clock.append(frame['chwp_irig_clock'])
                # There should be 10 IRIG synchronization pulses
                # for i in range(len(frame['chwp_irig_synch'])):
                #    self.irig_synch.append(frame['chwp_irig_synch'][i])
                # Drop the original frames if requested
                if not self.drop_timepoints:
                    ret_frames.append(frame)
                return ret_frames

        # If this is the last scan, dump to the current scan frame
        if frame.type == core.G3FrameType.EndProcessing:
            if self.waiting:
                self._dump_to_scan_frame(ret_frames)
            # let the rest of the pipeline see the EndProcessing frame too
            ret_frames.append(frame)
            return ret_frames

        # If we arrive at a scan boundary, save stuff from the previous scan
        if frame.type == core.G3FrameType.Scan:
            if self.waiting:
                self._dump_to_scan_frame(ret_frames)
            assert(len(self.buffer) == 0)  # hopefully this is never an issue
            self.buffer.append(frame)  # now start the current scan
            self.waiting = True
            return ret_frames

        # Don't do any processing on other kinds of frames
        if self.waiting:
            self.buffer.append(frame)
        else:
            ret_frames.append(frame)
        return ret_frames

    def _dump_to_scan_frame(self, ret_frames):
        # Append the vectors to the Scan frame (first item in the buffer)
        self.buffer[0]['chwp_encoder_count'] = self.encoder_count
        self.buffer[0]['chwp_encoder_clock'] = self.encoder_clock
        self.buffer[0]['chwp_encoder_quad'] = self.encoder_quad
        self.buffer[0]['chwp_irig_time'] = self.irig_time
        self.buffer[0]['chwp_irig_clock'] = self.irig_clock
        # self.buffer[0]['chwp_irig_synch'] = self.irig_synch

        # Return frames to the next module and clear the buffer.
        # This should be 1 scan's worth of data
        ret_frames += self.buffer
        self.buffer = []

        # Even though this may be set to True immediately,
        # just in case wait for it to be set again
        self.waiting = False

        # Save the data to a test file if requested

        # Clear data from the hwp data lists
        self._generate_g3_dicts()
        return

    def _generate_g3_dicts(self):
        # Dictionaries of double vectors that will hold CHWP data
        # Data must be type casted from doubles to uint32_t's later
        self.encoder_count = core.G3VectorUInt()
        self.encoder_clock = core.G3VectorUInt()
        self.encoder_quad = core.G3VectorUInt()
        # Dictionaries of double vectors that will hold the IRIG data
        self.irig_time = core.G3VectorDouble()
        self.irig_clock = core.G3VectorUInt()
        # self.irig_synch = core.G3VectorDouble()
        return
