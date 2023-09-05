from spt3g import core  # pylint: disable=import-error
from spt3g import chwp  # pylint: disable=import-error
import numpy as np

num_frames = 0


def dummy_data(frame, frames_to_generate, sample_rate=152):
    """
    Adds a bunch of dfmux, encoder, and irig timepoint frames
    For now, just store zeroes in all the entries
    Match the types, but not necessarily the lengths, of the data we'll get
    """
    global num_frames
    out_frames = []
    while num_frames < frames_to_generate:
        # Create a DfMux frame. Store a zero for testing
        dfmux = core.G3Frame(core.G3FrameType.Timepoint)
        dfmux['DfMux'] = 0
        out_frames.append(dfmux)

        # Create CHWP encoder frame. Stuff with zeros for testing
        encoder = core.G3Frame(core.G3FrameType.Timepoint)
        encoder['chwp_encoder_clock'] = core.G3VectorUInt(
            np.zeros(1000, dtype=np.uint32))
        encoder['chwp_encoder_count'] = core.G3VectorUInt(
            np.zeros(1000, dtype=np.uint32))
        out_frames.append(encoder)

        # IRIG frames once every second. Stuff with zeros for testing
        if num_frames % sample_rate == 0:
            irig = core.G3Frame(core.G3FrameType.Timepoint)
            irig['chwp_irig_time'] = core.G3UInt(int(float(
                core.G3Time.Now().time) / float(core.G3Units.seconds)))
            irig['chwp_irig_clock'] = core.G3UInt(0)
            irig['chwp_irig_synch'] = core.G3VectorUInt(
                np.zeros(10, dtype=np.uint32))
            out_frames.append(irig)

        num_frames += 1

        return out_frames

    # Finish by returning an empty list to the next module
    return []


# John's hack on the dfmux version since we have a
# few kinds of timepoint frames now
class fixed_length_scans(object):
    def __init__(self, N=1000):
        self._N = N
        self._count = 0

    def __call__(self, frame):
        ret = []
        if (frame.type == core.G3FrameType.Timepoint and
           'DfMux' in frame.keys()):
            if self._count % self._N == 0:
                ret.append(core.G3Frame(core.G3FrameType.Scan))
            self._count += 1
        ret.append(frame)
        return ret


def check_contents(frame):
    if frame.type == core.G3FrameType.Scan:
        print("Length of encoder counts timestream for this scan: %d"
              % (len(frame['chwp_encoder_count'])))
    return


# Instantiate the G3 pipeline
pipe = core.G3Pipeline()
# Generate frames of dummy data
pipe.Add(dummy_data, frames_to_generate=10000)
# Generate scans of fixed length
print('Creating scans with a fixed length of 1000')
pipe.Add(fixed_length_scans, N=1000)
# Remove the dfmux frames
pipe.Add(lambda frame: not (
    frame.type == core.G3FrameType.Timepoint and 'DfMux' in frame.keys()))
# Test the CHWP collator
pipe.Add(chwp.CHWPCollator)
# Check pipeline CHWP contents
pipe.Add(check_contents)
# Save the dummy data to disk
pipe.Add(core.G3Writer, filename='scanified.g3')

# Run the constructed pipeline
pipe.Run()
