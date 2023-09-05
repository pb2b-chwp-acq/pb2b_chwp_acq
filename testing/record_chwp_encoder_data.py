from spt3g import core  # pylint: disable=import-error
from spt3g import chwp # pylint: disable=import-error
import time
import argparse


# generate bolometer timepoint frames artificially
num_frames = 0


def timed_read(frame, start_time, time_len, samp_rate=10):
    global num_frames
    while time.time() < start_time + time_len:
        if num_frames % (10 * samp_rate) == 0:
            print("%d secs remaining..."
                  % (int(time_len-(num_frames / samp_rate))))
        frame = core.G3Frame(core.G3FrameType.Timepoint)
        time.sleep(1. / float(samp_rate))
        frame['EventHeader'] = core.G3Time.Now()
        # Store zero for DfMux data
        frame['DfMux'] = 0
        num_frames += 1
        return [frame]
    # Return an empty list once we're done so the pipeline stops
    return []


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--time', type=int, help='Number of seconds to record for',
    default=10)
parser.add_argument(
    '--out_file', type=str, help='Save data to output file with this path',
    default=None)
args = parser.parse_args()

# ***** MAIN *****
# Connect to G3 pipeline
pipe = core.G3Pipeline()
# Add module to generate dummy MUX data
pipe.Add(timed_read, start_time=time.time(), time_len=args.time)
# Start the collection of packets from the CHWP MCU
chwp_collector = chwp.CHWPCollector()
# Insert CHWP data into dummy MUX frames in the pipeline
pipe.Add(chwp.CHWPBuilder, collector=chwp_collector)
# Send CHWP data out for slow DAQ publishing
pipe.Add(chwp.CHWPSlowDAQTee)
# Write data to a G3 file
if args.out_file is not None:
    pipe.Add(core.G3Writer, filename=args.out_file)

# Run the pipeline
pipe.Run(profile=True)
# End the collector nicely (separate process)
chwp_collector.stop()
