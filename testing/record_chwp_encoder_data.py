from spt3g import core  # pylint: disable=import-error
from spt3g import chwp # pylint: disable=import-error
import time
import os
import argparse
from datetime import datetime


# generate bolometer timepoint frames artificially
num_frames = 0

db_path = '/data2/home/polarbear/bbixler/pb2b_chwp_acq/data'

def timed_read(frame, start_time, time_len, samp_rate=10):
    global num_frames

    while datetime.utcnow().timestamp() < start_time + time_len:
        cur_day = datetime.utcnow().strftime("%Y%m%d")
        if not os.path.isdir(os.path.join(db_path, cur_day)):
            os.mkdir(os.path.join(db_path, cur_day))

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


def add_datetime(frame):
    #if (frame.type == core.G3FrameType.Timepoint and
    #        ('chwp_encoder_clock' in frame.keys() or
    #        'chwp_irig_clock' in frame.keys())):
    now = datetime.utcnow()
    frame['frame_day'] = now.strftime("%Y%m%d")
    frame['frame_time'] = now.strftime("%Y%m%d%H%M%S")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--time', type=int, help='Number of seconds to record for',
    default=10)
parser.add_argument(
    '--port', type=int, help='Port to collect packets from',
    default=8080)
args = parser.parse_args()

# ***** MAIN *****
# Connect to G3 pipeline
pipe = core.G3Pipeline()
# Add module to generate dummy MUX data
pipe.Add(timed_read, start_time=datetime.utcnow().timestamp(), time_len=args.time)
# Start the collection of packets from the CHWP MCU
chwp_collector = chwp.CHWPCollector(mcu_port=args.port)
# Insert CHWP data into dummy MUX frames in the pipeline
pipe.Add(chwp.CHWPBuilder, collector=chwp_collector)
# Send CHWP data out for slow DAQ publishing
#pipe.Add(chwp.CHWPSlowDAQTee)

pipe.Add(add_datetime)
# Write data to a G3 file
pipe.Add(core.G3MultiFileWriter, filename=lambda fr, _: os.path.join(db_path,fr['frame_day'],fr['frame_time']+'.g3'),
         size_limit=2**25)

# Run the pipeline
pipe.Run(profile=True)
# End the collector nicely (separate process)
chwp_collector.stop()
