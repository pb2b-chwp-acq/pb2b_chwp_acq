from spt3g import core  # pylint: disable=import-error
from spt3g import chwp # pylint: disable=import-error
import time
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--in_file', type=str, help='Input data file')
args = parser.parse_args()

# Output file name
in_fname_path, in_fname = os.path.split(args.in_file)
in_fname_head, fname_ext = in_fname.split('.')
out_fname_head = "%s_%s" % (in_fname_head, "analyzed")
out_fname = '.'.join([out_fname_head, fname_ext])
out_fname_path = os.path.join(in_fname_path, out_fname)

# ***** MAIN *****
# Connect to G3 pipeline
pipe = core.G3Pipeline()
# Add module to generate dummy MUX data
pipe.Add(core.G3Reader, filename=args.in_file)
# Process the data
pipe.Add(chwp.CHWPDataProcess)
# Analyze the data
pipe.Add(chwp.CHWPDataAnalyze)
# Write output data to a G3 file
pipe.Add(core.G3Writer, filename=out_fname_path)

# Run the pipeline
pipe.Run(profile=True)
