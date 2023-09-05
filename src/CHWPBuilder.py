import numpy as np
import time
import struct
from spt3g import core  # pylint: disable=import-error


@core.indexmod
class CHWPBuilder(object):
    """
    Every time a frame associated with a bolometer sample is reached,
    empty the queue that the WHWPCollector module is populating with
    CHWP encoder data. Put these nicely into the pipeline
    """
    def __init__(self, collector):
        # Store inherited attributes
        self._collector = collector
        # Multiprocessing queue
        self.queue = self._collector.queue
        # Data string to be stored into the collector queue
        self._data = ''

        # Read the data little endian
        self._endian = '<'
        # Define an unsigned long int
        self._unsigned_long_int_size = 4  # bytes
        self._unsigned_long_int_str = 'L'
        # Number of bits in each counter overflow
        self._num_overflow_bits = self._unsigned_long_int_size * 8  # bits
        # All heaaders are unsigned long ints
        self._header_size = self._unsigned_long_int_size

        # Logger messages
        # CHWP Timeout
        self._timeout_msg = "CHWP DAQ Timeout"
        self._error_msg = "CHWP DAQ Error"

        # Define an encoder packet
        self._define_encoder_packet()
        # Define an IRIG packet
        self._define_irig_packet()
        # Define an error packet
        self._define_error_packet()
        # Define a timeout packet
        self._define_timeout_packet()

    def __call__(self, frame):
        # Only act when a timepoint frame from bolo samples is reached
        check_frame = (frame.type == core.G3FrameType.Timepoint and
                       'DfMux' in frame.keys())
        if not check_frame:
            return None
        else:
            # Frames to pass to the next module,
            # including the current bolometer sample frame
            return_frames = [frame]

            # Empty the queue and parse its contents appropriately
            approx_size = self.queue.qsize()
            for _ in range(approx_size):
                # Block execution until there is something in
                # the queue to retrieve
                # timeout=None means the get() command will try indefinitely
                self._data = self.queue.get(block=True, timeout=None)

                # Once data is extracted from the queue, parse its contents
                # and loop until data is empty
                data_len = len(self._data)
                parse_index = 0
                while parse_index < data_len:
                    # Extract header
                    header = self._data[
                        parse_index : parse_index + self._header_size]
                    header = struct.unpack(("%s%s" % (
                        self._endian,
                        self._unsigned_long_int_str)),
                        header)[0]
                    # Check for Encoder packet
                    if header == self._encoder_header:
                        return_frames.append(
                            self._process_encoder_packet(parse_index))
                        parse_index += self._encoder_packet_size
                    # Check for IRIG packet
                    elif header == self._irig_header:
                        return_frames.append(
                            self._process_irig_packet(parse_index))
                        parse_index += self._irig_packet_size
                    # Check for Timeout packet
                    elif header == self._timeout_header:
                        self._process_timeout_packet(parse_index)
                        parse_index += self._timeout_packet_size
                    # Check for Error packet
                    elif header == self._error_header:
                        self._process_error_packet(parse_index)
                        parse_index += self._error_packet_size
                    else:
                        raise RuntimeError(
                            ("%s: Bad header: %s" % (
                                self._error_msg, str(header))))
                # Reset data string
                self._data = ''
            return return_frames

    def _define_encoder_packet(self):
        # EncoderInfo packet data format
        # [header] + [quadrature] + [120 datapoint information] * 3

        # Encoder header
        # -- (unsigned long int) header
        self._encoder_header = 0x1EAF
        self._encoder_header_units = 1
        self._encoder_header_size = (
            self._encoder_header_units * self._unsigned_long_int_size)
        self._encoder_header_str = (
            self._encoder_header_units * self._unsigned_long_int_str)

        # Encoder quadrature
        # -- (unsigned long int) quad
        self._encoder_quad_units = 1
        self._encoder_quad_size = (
            self._encoder_quad_units * self._unsigned_long_int_size)
        self._encoder_quad_str = (
            self._encoder_quad_units * self._unsigned_long_int_str)

        # Encoder data
        # -- (unsigned long int) clock[counter_info_length]
        # -- (unsigned long int) clock_overflow[counter_info_length]
        # -- (unsigned long int) count[counter_info_length]
        self._encoder_data_length = 120
        self._encoder_data_units = 3 * self._encoder_data_length
        self._encoder_data_size = (
            self._encoder_data_units * self._unsigned_long_int_size)
        self._encoder_data_str = (
            self._encoder_data_units * self._unsigned_long_int_str)

        # Encoder packet size
        self._encoder_packet_size = (
            self._encoder_header_size + self._encoder_quad_size +
            self._encoder_data_size)

        # String to unpack encoder data
        self._encoder_unpack_str = ( "%s%s%s%s" % (
            self._endian,
            self._encoder_header_str, self._encoder_quad_str,
            self._encoder_data_str))
        return
    
    def _define_irig_packet(self):
        # IRIG packet data format:
        # [header] + [rising_edge_time] + [init_overflow] + [10 info] +
        # [10 re_count] + [10 re_count_overflow]

        # Header
        # -- (unsigned long int) header
        self._irig_header = 0xCAFE
        self._irig_header_units = 1
        self._irig_header_size = (
            self._irig_header_units * self._unsigned_long_int_size)
        self._irig_header_str = (
            self._irig_header_units * self._unsigned_long_int_str)

        # IRIG clock used to define time
        # -- (unsigned long int) clock
        self._irig_clock_units = 1
        self._irig_clock_size = (
            self._irig_clock_units * self._unsigned_long_int_size)
        self._irig_clock_str = (
            self._irig_clock_units * self._unsigned_long_int_str)

        # IRIG clock overflow
        # -- (unsigned long int) clock_overflow
        self._irig_overflow_units = 1
        self._irig_overflow_size = (
            self._irig_overflow_units * self._unsigned_long_int_size)
        self._irig_overflow_str = (
            self._irig_overflow_units * self._unsigned_long_int_str)

        # IRIG data
        # -- (unsigned long int) info[10]
        # -- (unsigned long int) synch[10]
        # -- (unsigned long int) synch_overflow[10]
        self._irig_data_length = 10
        self._irig_data_units = 3 * self._irig_data_length
        self._irig_data_size = (
            self._irig_data_units * self._unsigned_long_int_size)
        self._irig_data_str = (
            self._irig_data_units * self._unsigned_long_int_str)

        # Total IRIG packet size
        self._irig_packet_size = (
            self._irig_header_size + self._irig_clock_size +
            self._irig_overflow_size + self._irig_data_size)

        # String to unpack IRIG data
        self._irig_unpack_str = ("%s%s%s%s%s" % (
            self._endian,
            self._irig_header_str, self._irig_clock_str,
            self._irig_overflow_str, self._irig_data_str))

        return

    def _define_timeout_packet(self):
        # Timeout packet data format:
        # [header] + [timeout type]

        # Encoder timeout type
        self._encoder_timeout_type = 1
        # IRIG timeout type
        self._irig_timeout_type = 2

        # Header
        # -- (unsigned long int) header = 0x1234
        self._timeout_header = 0x1234
        self._timeout_header_units = 1
        self._timeout_header_size = (
            self._timeout_header_units * self._unsigned_long_int_size)
        self._timeout_header_str = (
            self._timeout_header_units * self._unsigned_long_int_str)

        # Timeout type
        # -- (unsigned long int) type
        self._timeout_type_units = 1
        self._timeout_type_size = (
            self._timeout_type_units * self._unsigned_long_int_size)
        self._timeout_type_str = (
            self._timeout_type_units * self._unsigned_long_int_str)

        # Total timeout packet size
        self._timeout_packet_size = (
            self._timeout_header_size + self._timeout_type_size)

        # String to unpack timeout data
        self._timeout_unpack_str = ("%s%s%s" % (
            self._endian,
            self._timeout_header_str, self._timeout_type_str))
        
        return

    def _define_error_packet(self):
        # Error packet data format:
        # [header] + [error code]

        # Error codes
        # No error
        self._error_code_none = 0x0000
        # IRIG desync
        self._error_code_desync = 0x0001

        # Header
        # -- (unsigned long int) header
        self._error_header = 0xE12A
        self._error_header_units = 1
        self._error_header_size = (
            self._error_header_units * self._unsigned_long_int_size)
        self._error_header_str = (
            self._error_header_units * self._unsigned_long_int_str)

        # Error code
        # -- (unsigned long int) code
        self._error_code_units = 1
        self._error_code_size = (
            self._error_code_units * self._unsigned_long_int_size)
        self._error_code_str = (
            self._error_code_units * self._unsigned_long_int_str)

        # Total error packet size
        self._error_packet_size = (
            self._error_header_size + self._error_code_size)

        # String to unpack error data
        self._error_unpack_str = ("%s%s%s" % (
            self._endian,
            self._error_header_str, self._error_code_str))
        
        return

    def _process_encoder_packet(self, parse_index):
        # Unpack the encoder data
        start_ind = parse_index
        end_ind = start_ind + self._encoder_packet_size
        unpacked_data = np.array(struct.unpack(
            self._encoder_unpack_str,
            self._data[start_ind:end_ind]))
        # Parse the counter packets
        # Extract the header
        ind1 = 0
        ind2 = ind1 + self._encoder_header_units
        header = unpacked_data[ind1:ind2][0]
        if header != self._encoder_header:
            raise RuntimeError(
                "%s: Encoder header error: 0x%04X" % (
                    self._error_msg, header))
        # Extract the quadrature data
        ind1 = ind2
        ind2 = ind1 + self._encoder_quad_units
        quad_data = unpacked_data[ind1:ind2][0]
        # Extract the encoder clock data
        ind1 = ind2
        ind2 = ind1 + self._encoder_data_length
        clock_data = unpacked_data[ind1:ind2]
        # Extract the overflow data
        ind1 = ind2
        ind2 = ind1 + self._encoder_data_length
        ovflw_data = unpacked_data[ind1:ind2]
        # Extract the encoder count data
        ind1 = ind2
        ind2 = ind1 + self._encoder_data_length
        count_data = unpacked_data[ind1:ind2]
        # Save the quadrature data as a G3UInt
        quad = core.G3UInt(int(quad_data))
        # Save the clock data as a G3VectorUInt
        clk = clock_data + (ovflw_data << self._num_overflow_bits)
        clk = core.G3VectorUInt(clk)
        # Save the count data as a G3VectorIntc
        cnts = core.G3VectorUInt(count_data)

        # Create and return a G3 timepoint frame with clock and encoder data
        encoder_frame = core.G3Frame(core.G3FrameType.Timepoint)
        encoder_frame['chwp_encoder_quad'] = quad
        encoder_frame['chwp_encoder_clock'] = clk
        encoder_frame['chwp_encoder_count'] = cnts
        return encoder_frame

    def _process_irig_packet(self, parse_index):
        # Unpack the IRIG data
        start_ind = parse_index
        end_ind = start_ind + self._irig_packet_size
        unpacked_data = np.array(struct.unpack(
            self._irig_unpack_str, self._data[start_ind:end_ind]))
        # Unpack the IRIG header
        ind1 = 0
        ind2 = ind1 + self._irig_header_units
        header = unpacked_data[ind1:ind2][0]
        if header != self._irig_header:
            raise RuntimeError(
                "%s: IRIG header error: 0x%x" % (
                    self._error_msg, header))
        # Unpack the IRIG clock
        ind1 = ind2
        ind2 = ind1 + self._irig_clock_units
        clock = unpacked_data[ind1:ind2][0]
        # Unpack the IRIG clock overflows
        ind1 = ind2
        ind2 = ind1 + self._irig_overflow_units
        overflow = unpacked_data[ind1:ind2][0]
        # Adjust the IRIG clock for overflows
        clock_adjusted = clock + (overflow << self._num_overflow_bits)
        # Unpack the IRIG info
        ind1 = ind2
        ind2 = ind1 + self._irig_data_length
        info = unpacked_data[ind1:ind2]
        # Unpack the IRIG synch pulses
        ind1 = ind2
        ind2 = ind1 + self._irig_data_length
        synch = unpacked_data[ind1:ind2]
        # Unpack the IRIG synch pulse overflows
        ind1 = ind2
        ind2 = ind1 + self._irig_data_length
        synch_overflow = unpacked_data[ind1:ind2]
        synch_adjusted = (synch + (synch_overflow << self._num_overflow_bits))

        # Convert raw IRIG bits to a meaningful time
        year, yday, hour, mins, secs = self._irig_time_conversion(info)
        # Obtain the time using the G3 object
        irig_time = core.G3Time(
            y=int(year), d=int(yday), h=int(hour), m=int(mins),
            s=int(secs), ss=0)  # no subseconds

        # Store the time as a G3Int
        time_s = core.G3Double(float(irig_time.time) / float(core.G3Units.seconds))
        # time_s = core.G3UInt(irig_time.time)
        # Store clock value as a G3UInt
        clock_adjusted = core.G3UInt(int(clock_adjusted))
        # Store the synchronization pulse clock values as a
        # synch_adjusted = core.G3VectorUInt(np.array(synch_adjusted))

        # Create and return a G3 timepoint frame with clock and irig data
        irig_frame = core.G3Frame(core.G3FrameType.Timepoint)
        irig_frame['chwp_irig_time'] = time_s
        irig_frame['chwp_irig_clock'] = clock_adjusted
        # irig_frame['chwp_irig_synch'] = synch_adjusted
        return irig_frame

    def _process_timeout_packet(self, parse_index):
        # Unpack the timeout data
        start_ind = parse_index
        end_ind = start_ind + self._timeout_packet_size
        unpacked_data = np.array(struct.unpack(
            self._timeout_unpack_str, self._data[start_ind:end_ind]))
        # Unpack the Timeout header
        ind1 = 0
        ind2 = ind1 + self._timeout_header_units
        header = unpacked_data[ind1:ind2]
        if header != self._timeout_header:
            raise RuntimeError("Timeout header error: 0x%x" % (header))
        # Unpack the Timeout type
        ind1 = ind2
        ind2 = ind1 + self._timeout_type_size
        timeout_type = unpacked_data[ind1:ind2]
        if timeout_type == self._encoder_timeout_type:
            print("Timeout: no encoder data detected")
        elif timeout_type == self._irig_timeout_type:
            print("Timeout: no IRIG data detected")
        else:
            print("Timeout: unknown type '0x%X'" % (timeout_type))
        return

    def _process_error_packet(self, parse_index):
        # Unpack the Error data
        start_ind = parse_index
        end_ind = start_ind + self._error_packet_size
        unpacked_data = np.array(struct.unpack(
            self._error_unpack_str, self._data[start_ind:end_ind]))
        # Unpack the Error header
        ind1 = 0
        ind2 = ind1 + self._error_header_units
        header = unpacked_data[ind1:ind2]
        if header != self._error_header:
            raise RuntimeError(
                "%s: Error header error: 0x%x" % (
                    self._error_msg, int(header)))
        # Unpack the Error code
        ind1 = ind2
        ind2 = ind1 + self._error_code_units
        error_code = unpacked_data[ind1:ind2]
        if error_code == self._error_code_none:
            print("%s: no error" % (self._error_msg))
        elif error_code == self._error_code_desync:
            print("%s: IRIG desync" % (self._error_msg))
        else:
            print("%s: unknown error (code '0x%X')" % (
                self._error_msg, hex(error_code)))
        return

    def _irig_time_conversion(self, info):
        # Correct for the bit shift in the seconds field
        info[0] = info[0] >> 1
        # Convert to binary to decimal
        info = self._binary_to_decimal(info)
        year = info[5]
        yday = info[3] + (info[4] & 0x3) * 100
        hour = info[2]
        mins = info[1]
        secs = info[0]
        # If the IRIG generator doesn't send out a year,
        # get it locally
        if year == 0:
            systime = time.gmtime()
            year = systime.tm_year - 2000
            # Handle new years
            if systime.tm_yday <= 1 and yday >= 364:
                year -= 1
            elif systime.tm_yday >= 364 and yday <= 1:
                year += 1
        return year, yday, hour, mins, secs

    def _binary_to_decimal(self, inp_bytes):
        """ Convert each block of 10 bits in the IRIG to decimal """
        # Should be 10 blocks for 100 bits total
        outbytes = [0, ] * len(inp_bytes)
        for i in range(len(inp_bytes)):
            b = inp_bytes[i]
            lowb = b & 0xf  # low bits are 2, 4, 8
            highb = b >> 5  # high bits are 10, 20, 40, etc
            outbytes[i] = lowb + highb * 10
        return outbytes
