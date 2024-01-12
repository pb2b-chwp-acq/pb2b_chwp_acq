import multiprocessing
import socket
import errno
import ctypes
import signal
import time
import select

class CHWPCollector(object):
    """
    In a separate process, listens for UDP packets from the CHWP MCU and
    populates a queue that the CHWPBuilder object will read from
    """
    def __init__(self, mcu_port=8080):
        """
        set up the socket connection (non-blocking)
        note that sockets are automatically closed
        when they are garbage collected
        Arguments:
            mcu_port - port to use
        """
        # Number of bits to read in chunks from the input buffer
        #self._read_chunk_size = 16384
        # self._read_chunk_size = 1048576
        self._read_chunk_size = 2**20
        # Use IPv4 internet protocol, datagram type
        self._s = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        # Allow reuse of addresses in the bind call below
        self._s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Make the connection non-blocking. This is critical
        self._s.setblocking(False)
        # Connect to the microcontroller unit
        self._s.bind(("", mcu_port))
        # Data colleciton object
        self._data = b''
        # Timeout seconds for waiting for data
        self._timeout_sec = 1

        # Used to signal to stop recording
        self._should_stop = multiprocessing.Value(ctypes.c_bool, False)
        self._stopped = multiprocessing.Value(ctypes.c_bool, False)

        # Start the relaying process
        self.queue = multiprocessing.Queue()
        self._process = multiprocessing.Process(
            target=self.relay_HWP_data,
            args=(self._should_stop, self._stopped))
        self._process.start()

        # Deal with SIGINT interruptions (nominally how we stop the DAQ) nicely
        signal.signal(signal.SIGINT, self.sigint_handler_parent)

    def sigint_handler_parent(self, signal, frame):
        """
        If SIGINT is sent to the constructor, nicely terminate the
        child listening process without corrupting the queue
        """
        self.stop()

    def sigint_handler_child(self, signal, frame):
        """ Ignore SIGINT in relay_HWP_data() process """
        pass

    def relay_HWP_data(self, should_stop, stopped):
        # In case the SIGINT signal is sent to this process,
        # ignore it to avoid corrupting things
        signal.signal(signal.SIGINT, self.sigint_handler_child)

        while True:
            # check to see if we need to stop recording
            if should_stop.value:
                break

            # check the socket, continuing if there's nothing there yet
            try:
                ready = select.select([self._s], [], [], self._timeout_sec)
                if ready[0]:
                    self._data += self._s.recv(self._read_chunk_size)
            except socket.error as err:
                # In the case of a real problem, re-raise the exception
                if err.errno != errno.EAGAIN:
                    raise
                # Otherwise if it's just empty, continue
                else:
                    pass

            # Push the data to the queue
            if len(self._data) > 0:
                self.queue.put(obj=self._data, block=True, timeout=None)
                self._data = b''

        # let the parent process know that it's now
        # safe to terminate this child process
        with stopped.get_lock():
            stopped.value = True

    def stop(self):
        """
        Stops the listening process that's
        grabbing packets from the CHWP MCU
        """
        # Signal to the child process that it should finish up and end
        with self._should_stop.get_lock():
            self._should_stop.value = True

        # Wait until the process is in a safe state, then terminate it
        while not self._stopped.value:
            time.sleep(0.001)
        self._process.terminate()

        # Block until the child process really is dead
        self._process.join()
