*** PRU assignment ***
PRU0 = CHWP encoder (signal ~ 1 kHz)
PRU1 = IRIG (signal ~200  Hz)

*** PRU memory map ***
(according to https://elinux.org/images/d/da/Am335xPruReferenceGuide.pdf)
Shared memory is addressed starting at 0x10000
Size is 12 kB = 0x3000
The PRU has one byte per address, so the total available shared memory
is between 0x10000 and 0x1300

*** CHWP memory map ***
The CHWP  uses the following memory assignments
-- shared addresses between the IRIG and encoder
0x10000 = unsigned short int on, which tells whether the PRU is collecting data
0x10002 = unsigned short int overflow, which counts the number of counter overflows
-- encoder-specific addresses
0x10010 = unsigned short int counter identifier
0x10012 = unsigned long int counter values
-- IRIG-specific addresses
0x12000 = unsigned short int IRIG identifier
0x12002 = unsigned long int IRIG values
-- Error-specific addresses
0x12300 = unsigned short int Error identifier
0x12302 = unsigned long int Error values

*** CHWP objects written to memory ***
-- CHWP encoder packet (3610 = 0xe1a bytes)
  - Identifier (unsigned short int = 2 bytes)
  - Quadrature value (unsigned short int = 8 bytes)
  - 150 clock values (unsigned long int = 8 bytes)
  - 150 overflow values (unsigned long int = 8 bytes)
  - 150 counter values (unsigned long int = 8 bytes)
-- IRIG packet (258 = 0x102 bytes)
  - Identifier (unsigned short int = 2 bytes)
  - Rising edge time (unsigned long int = 8 bytes)
  - Rising edge time overflows (unsigned long int = 8 bytes)
  - 10 IRIG info values (unsigned long int = 8 bytes)
  - 10 IRIG synch pulses (unsigned long int = 8 bytes)
  - 10 IRIG synch overflows (unsigned long int = 8 bytes)
-- Error packet (4 = 0x4 bytes)
  - Identifier (unsigned short int = 2 bytes)
  - Error state (unsigned short int = 2 bytes)
