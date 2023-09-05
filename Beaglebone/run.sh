#!/bin/sh
if [ $$ != `pgrep -fo $0`  ]; then
	    echo "encoder is already running. exit."
			    exit 1
fi
./Beaglebone_Encoder_DAQ Encoder1.bin Encoder2.bin IRIG1.bin IRIG2.bin
