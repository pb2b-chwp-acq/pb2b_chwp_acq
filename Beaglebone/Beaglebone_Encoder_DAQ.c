// Used to load code for detecting Encoder and IRIG signals onto PRUs
// Encoder code is loaded onto PRU1 and IRIG code onto PRU0
//
// Usage:
// $ ./Beaglebone_Encoder_DAQ Encoder1.bin Encoder2.bin IRIG1.bin IRIG2.bin
//
// Compile with:
// gcc -o Beaglebone_Encoder_DAQ Beaglebone_Encoder_DAQ.c -lprussdrv

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// PRU Subsystem Driver
// (installed from https://github.com/beagleboard/am335x_pru_package)
#include <prussdrv.h>
// PRU Subsystem Interupt Controller Mapping
// (installed from https://github.com/beagleboard/am335x_pru_package)
#include <pruss_intc_mapping.h>
#include <string.h>
// The rest of these libraries are for UDP service
#include <sys/types.h> 
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

// Port used for the UDP connection
#define PORT 8080

// Below variables are defined in pruss_intc_mapping and prussdrv,
// they are mapping interrupts from PRUs to ARM processor
#define PRUSS_INTC_CUSTOM {   \
    { PRU0_PRU1_INTERRUPT, PRU1_PRU0_INTERRUPT, PRU0_ARM_INTERRUPT, \
      PRU1_ARM_INTERRUPT, ARM_PRU0_INTERRUPT, ARM_PRU1_INTERRUPT,  24, (char)-1  },  \
    { {PRU0_PRU1_INTERRUPT,CHANNEL1}, {PRU1_PRU0_INTERRUPT, CHANNEL0}, \
      {PRU0_ARM_INTERRUPT,CHANNEL2}, {PRU1_ARM_INTERRUPT, CHANNEL3}, \
      {ARM_PRU0_INTERRUPT, CHANNEL0}, {ARM_PRU1_INTERRUPT, CHANNEL1}, \
      {24, CHANNEL3}, {-1,-1}},  \
    { {CHANNEL0,PRU0}, {CHANNEL1, PRU1}, {CHANNEL2, PRU_EVTOUT0}, \
      {CHANNEL3, PRU_EVTOUT1}, {-1,-1} },  \
    (PRU0_HOSTEN_MASK | PRU1_HOSTEN_MASK | PRU_EVTOUT0_HOSTEN_MASK | PRU_EVTOUT1_HOSTEN_MASK) \
}

// Encoder packet size
#define ENCODER_COUNTER_SIZE 120

// Number of packets to send at once
// ~0.3 Hz in normal operation
#define ENCODER_PACKETS_TO_SEND 1
#define IRIG_PACKETS_TO_SEND 1
#define ERROR_PACKETS_TO_SEND 1
#define TIMEOUT_PACKETS_TO_SEND 1

// Definining the address offsets from the start of shared memory
// for the structures and variables used by PRUs
#define ON_OFFSET 0x0000
#define OVERFLOW_OFFSET 0x0008
#define ENCODER_READY_OFFSET 0x0010
#define ENCODER_OFFSET 0x0018
#define IRIG_READY_OFFSET 0x1850
#define IRIG_OFFSET 0x1858
#define ERROR_READY_OFFSET 0x2000
#define ERROR_OFFSET 0x2008
// Size of readout chunks in bytes
#define READOUT_BYTES 4

// Timeout values for each packet type [s]
#define ENCODER_TIMEOUT 10
#define IRIG_TIMEOUT 10

// Timeout flags
#define ENCODER_TIMEOUT_FLAG 1
#define IRIG_TIMEOUT_FLAG 2

// Function which returns pointer to shared memory
// Indexes over memory in 4-byte increments
volatile int32_t* init_prumem()
{
    volatile int32_t* p;
    prussdrv_map_prumem(PRUSS0_SHARED_DATARAM, (void**)&p);
    return p;
}

// Structure to store clock count of edges and
// the number of times the counter has overflowed
struct EncoderInfo {
    unsigned long int header;
    unsigned long int quad;
    unsigned long int clock[ENCODER_COUNTER_SIZE];
    unsigned long int clock_overflow[ENCODER_COUNTER_SIZE];
    unsigned long int count[ENCODER_COUNTER_SIZE];
};

// IRIG packet
struct IrigInfo{
    unsigned long int header;
    unsigned long int clock;
    unsigned long int clock_overflow;
    unsigned long int info[10];
    unsigned long int synch[10];
    unsigned long int synch_overflow[10];
};
// Error packets sent when IRIG isn't synced
struct ErrorInfo{
    unsigned long int header;
    unsigned long int err_code;
};

// Packet to send in the event of a data collection timeout
struct TimeoutInfo {
    unsigned long int header;
    unsigned long int type;
};

//Creating pointers to all shared variables and data structures in shared memory
// Pointer to variable to let the ARM know that the PRUs are still executing code
volatile unsigned long int* on;
// Pointer to number of clock overflows
volatile unsigned long int* clock_overflow;
// Pointer to flag signifying encoder packets are ready to be collected
volatile unsigned long int* encoder_ready;
// Pointer to data structure for encoder/counter packets
volatile struct EncoderInfo* encoder_packets;
// Pointer to flag signifying irig packets are ready to be collected
volatile unsigned long int* irig_ready;
// Pointer to data structure for IRIG packets
volatile struct IrigInfo* irig_packets;
// Pointer to variable to identify that an error packet is ready to be writtento UDP
volatile unsigned long int* error_ready;
// Pointer to data structure for error packets
volatile struct ErrorInfo* error_packets;

// Arrays for storing packets to be sent over UDP
volatile struct EncoderInfo encoder_to_send[ENCODER_PACKETS_TO_SEND];
volatile struct IrigInfo irig_to_send[IRIG_PACKETS_TO_SEND];
volatile struct ErrorInfo error_to_send[ERROR_PACKETS_TO_SEND];
volatile struct TimeoutInfo timeout_packet[TIMEOUT_PACKETS_TO_SEND];

// ***** LOCAL VARIABLES *****
// For swapping between the two stored packets
unsigned long int offset;
// For indexing over Encoder, IRIG, and Error packets to send out
unsigned long int encd_ind, irig_ind, err_ind;
// Monitor the time since the packet was sent
clock_t curr_time, encd_time, irig_time;
// Creates socket to write UDP packets with
int sockfd;
struct sockaddr_in servaddr;
int tos_write = 0b10100100;
int tos_read;
int tos_read_len = sizeof(tos_read);

// **************************
// ********** MAIN **********
// **************************

int main(int argc, char **argv) {
    // *** Configure the PRUs ***
    // Run a bash file to configure the input pins
    system("./pinconfig");
    
    // Check command-line arguments
    if (argc != 5) {
        printf("Usage: %s Beaglebone_Encoder_DAQ Encoder1.bin \
                Encoder2.bin IRIG1.bin IRIG2.bin\n", argv[0]);
        return 1;
    }

    // Initialize PRU subsystem driver
    prussdrv_init();
    // Allow the use of interrupt: PRU_EVTOUT_1
    // Used to notify the ARM that the PRUs have finished
    if (prussdrv_open(PRU_EVTOUT_1) == -1) {
        printf("prussdrv_open() failed\n");
        return 1;
    }
    // Functions to map and initialize the interrupts defined above
    tpruss_intc_initdata pruss_intc_initdata = PRUSS_INTC_CUSTOM;
    prussdrv_pruintc_init(&pruss_intc_initdata);

    // Set pointer addresses in shared memory
    on = (volatile unsigned long int*) (init_prumem() + (ON_OFFSET / READOUT_BYTES));
    // Pointer to number of clock overflows
    clock_overflow = (volatile unsigned long int*) (init_prumem() + (OVERFLOW_OFFSET / READOUT_BYTES));
    // Pointer to flag signifying encoder packets are ready to be collected
    encoder_ready = (volatile unsigned long int*) (init_prumem() + (ENCODER_READY_OFFSET / READOUT_BYTES));
    // Pointer to data structure for encoder/counter packets
    encoder_packets = (volatile struct EncoderInfo*) (init_prumem() + (ENCODER_OFFSET / READOUT_BYTES));
    // Pointer to flag signifying irig packets are ready to be collected
    irig_ready = (volatile unsigned long int *) (init_prumem() + (IRIG_READY_OFFSET / READOUT_BYTES));
    // Pointer to data structure for IRIG packets
    irig_packets = (volatile struct IrigInfo *) (init_prumem() + (IRIG_OFFSET / READOUT_BYTES));
    // Pointer to variable to identify that an error packet is ready to be writtento UDP
    error_ready = (volatile unsigned long int *) (init_prumem() + (ERROR_READY_OFFSET / READOUT_BYTES));
    // Pointer to data structure for error packets
    error_packets = (volatile struct ErrorInfo *) (init_prumem() + (ERROR_OFFSET / READOUT_BYTES));

    // *** Configure memory allocation ***
    // Sets memory to be used by data structures to 0
    memset((struct EncoderInfo *) &encoder_packets[0], 0, sizeof(*encoder_packets));
    memset((struct EncoderInfo *) &encoder_packets[1], 0, sizeof(*encoder_packets));   
    memset((struct IrigInfo *) &irig_packets[0], 0, sizeof(*irig_packets));
    memset((struct IrigInfo *) &irig_packets[1], 0, sizeof(*irig_packets));
    memset((struct ErrorInfo *) &error_packets[0], 0, sizeof(*error_packets));
    // Reset ready flags
    *encoder_ready = 0;
    *irig_ready = 0;
    *error_ready = 0; 
 
    // Load code to PRU1
    printf("Initializing PRU1\n");
    if (argc > 2) {
        if (prussdrv_load_datafile(1, argv[2]) < 0) {
            fprintf(stderr, "Error loading %s\n", argv[2]);
            exit(-1);
        }
    }
    if (prussdrv_exec_program(1, argv[1]) < 0) {
        fprintf(stderr, "Error loading %s\n", argv[1]);
       exit(-1);
    }
    // Load code to PRU0
    printf("Initializing PRU0\n");
    if (argc == 5) {
      if (prussdrv_load_datafile(0, argv[4]) < 0) {
          fprintf(stderr, "Error loading %s\n", argv[4]);
          exit(-1);
      }
    }
    if (prussdrv_exec_program(0, argv[3]) < 0) {
        fprintf(stderr, "Error loading %s\n", argv[3]);
        exit(-1);
    }

    // *** Set up UDP connection ***
    if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(PORT);
    inet_pton(AF_INET, "192.168.2.54", &(servaddr.sin_addr.s_addr));
    setsockopt(sockfd, IPPROTO_IP, IP_TOS, &tos_write, sizeof(tos_write));
    getsockopt(sockfd, IPPROTO_IP, IP_TOS, &tos_read, &tos_read_len);
    printf("IP UDP TOS byte set to 0x%X\n", tos_read);
    printf("   Precedence = 0x%X\n", (tos_read >> 5) & 0x7);
    printf("   TOS = 0x%X\n", (tos_read >> 1) & 0xF);
   
    // Set the timeout header
    timeout_packet->header = 0x1234;
    // Start stashing data into data objects to send over UDP
    encd_ind = 0;
    irig_ind = 0;
    curr_time = clock();
    // Continuously loops, looking for data, while PRUs are executing
    printf("Initializing DAQ\n");
    while(*on != 1) {
        // Record the current time
        curr_time = clock();
        // Gather encoder data
        if(*encoder_ready != 0) {
            offset = *encoder_ready - 1;
            encoder_to_send[encd_ind] = encoder_packets[offset];
            encd_ind += 1;
            *encoder_ready = 0;
            // Update the last time the encoder data was recorded
            encd_time = curr_time;
        }
        if(*irig_ready != 0) {
            offset = *irig_ready - 1;
            irig_to_send[irig_ind] = irig_packets[offset];
            irig_ind += 1;
            *irig_ready = 0;
            // Update the last time the IRIG data was recorded
            irig_time = curr_time;
        }
        if(*error_ready != 0) {
            offset = *error_ready - 1;
            error_to_send[err_ind] = error_packets[offset];
            err_ind += 1;
            *error_ready = 0;
        }
        // Send encoder data if the buffer is full
        if(encd_ind == ENCODER_PACKETS_TO_SEND) {
            sendto(sockfd, (struct EncoderInfo *) encoder_to_send, sizeof(encoder_to_send), 
                   MSG_CONFIRM, (const struct sockaddr *) &servaddr, sizeof(servaddr));
            encd_ind = 0;
        }
	// Send IRIG data if the buffer is full
        if(irig_ind == IRIG_PACKETS_TO_SEND) {
            sendto(sockfd, (struct IRIGInfo *) irig_to_send, sizeof(irig_to_send),
                   MSG_CONFIRM, (const struct sockaddr *) &servaddr, sizeof(servaddr));
            irig_ind = 0;
        }
	// Send error data if the buffer is full
        if(err_ind == ERROR_PACKETS_TO_SEND) {
	    printf("%lu: sending error packets\n", curr_time);
            sendto(sockfd, (struct ErrorInfo *) error_to_send, sizeof(error_to_send), MSG_CONFIRM,
                   (const struct sockaddr *) &servaddr, sizeof(servaddr));
            err_ind = 0;
        }
	    
        // Send timeout packets if no packets have been picked up in a while
        if(((double) (curr_time - encd_time))/CLOCKS_PER_SEC > ENCODER_TIMEOUT) {
	    printf("%lu: sending encoder timeout packet\n", curr_time);
            timeout_packet->type = ENCODER_TIMEOUT_FLAG;
            sendto(sockfd, (struct TimeoutInfo *) &timeout_packet, sizeof(*timeout_packet),
                   MSG_CONFIRM, (const struct sockaddr *) &servaddr, sizeof(servaddr));
	    // Reset the last time the encoder was monitored
	    encd_time = curr_time;
        }
        if(((double) (curr_time - irig_time))/CLOCKS_PER_SEC > IRIG_TIMEOUT) {
	    printf("%lu: sending IRIG timeout packets\n", curr_time);
            timeout_packet->type = IRIG_TIMEOUT_FLAG;
            sendto(sockfd, (struct TimeoutInfo *) &timeout_packet, sizeof(*timeout_packet),
                   MSG_CONFIRM, (const struct sockaddr *) &servaddr, sizeof(servaddr));
	    // Reset the last time the IRIG was monitored
	    irig_time = curr_time;
        }
    }

    // Disable PRUs when finished
    if(*on == 1) {
        prussdrv_pru_wait_event(PRU_EVTOUT_1);
        printf("All done\n");
        prussdrv_pru_disable(1);
        prussdrv_pru_disable(0);
        prussdrv_exit();
    }

    return 0;
}
