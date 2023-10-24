# Python Program that simulates and implements the Go-Back-N protocol using a sliding window method.
# Author: Aadith Sukumar (https://www.github.com/aadi1011)

class GoBackN:
    def __init__(self, window_size, data):
        # Initialize the GoBackN protocol with the given window size and data
        self.window_size = window_size
        self.data = data
        self.next_seq_num = 0

    def send_packet(self, seq_num):
        # Simulate sending a packet and print the details
        print(f"Comp1: Sending packet {seq_num}: {self.data[seq_num]}")

    def simulate(self):
        while self.next_seq_num < len(self.data):
            # Send packets within the window
            for i in range(self.next_seq_num, min(self.next_seq_num + self.window_size, len(self.data))):
                self.send_packet(i)

            # Receive acknowledgments
            ack = int(input("Comp2: Enter received ACK (enter -1 for timeout): "))
            if ack >= self.next_seq_num:
                # If received acknowledgment is within the current window, update next_seq_num
                print(f"Comp1: Received ACK for packet {ack}")
                self.next_seq_num = ack + 1
            else:
                # If acknowledgment is not received for some packets, resend the current window
                print("Comp1: Timeout - Resending packets in current window")

if __name__ == "__main__":
    data = ["Data0", "Data1", "Data2", "Data3", "Data4", "Data5", "Data6", "Data7", "Data8"]
    window_size = 3

    # Create an instance of GoBackN protocol and start the simulation
    protocol = GoBackN(window_size, data)
    protocol.simulate()