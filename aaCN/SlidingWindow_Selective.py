#selective repeat

import random

class SelectiveRepeat:
    def __init__(self, window_size, data):
        self.window_size = window_size
        self.data = data
        self.next_seq_num = 0
        self.received_ack = set()
    
    def send_packet(self, seq_num):
        if seq_num < len(self.data):
            print(f"Comp1: Sending packet {seq_num}: {self.data[seq_num]}")

    def simulate(self):
        while self.next_seq_num < len(self.data):
            ack_range_start = self.next_seq_num
            ack_range_end = min(self.next_seq_num + self.window_size, len(self.data))

            # Send packets within the window
            for i in range(self.next_seq_num, min(self.next_seq_num + self.window_size, len(self.data))):
                self.send_packet(i)

            # Simulate receiving acknowledgments for the entire window
            ack_received = [random.choice([True, False]) for _ in range(ack_range_start, ack_range_end)]

            for i, ack in enumerate(ack_received):
                if ack:
                    print(f"Comp1: Received ACK for packet {ack_range_start + i}")
                    self.received_ack.add(ack_range_start + i)

            # Check for missing ACKs and resend missing packets
            missing_packets = [i for i in range(ack_range_start, ack_range_end) if i not in self.received_ack]

            while missing_packets:
                for packet in missing_packets:
                    print(f"Comp1: Did not receive ACK for packet {packet}... resending...")
                
                # Simulate receiving acknowledgments for resent packets
                ack_received = [random.choice([True, False]) for _ in missing_packets]

                for i, ack in enumerate(ack_received):
                    if ack:
                        print(f"Comp1: Received ACK for resent packet {missing_packets[i]}")
                        self.received_ack.add(missing_packets[i])

                # Check for missing ACKs after resending
                missing_packets = [i for i in missing_packets if i not in self.received_ack]

            self.next_seq_num = ack_range_end

if __name__ == "__main__":
    data = ["Data0", "Data1", "Data2", "Data3", "Data4", "Data5", "Data6", "Data7", "Data8", "Data9", "Data10"]
    window_size = 11  # Adjust window size as needed

    protocol = SelectiveRepeat(window_size, data)
    protocol.simulate()