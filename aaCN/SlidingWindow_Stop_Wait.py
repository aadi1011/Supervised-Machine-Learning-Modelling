# implementing a sliding window method to perform Stop&Wait method protocol to simulate data trasfer between two devices in a network

# Create two lists, one for sender and one for receiver
# Each list has 8 elements, each element is a tuple (data, ack)
Comp1 = [(0,0)]*8
Comp2 = [(0,0)]*8

# Create a list to store the data to be sent
data = [1,2,3,4,5,6,7,8]

# Create a list to store the acks received
ack = [0,0,0,0,0,0,0,0]

# Create a list to store the data received
data_received = [0,0,0,0,0,0,0,0]

# Create a list to store the acks to be sent
ack_to_send = [0,0,0,0,0,0,0,0]

# Function to send data from sender to receiver
# Set slide window size to 4
def send_data():
    global data, ack, data_received, ack_to_send, Comp1, Comp2
    # Send data from sender to receiver
    for i in range(0,4):
        # Check if the ack is 0
        if ack[i] == 0:
            # Send the data
            Comp2[i] = (data[i], 0)
            # Update the ack to 1
            ack[i] = 1
            # Update the ack to be sent to 1
            ack_to_send[i] = 1
            print("Sending data: ", Comp2[i], " from Comp1 to receiver, Comp2")
            print("Sending ack: ", ack_to_send[i], " from Comp2 to sender, Comp1")
            print("------------------------------------------------------------")
        else: # if ack is already 1
            print("As ack is 1, the data has been sent and received, so we can move the window")
            print("Moving the window to next set of data")
            # Send the data
            Comp2[i] = (data[i], 0)
            # Update the ack to 1
            ack[i] = 1
            # Update the ack to be sent to 0
            ack_to_send[i] = 0
            print("Sending data: ", Comp2[i], " from Comp1 to receiver, Comp2")
            print("Sending ack: ", ack_to_send[i], " from Comp2 to sender, Comp1")
            print("------------------------------------------------------------")

# Function to receive data from sender
def receive_data():
    global data, ack, data_received, ack_to_send, Comp1, Comp2
    # Receive data from sender
    for i in range(0,4):
        # Check if the ack is 0
        if ack_to_send[i] == 0:
            # Receive the data
            data_received[i] = Comp1[i][0]
            # Update the ack to 1
            ack_to_send[i] = 1
            print("Receiving data: ", data_received[i], " from Comp1 to receiver, Comp2")
            print("Receiving ack: ", ack_to_send[i], " from Comp2 to sender, Comp1")
            print("------------------------------------------------------------")
        else: # if ack is already 1
            print("As ack is 1, the data has been sent and received, so we can move the window")
            print("Moving the window to next set of data")
            # Receive the data
            data_received[i] = Comp1[i][0]
            # Update the ack to 0
            ack_to_send[i] = 0
            print("Receiving data: ", data_received[i], " from Comp1 to receiver, Comp2")
            print("Receiving ack: ", ack_to_send[i], " from Comp2 to sender, Comp1")
            print("------------------------------------------------------------")

# Function to print the data sent and received
def print_data():
    global data, ack, data_received, ack_to_send, Comp1, Comp2
    # From Comp1 to Comp2
    print("\n\nPRINTING DATA")
    print("From Comp1 to Comp2")
    print("Data sent: ", data)
    print("Ack sent: ", ack)
    print("Data received: ", data_received)
    print("Ack received: ", ack_to_send)
    # From Comp2 to Comp1
    print("\n\nFrom Comp2 to Comp1")
    print("Data sent: ", data_received)
    print("Ack sent: ", ack_to_send)
    print("Data received: ", data)
    print("Ack received: ", ack)

# Sliding window method
def sliding_window():
    global data, ack, data_received, ack_to_send, Comp1, Comp2
    print("Data sent and received before sending data: ")
    print_data()
    print("------------------------------------------------------------")
    # Send data from sender to receiver
    send_data()
    # Receive data from sender
    receive_data()
    # Print the data sent and received
    print("Data sent and received after sending data: ")
    print_data()
    print("------------------------------------------------------------")

# Call the sliding window method
sliding_window()