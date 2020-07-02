import socket
import sys
import json

import socket
import tqdm
import os

files_new = [
    'samples_new/688AB5015D8A%3A02-energy_aminus_inc.csv',
    'samples_new/688AB5015D8A%3A02-energy_aplus_inc.csv',
    'samples_new/688AB5015D8A%3A02-power_aplus_peak.csv'
]

files = [
    'samples/688AB5004D91_02-energy_aminus_inc.csv',
    'samples/688AB5004D91_02-energy_aplus_inc.csv',
    'samples/688AB5004D91_02-power_aplus_peak.csv'
]

SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 4096 # send 4096 bytes each time step

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

# create the client socket
s = socket.socket()

print(f"[+] Connecting to {HOST}:{PORT}")
s.connect((HOST, PORT))
print("[+] Connected.")

k = 0
while k < 3:
    #filename = sys.argv[k+1]
    filename = files[k]
    size = len(filename)
    size = bin(size)[2:].zfill(16) # encode filename size as 16 bit binary
    s.send(bytes(size, encoding="utf-8"))
    s.send(bytes(filename, encoding="utf-8"))

    filesize = os.path.getsize(filename)
    filesize = bin(filesize)[2:].zfill(32) # encode filesize as 32 bit binary
    s.send(bytes(filesize, encoding="utf-8"))

    file_to_send = open(filename, 'rb')

    l = file_to_send.read()
    s.sendall(l)
    file_to_send.close()
    print('File Sent')
    
    k += 1
# close the socket
s.close()

