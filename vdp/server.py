import random
#from socket import *
import pickle
import socketserver
from vdp.vision_data_processor import VisionDataProcessor

class SocketReciever(socketserver.BaseRequestHandler):

    def handle(self):
        stargate = self.request.makefile('rb')
        arrival = pickle.load(stargate)
        print(arrival.spaceship.info())

serverSocket = socket(AF_INET, SOCK_DGRAM)
serverSocket.bind(('', 12000))

print("[INFO] Starting server...")

while True:
    rand = random.randint(0, 10)
    message, address = serverSocket.recvfrom(1024)
    message = message.upper()
    if rand >= 4:
        serverSocket.sendto(message, address)
        break




if __name__ == "__main__":
    HOST, PORT = "localhost", 6668

    # Create the server, binding to localhost on port 9999
    server = socketserver.TCPServer((HOST, PORT), SocketReciever)
    server.serve_forever()