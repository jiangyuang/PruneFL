import socket
import struct
from threading import Thread
from time import sleep

from bases.fl.messages import MessageTypes, ClientToServerAckMessage
from utils.save_load import dumps, loads

__all__ = ["ServerSocket", "ClientSocket"]


class Socket(socket.socket):
    def recv_msg(self):
        msg_len = struct.unpack(">I", self.recv(4))[0]
        msg = self.recv(msg_len, socket.MSG_WAITALL)
        msg = loads(msg)
        return msg

    @staticmethod
    def send_to_sock(sock, msg):
        msg_pickled = dumps(msg)
        sock.sendall(struct.pack(">I", len(msg_pickled)))
        sock.sendall(msg_pickled)

    @staticmethod
    def recv_msg_async(sock, ret: list, index):
        msg_len = struct.unpack(">I", sock.recv(4))[0]
        msg = sock.recv(msg_len, socket.MSG_WAITALL)
        msg = loads(msg)
        ret[index] = msg


class ServerSocket(Socket):
    def __init__(self, server_addr, server_port, n_clients):
        super(ServerSocket, self).__init__(socket.AF_INET, socket.SOCK_STREAM)
        self.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.bind((server_addr, server_port))

        self.n_clients = n_clients
        self.list_client_sockets = []

    def init_connections(self, init_msg):
        self.send_msg_to_all(init_msg)
        self.recv_ack_msg_from_all()

    def wait_for_connections(self):
        while len(self.list_client_sockets) < self.n_clients:
            self.listen(self.n_clients * 2)
            print("Waiting for {} connections...".format(self.n_clients))
            (client_sock, (ip, port)) = self.accept()
            self.list_client_sockets.append(client_sock)
            print('New connection from {}:{}, ({}/{})'.format(ip, port, len(self.list_client_sockets), self.n_clients))

    def recv_msg_from_all(self, expected_msg_type: MessageTypes = None):
        msgs = [None for _ in range(self.n_clients)]
        threads = []
        for idx in range(self.n_clients):
            t = Thread(target=self.recv_msg_async, args=(self.list_client_sockets[idx], msgs, idx))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        for msg in msgs:
            if msg is None:
                raise RuntimeError("Message incomplete")
            elif expected_msg_type is not None and msg.msg_type != expected_msg_type:
                raise TypeError("Message type should be ", expected_msg_type)

        return msgs

    def recv_ack_msg_from_all(self):
        return self.recv_msg_from_all(MessageTypes.ClientToServerAckMessage)

    def recv_update_msg_from_all(self):
        return self.recv_msg_from_all(MessageTypes.ClientToServerUpdateMessage)

    def send_msg_to_all(self, msgs):
        """
        Supports both one message and list of messages
        """
        threads = []
        for idx in range(self.n_clients):
            msg = msgs[idx] if isinstance(msgs, list) else msgs
            t = Thread(target=self.send_to_sock, args=(self.list_client_sockets[idx], msg))
            t.start()
            threads.append(t)


class ClientSocket(Socket):
    def __init__(self, server_addr, server_port):
        super(ClientSocket, self).__init__()
        self.server_addr = server_addr
        self.server_port = server_port

    def init_connections(self, max_try=100):
        self.connect_to_server(max_try=max_try)
        print("Connected to server.")
        init_msg = self.recv_init_msg()
        self.send_ack_msg()
        return init_msg

    def connect_to_server(self, max_try=100):
        for _ in range(max_try):
            try:
                self.connect((self.server_addr, self.server_port))
                break
            except ConnectionRefusedError:
                sleep(1)
        else:
            raise ConnectionRefusedError("Connection refused")

    def recv_msg_check_type(self, expected_msg_type: MessageTypes):
        msg = self.recv_msg()
        if msg.msg_type != expected_msg_type:
            raise TypeError("Message type should be ", expected_msg_type)
        return msg

    def recv_init_msg(self):
        return self.recv_msg_check_type(MessageTypes.ServerToClientInitMessage)

    def recv_update_msg(self):
        return self.recv_msg_check_type(MessageTypes.ServerToClientUpdateMessage)

    def send_msg(self, msg):
        self.send_to_sock(self, msg)

    def send_ack_msg(self):
        self.send_msg(ClientToServerAckMessage())
