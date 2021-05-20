from enum import Enum
from typing import Any, Tuple

__all__ = ["ServerToClientInitMessage", "ServerToClientUpdateMessage", "ClientToServerAckMessage",
           "ClientToServerUpdateMessage"]


class MessageTypes(Enum):
    ServerToClientInitMessage = 0
    ServerToClientUpdateMessage = 1
    ClientToServerAckMessage = 2
    ClientToServerUpdateMessage = 3


class BaseMessage:
    def __init__(self, msg_type: MessageTypes, data):
        self.msg_type = msg_type
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __repr__(self):
        return str(self.msg_type)


class ServerToClientInitMessage(BaseMessage):
    def __init__(self, data: Tuple[int, Any, Any, tuple, Tuple[bool, int, bool]]):
        super(ServerToClientInitMessage, self).__init__(MessageTypes.ServerToClientInitMessage, data)

    @property
    def client_id(self) -> int:
        return self.data[0]

    @property
    def exp_config(self):
        return self.data[1]

    @property
    def model(self):
        return self.data[2]

    @property
    def extra_params(self) -> tuple:
        return self.data[3]

    @property
    def resume_params(self) -> tuple:
        # tuple(resume: bool, round: int, resume_to_sparse: bool)
        return self.data[4]


class ServerToClientUpdateMessage(BaseMessage):
    def __init__(self, data: Tuple[dict, bool, bool, bool]):
        super(ServerToClientUpdateMessage, self).__init__(MessageTypes.ServerToClientUpdateMessage, data)

    @property
    def state_dict(self):
        return self.data[0]

    @property
    def adjustment(self) -> bool:
        return self.data[1]

    @property
    def to_sparse(self) -> bool:
        return self.data[2]

    @property
    def terminate(self) -> bool:
        return self.data[3]


class ClientToServerAckMessage(BaseMessage):
    def __init__(self):
        super(ClientToServerAckMessage, self).__init__(MessageTypes.ClientToServerAckMessage, None)

    def __getitem__(self, key):
        raise IndexError("ClientToServerAckMessage is not subscriptable")


class ClientToServerUpdateMessage(BaseMessage):
    def __init__(self, data: Tuple[dict, int, float]):
        super(ClientToServerUpdateMessage, self).__init__(MessageTypes.ClientToServerUpdateMessage, data)

    @property
    def state_dict(self):
        return self.data[0]

    @property
    def num_processed(self):
        return self.data[1]

    @property
    def lr(self):
        return self.data[2]
