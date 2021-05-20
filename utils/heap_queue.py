import heapq
from typing import Iterable


class HeapQueue:
    def __init__(self, init_h: Iterable):
        self.h = [(-val, index) for index, val in init_h]
        heapq.heapify(self.h)

    def replace_largest(self, new_val):
        heapq.heapreplace(self.h, (-new_val, self.max_index))

    def pop(self):
        heapq.heappop(self.h)

    @property
    def max_index(self):
        return self.h[0][1]

    @property
    def max_val(self):
        return -self.h[0][0]

    def __repr__(self):
        return "HeapQueue instance containing data {}.".format(self.h)
