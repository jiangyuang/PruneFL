import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from collections import OrderedDict
from typing import Union, List


class SGD(optim.SGD):
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("closure not supported")

        list_grad = []

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if (p.grad is None and not hasattr(p, "is_sparse_param")) or hasattr(p, "is_placeholder"):
                    # exclude 1) dense param with None grad and 2) dense placeholders for sparse params
                    continue
                elif hasattr(p, "is_sparse_param"):
                    d_p = p.dense.grad.masked_select(p.mask)
                    if weight_decay != 0:
                        d_p = d_p.add(p._values(), alpha=weight_decay)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf

                    p._values().add_(d_p, alpha=-group['lr'])

                else:
                    d_p = p.grad
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf

                    p.add_(d_p, alpha=-group['lr'])

                list_grad.append(d_p.clone())
        return list_grad

    def clear_state(self):
        for state in self.state.values():
            if "momentum_buffer" in state:
                del state["momentum_buffer"]
