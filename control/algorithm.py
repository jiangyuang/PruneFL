import torch
from timeit import default_timer as timer

from utils.heap_queue import HeapQueue


def process_layer(layer, layer_prefix, sgrad: dict, coeff, dtp) -> (float, float, list, torch.Tensor):
    w_name = "{}.weight".format(layer_prefix)
    b_name = "{}.bias".format(layer_prefix)
    sqg = sgrad[w_name]
    iu_mask, niu_mask = layer.mask == 1., layer.mask == 0.
    num_iu, num_niu = iu_mask.sum().item(), niu_mask.sum().item()

    # Decrease
    max_dec_num = int(dtp * num_iu)
    w_iu = layer.weight[iu_mask]  # use magnitude
    w_thr = torch.sort(torch.abs(w_iu))[0][max_dec_num]
    tbk_mask = (torch.abs(layer.weight) >= w_thr) * iu_mask
    tba_dec_mask = (torch.abs(layer.weight) < w_thr) * iu_mask

    # Increase
    tba_inc_mask = niu_mask

    total_sqg = sqg[tbk_mask].sum().item()
    if b_name in sgrad.keys():
        total_sqg += sgrad[b_name].sum().item()
    total_time = coeff * tbk_mask.sum().item()
    tba_mask = tba_dec_mask + tba_inc_mask
    tba_values, tba_indices = sqg[tba_mask], tba_mask.nonzero()
    sorted_tba_values, sort_perm = torch.sort(tba_values, descending=True)
    sorted_tba_indices = tba_indices[sort_perm]

    layer.prune_by_pct(dtp)

    return total_sqg, total_time, sorted_tba_values.tolist(), sorted_tba_indices


def architecture_search(model, sum_sqg, sum_time, list_coefficient, list_tba_values, list_tba_indices,
                        max_density=None):
    list_len = [len(tba) for tba in list_tba_values]
    list_iter = [iter(tba) for tba in list_tba_values]
    # number of params to be added/removed
    list_n = [0] * len(list_len)

    heap = HeapQueue([(index, next(_iter) / _coeff) for index, (_iter, _coeff, _len) in
                      enumerate(zip(list_iter, list_coefficient, list_len)) if _len > 0])
    numerator = sum_sqg
    denominator = sum_time

    num_params, max_num = None, None
    if max_density is not None:
        num_params, max_num = model.calc_num_prunable_params(False)
        max_num = int(max_num * max_density)

    end_condition = False
    while not end_condition:
        obj_val = numerator / denominator
        pos, val = heap.max_index, heap.max_val
        if val > obj_val:
            if max_num is not None:
                if num_params > max_num:
                    print("Exceeds max num")
                    break
                else:
                    num_params += 1
            coeff = list_coefficient[pos]
            numerator += val * coeff
            denominator += coeff
            list_n[pos] += 1
            if list_n[pos] == list_len[pos]:
                heap.pop()
            else:
                heap.replace_largest(next(list_iter[pos]) / coeff)
        else:
            end_condition = True

    print(list_n)

    for layer, tba_indices, tba_n in zip(model.prunable_layers, list_tba_indices, list_n):
        layer.mask[tba_indices[:tba_n].t().tolist()] = 1.


def main_control(model, squared_grad_dict: dict, config, dec_thr_pct, max_density=None):
    sum_sqg = 0
    sum_time = config.TIME_CONSTANT
    list_tba_values, list_tba_indices = [], []
    list_coefficient = []

    proc_start = timer()
    comp_coeff_iter = iter(config.COMP_COEFFICIENTS)
    comm_coeff = config.COMM_COEFFICIENT
    for layer, layer_prefix in zip(model.param_layers, model.param_layer_prefixes):
        if layer_prefix in model.prunable_layer_prefixes:
            coeff = comm_coeff + next(comp_coeff_iter)
            sqg, time, sorted_tba_values, sorted_tba_indices = process_layer(layer, layer_prefix, squared_grad_dict,
                                                                             coeff, dec_thr_pct)
            sum_sqg += sqg
            sum_time += time

            list_coefficient.append(coeff)
            list_tba_values.append(sorted_tba_values)
            list_tba_indices.append(sorted_tba_indices)
        else:
            w_name = "{}.weight".format(layer_prefix)
            b_name = "{}.bias".format(layer_prefix)
            sqg = squared_grad_dict[w_name]
            sum_sqg += sqg.sum().item()
            if b_name in squared_grad_dict.keys():
                sum_sqg += squared_grad_dict[b_name].sum().item()

    print("\tProcessing layers, time = {}.".format(timer() - proc_start))
    nas_start = timer()
    architecture_search(model, sum_sqg, sum_time, list_coefficient, list_tba_values, list_tba_indices, max_density)
    print("\tNAS time = {}.".format(timer() - nas_start))


class ControlModule:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.squared_grad_dict = dict()

    @torch.no_grad()
    def accumulate(self, key, sgrad):
        if key in self.squared_grad_dict.keys():
            self.squared_grad_dict[key] += sgrad
        else:
            self.squared_grad_dict[key] = sgrad

    def adjust(self, dec_thr_pct, max_density=None):
        main_control(self.model, self.squared_grad_dict, self.config, dec_thr_pct, max_density)
        self.squared_grad_dict = dict()
