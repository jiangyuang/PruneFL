import numpy as np
import random
import argparse
import os
from timeit import default_timer as timer
import torch
from copy import deepcopy
from utils.save_load import mkdir_save

from abc import ABC, abstractmethod


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--client-selection',
                        help="If use client-selection",
                        action='store_true',
                        dest='client_selection',
                        default=False,
                        required=False)
    parser.add_argument('-s', '--seed',
                        help="The seed to use for the prototype",
                        action='store',
                        dest='seed',
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument('-e', '--exp-name',
                        help="Experiment name",
                        action='store',
                        dest='experiment_name',
                        type=str,
                        required=True)

    return parser.parse_args()


@torch.no_grad()
def reset_dense_mask(model):
    for layer in model.prunable_layers:
        layer.mask = torch.ones_like(layer.mask, dtype=torch.bool)


def retain_by_num(model, k):
    """
    k: # params to keep
    """
    reset_dense_mask(model)
    nelement = model.nelement()
    model.prune_by_pct([1 - k / nelement for _ in model.prunable_layers])


def calc_cost(k, config, model):
    cost = config.TIME_CONSTANT
    frac = k / model.nelement()
    for layer, comp_coeff in zip(model.prunable_layers, config.COMP_COEFFICIENTS):
        cost += frac * layer.num_weight * (comp_coeff + config.COMM_COEFFICIENT)
    return cost


class OcoGradEstimation:
    def __init__(self, k_min, k_max):
        self.k_min_orig = k_min
        self.k_max_orig = k_max
        self.k_min = k_min
        self.k_max = k_max
        self.d = k_max - k_min
        # NEW: min. value for k_aux_next is self.k_min * self.delta   # Tunable parameter
        self.delta = 0.1

        self.min_max_update_window = 20  # Tunable parameter
        self.alpha = 1.5  # Tunable parameter
        self.min_max_update_count = 0
        self.k_min_window = self.k_max_orig  # Note inverse min/max assignment
        self.k_max_window = self.k_min_orig  # Note inverse min/max assignment
        self.reference_iter = 0
        self.m_prev = 0

        self.timer = 0
        self.grad_value_prev = 0

    def tuning_k_grad_sign(self, k, k_aux, cost, cost_aux, time):  # k>k_aux
        eta = self.d / np.sqrt(2 * (time - self.reference_iter))
        k_unchanged_due_to_cost_none = False
        if cost_aux is None:
            # if k > eta and self.timer <= 5:
            if self.timer <= 10:
                k_next = k
                self.timer += 1
                k_unchanged_due_to_cost_none = True
            else:
                self.timer = 0
                k_next = self.stochastic_rounding(k + eta)  # When any loss of k and k_aux increases
        else:
            self.timer = 0
            k_next = self.stochastic_rounding(k - eta * np.sign((cost - cost_aux) / (k - k_aux)))  # To minimize cost

        if k_next < self.k_min:
            k_next = self.k_min
        elif k_next > self.k_max:
            k_next = self.k_max

        k_aux_next = self.stochastic_rounding(k_next - eta / 2.0)
        if k_aux_next < self.k_min * self.delta:
            k_aux_next = int(np.ceil(self.k_min * self.delta))

        if k_aux_next >= k_next:
            k_aux_next = k_next - 1

        if not k_unchanged_due_to_cost_none:
            self.k_min_window = min(self.k_min_window, k_next)
            self.k_max_window = max(self.k_max_window, k_next)
            self.min_max_update_count += 1

        if self.min_max_update_count >= self.min_max_update_window:
            self.min_max_update_count = 0
            k_min_window_change = self.k_min_window / self.alpha
            k_max_window_change = self.k_max_window * self.alpha
            k_min_window_change = int(np.round(max(k_min_window_change, self.k_min_orig)))
            k_max_window_change = int(np.round(min(k_max_window_change, self.k_max_orig)))
            b_new = k_max_window_change - k_min_window_change
            b_orig = self.k_max - self.k_min
            m_current = time - self.reference_iter

            if b_new > 0 and m_current >= self.m_prev and b_orig + b_new <= b_orig * np.sqrt(2):
                # if b_new > 0:
                self.k_min = k_min_window_change
                self.k_max = k_max_window_change
                self.d = self.k_max - self.k_min
                self.reference_iter = time
                self.m_prev = m_current
                print('******** New k_min:', self.k_min, 'new k_max:', self.k_max)
            else:
                print('******** Same range - New k_min_window:', self.k_min, 'new k_max_window:', self.k_max, 'b_orig:',
                      b_orig, 'b_new:', self.k_max - self.k_min)
                print('m_current:', m_current, 'self.m_prev:', self.m_prev)
                print('b_orig * np.sqrt(m_current) + b_new * np.sqrt(self.min_max_update_window) =',
                      b_orig * np.sqrt(m_current) + b_new * np.sqrt(self.min_max_update_window))
                print('b_orig * np.sqrt(m_current + self.min_max_update_window) =',
                      b_orig * np.sqrt(m_current + self.min_max_update_window))

            self.k_min_window = self.k_max_orig  # Note inverse min/max assignment
            self.k_max_window = self.k_min_orig  # Note inverse min/max assignment

        return k_next, k_aux_next

    def tuning_k_grad_value(self, k, k_aux, cost, cost_aux, time):  # k>k_aux
        eta = self.d / np.sqrt(2 * time)
        grad_value = None
        if cost_aux is None:
            if self.timer <= 10:
                k_next = k
                self.timer += 1
            else:
                self.timer = 0
                k_next = self.stochastic_rounding(
                    k + eta * self.grad_value_prev)  # When any loss of k and k_aux increases
        else:
            self.timer = 0
            grad_value = (cost - cost_aux) / (k - k_aux)
            k_next = self.stochastic_rounding(k - eta * grad_value)  # To minimize cost
            self.grad_value_prev = grad_value

        if k_next < self.k_min:
            k_next = self.k_min
        elif k_next > self.k_max:
            k_next = self.k_max

        if grad_value is None:
            grad_value = self.grad_value_prev
        k_aux_next = self.stochastic_rounding(k_next - eta * np.abs(grad_value) / 2.0)
        if k_aux_next < self.k_min * self.delta:
            k_aux_next = int(np.ceil(self.k_min * self.delta))

        if k_aux_next >= k_next:
            k_aux_next = k_next - 1

        return k_next, k_aux_next

    @staticmethod
    def stochastic_rounding(x):
        floor_x = int(np.floor(x))
        prob = random.random()
        if prob < x - floor_x:
            x = floor_x + 1
        else:
            x = floor_x
        return x


class OnlineServer(ABC):
    def __init__(self, args, config, model, save_interval=50):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.experiment_name = args.experiment_name
        self.save_path = os.path.join("results", config.EXP_NAME, args.experiment_name)
        self.save_interval = save_interval

        self.model = model.to(self.device)
        self.model.train()
        mkdir_save(self.model, os.path.join(self.save_path, "init_model.pt"))

        self.num_all_params = self.model.nelement()
        self.k = self.num_all_params
        self.k_aux = int(np.ceil(self.k * 0.9))
        self.control = OcoGradEstimation(np.round(self.num_all_params * 0.002), self.num_all_params)

        self.test_loader = None
        self.prev_model = None

        self.init_test_loader()
        self.init_clients()

    @abstractmethod
    def init_test_loader(self):
        pass

    @abstractmethod
    def init_clients(self):
        pass

    def main(self, idx, list_sd, list_num_proc, list_data_proc, lr, start, list_loss, list_acc, list_est_time,
             list_model_size):
        total_num_proc = sum(list_num_proc)

        with torch.no_grad():
            for key, param in self.model.state_dict().items():
                avg_inc_val = None
                for num_proc, state_dict in zip(list_num_proc, list_sd):
                    if key in state_dict.keys():
                        mask = self.model.get_mask_by_name(key)
                        if mask is None:
                            inc_val = state_dict[key] - param
                        else:
                            inc_val = state_dict[key] - param * self.model.get_mask_by_name(key)

                        if avg_inc_val is None:
                            avg_inc_val = num_proc / total_num_proc * inc_val
                        else:
                            avg_inc_val += num_proc / total_num_proc * inc_val

                if avg_inc_val is None or key.endswith("num_batches_tracked"):
                    continue
                else:
                    param.add_(avg_inc_val)

        if idx % self.config.EVAL_DISP_INTERVAL == 0:
            loss, acc = self.model.evaluate(self.test_loader)
            list_loss.append(loss)
            list_acc.append(acc)

            print("Round #{} (Experiment = {}).".format(idx, self.experiment_name))
            print("Loss/acc (at round {}) = {}/{}".format((len(list_loss) - 1) * self.config.EVAL_DISP_INTERVAL, loss,
                                                          acc))
            print("Estimated time = {}".format(sum(list_est_time)))
            print("Elapsed time = {}".format(timer() - start))
            print("Current lr = {}".format(lr))
            print("Current density = {}".format(self.model.density()))

        # control
        if len(list_acc) >= 2:
            inputs = None
            labels = None
            for data in list_data_proc:
                inp, lab = data
                if inputs is None:
                    inputs = inp
                else:
                    inputs = torch.cat([inputs, inp], dim=0)

                if labels is None:
                    labels = lab
                else:
                    labels = torch.cat([labels, lab], dim=0)
            prev_loss = self.prev_model.evaluate([(inputs, labels)])[0]
            cur_loss = self.model.evaluate([(inputs, labels)])[0]

            retain_by_num(self.model, self.k_aux)
            cur_aux_loss = self.model.evaluate([(inputs, labels)])[0]

            cost = calc_cost(self.k, self.config, self.model)
            if prev_loss > cur_loss and prev_loss > cur_aux_loss:
                cost_aux = calc_cost(self.k_aux, self.config, self.model)
                cost_aux *= (prev_loss - cur_loss) / (prev_loss - cur_aux_loss)
            else:
                cost_aux = None
            self.k, self.k_aux = self.control.tuning_k_grad_sign(self.k, self.k_aux, cost, cost_aux, idx + 1)
            self.k, self.k_aux = int(self.k), int(self.k_aux)
            retain_by_num(self.model, self.k)

        est_time = self.config.TIME_CONSTANT
        for layer, comp_coeff in zip(self.model.prunable_layers, self.config.COMP_COEFFICIENTS):
            est_time += layer.num_weight * (comp_coeff + self.config.COMM_COEFFICIENT)

        model_size = self.model.calc_num_all_active_params(True)
        list_est_time.append(est_time)
        list_model_size.append(model_size)

        if idx % self.save_interval == 0:
            mkdir_save(list_loss, os.path.join(self.save_path, "loss.pt"))
            mkdir_save(list_acc, os.path.join(self.save_path, "accuracy.pt"))
            mkdir_save(list_est_time, os.path.join(self.save_path, "est_time.pt"))
            mkdir_save(list_model_size, os.path.join(self.save_path, "model_size.pt"))
            mkdir_save(self.model, os.path.join(self.save_path, "model.pt"))

        self.prev_model = deepcopy(self.model)

        return [layer.mask for layer in self.model.prunable_layers], [self.model.state_dict() for _ in
                                                                      range(self.config.NUM_CLIENTS)]


class OnlineClient(ABC):
    def __init__(self, config, model):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_num_upload = config.MAX_NUM_UPLOAD
        self.model = deepcopy(model).to(self.device)
        self.optimizer = None
        self.optimizer_scheduler = None
        self.optimizer_wrapper = None
        self.train_loader = None

        self.list_mask = [None for _ in range(len(self.model.prunable_layers))]

    @abstractmethod
    def init_optimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_train_loader(self, *args, **kwargs):
        pass

    def main(self):
        self.model.train()
        lr = self.optimizer_wrapper.get_last_lr()
        accumulated_inputs = None
        accumulated_labels = None
        num_proc_data = 0
        for _ in range(self.config.NUM_LOCAL_UPDATES):
            with torch.no_grad():
                for layer, mask in zip(self.model.prunable_layers, self.list_mask):
                    if mask is not None:
                        layer.weight *= mask
            inputs, labels = self.train_loader.get_next_batch()
            self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))

            num_proc_data += len(inputs)

            if accumulated_inputs is None:
                accumulated_inputs = inputs
            else:
                accumulated_inputs = torch.cat([accumulated_inputs, inputs], dim=0)
            if accumulated_labels is None:
                accumulated_labels = labels
            else:
                accumulated_labels = torch.cat([accumulated_labels, labels], dim=0)

        with torch.no_grad():
            for layer, mask in zip(self.model.prunable_layers, self.list_mask):
                if mask is not None:
                    layer.weight *= mask

        self.optimizer_wrapper.lr_scheduler_step()

        return self.model.state_dict(), num_proc_data, (accumulated_inputs[:self.max_num_upload],
                                                        accumulated_labels[:self.max_num_upload]), lr

    def load_mask(self, masks):
        self.list_mask = masks

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class OnlineFL:
    def __init__(self, config, server, client_list):
        self.max_round = config.MAX_ROUND
        self.server = server
        self.client_list = client_list

        self.list_loss, self.list_acc, self.list_est_time, self.list_model_size = [], [], [], []

    def main(self):
        start = timer()
        for idx in range(self.max_round):
            list_state_dict, list_num, list_data, list_last_lr = [], [], [], []

            for client in self.client_list:
                sd, npc, data, last_lr = client.main()
                list_state_dict.append(sd)
                list_num.append(npc)
                list_data.append(data)
                list_last_lr.append(last_lr)
            last_lr = list_last_lr[0]
            for client_lr in list_last_lr[1:]:
                assert client_lr == last_lr

            list_mask, new_list_sd = self.server.main(idx, list_state_dict, list_num, list_data, last_lr, start,
                                                      self.list_loss, self.list_acc, self.list_est_time,
                                                      self.list_model_size)
            for client, new_sd in zip(self.client_list, new_list_sd):
                client.load_state_dict(new_sd)
                client.load_mask(list_mask)
