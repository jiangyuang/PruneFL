import argparse
import os
from copy import deepcopy
import torch
from utils.save_load import mkdir_save
from utils.functional import disp_num_params
from timeit import default_timer as timer
from utils.functional import deepcopy_dict

from abc import ABC, abstractmethod


def parse_args():
    parser = argparse.ArgumentParser()
    mutex = parser.add_mutually_exclusive_group(required=True)
    mutex.add_argument('-a', '--adaptive',
                       help="Use adaptive pruning",
                       action='store_true',
                       dest='use_adaptive')
    mutex.add_argument('-na', '--no-adaptive',
                       help="Do not use adaptive pruning",
                       action='store_false',
                       dest='use_adaptive')

    mutex1 = parser.add_mutually_exclusive_group(required=True)
    mutex1.add_argument('-i', '--init-pruning',
                        help="Use initial pruning",
                        action='store_true',
                        dest='initial_pruning')
    mutex1.add_argument('-ni', '--no-init-pruning',
                        help="Do not use initial pruning",
                        action='store_false',
                        dest='initial_pruning')

    parser.add_argument('-c', '--client-selection',
                        help="If use client-selection",
                        action='store_true',
                        dest='client_selection',
                        default=False,
                        required=False)
    parser.add_argument('-t', '--target-density',
                        help="Target density",
                        action='store',
                        dest='target_density',
                        type=float,
                        required=False)
    parser.add_argument('-m', '--max-density',
                        help="Max density",
                        action='store',
                        dest='max_density',
                        type=float,
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


class AdaptiveServer(ABC):
    def __init__(self, args, config, model, save_interval=50):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.experiment_name = args.experiment_name
        self.save_path = os.path.join("results", config.EXP_NAME, args.experiment_name)
        self.save_interval = save_interval
        self.use_adaptive = args.use_adaptive
        self.client_selection = args.client_selection

        if self.use_adaptive:
            print("Init max dec = {}. "
                  "Adjustment dec half-life = {}. "
                  "Adjustment interval = {}.".format(self.config.MAX_DEC_DIFF, self.config.ADJ_HALF_LIFE,
                                                     self.config.ADJ_INTERVAL))

        self.model = model.to(self.device)
        self.model.train()
        mkdir_save(self.model, os.path.join(self.save_path, "init_model.pt"))

        self.indices = None

        self.ip_train_loader = None
        self.ip_test_loader = None
        self.ip_optimizer_wrapper = None
        self.ip_control = None

        self.test_loader = None
        self.control = None
        self.init_test_loader()
        self.init_clients()
        self.init_control()
        self.init_ip_config()
        self.save_exp_config()

    @abstractmethod
    def init_test_loader(self):
        pass

    @abstractmethod
    def init_clients(self):
        pass

    @abstractmethod
    def init_control(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_exp_config(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_ip_config(self, *args, **kwargs):
        pass

    def initial_pruning(self, list_est_time, list_loss, list_acc, list_model_size):
        svdata, pvdata = self.ip_train_loader.len_data, self.config.IP_DATA_BATCH * self.config.CLIENT_BATCH_SIZE
        assert svdata >= pvdata, "server data ({}) < required data ({})".format(svdata, pvdata)
        server_inputs, server_outputs = [], []
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for _ in range(self.config.IP_DATA_BATCH):
            inp, out = self.ip_train_loader.get_next_batch()
            server_inputs.append(inp.to(dev))
            server_outputs.append(out.to(dev))

        prev_density = None
        prev_num = 5
        prev_ind = []
        start = timer()
        ip_start_adj_round = None

        for server_i in range(1, self.config.IP_MAX_ROUNDS + 1):
            model_size = self.model.calc_num_all_active_params(True)
            list_est_time.append(0)
            list_model_size.append(model_size)

            if (server_i - 1) % self.config.EVAL_DISP_INTERVAL == 0:
                # test data not observable to clients, this evaluation does not happen in real systems
                loss, acc = self.model.evaluate(self.ip_test_loader)
                train_loss, train_acc = self.model.evaluate(zip(server_inputs, server_outputs))
                list_loss.append(loss)
                list_acc.append(acc)
                if ip_start_adj_round is None and train_acc >= self.config.ADJ_THR_ACC:
                    ip_start_adj_round = server_i
                    print("Start reconfiguration in initial pruning at round {}.".format(server_i - 1))
                print("Initial pruning round {}. Accuracy = {}. Loss = {}. Train accuracy = {}. Train loss = {}. "
                      "Elapsed time = {}.".format(server_i - 1, acc, loss, train_acc, train_loss, timer() - start))

            for server_inp, server_out in zip(server_inputs, server_outputs):
                list_grad = self.ip_optimizer_wrapper.step(server_inp, server_out)
                for (key, param), g in zip(self.model.named_parameters(), list_grad):
                    assert param.size() == g.size()
                    self.ip_control.accumulate(key, g ** 2)

            if ip_start_adj_round is not None and (server_i - ip_start_adj_round) % self.config.IP_ADJ_INTERVAL == 0:
                self.ip_control.adjust(self.config.MAX_DEC_DIFF)
                cur_density = disp_num_params(self.model)

                if prev_density is not None:
                    prev_ind.append(abs(cur_density / prev_density - 1) <= self.config.IP_THR)
                prev_density = cur_density

                if len(prev_ind) >= prev_num and all(prev_ind[-prev_num:]):
                    print("Early-stopping initial pruning at round {}.".format(server_i - 1))
                    del list_loss[-1]
                    del list_acc[-1]
                    break

        len_pre_rounds = len(list_acc)
        print("End initial pruning. Total rounds = {}. Total elapsed time = {}.".format(
            len_pre_rounds * self.config.EVAL_DISP_INTERVAL, timer() - start))

        return len_pre_rounds

    def main(self, idx, list_sd, list_num_proc, lr, list_accumulated_sgrad, start, list_loss, list_acc, list_est_time,
             list_model_size, is_adj_round, density_limit=None):
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
            print("Loss/acc (at round #{}) = {}/{}".format((len(list_loss) - 1) * self.config.EVAL_DISP_INTERVAL, loss,
                                                           acc))
            print("Estimated time = {}".format(sum(list_est_time)))
            print("Elapsed time = {}".format(timer() - start))
            print("Current lr = {}".format(lr))

        if self.use_adaptive and is_adj_round:
            alg_start = timer()

            for d in list_accumulated_sgrad:
                for k, sg in d.items():
                    self.control.accumulate(k, sg)

            print("Running adaptive pruning algorithm")
            max_dec_diff = self.config.MAX_DEC_DIFF * (0.5 ** (idx / self.config.ADJ_HALF_LIFE))
            self.control.adjust(max_dec_diff, max_density=density_limit)
            print("Total alg time = {}. Max density = {}.".format(timer() - alg_start, density_limit))
            print("Num params:")
            disp_num_params(self.model)

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

        return [layer.mask for layer in self.model.prunable_layers], [self.model.state_dict() for _ in
                                                                      range(self.config.NUM_CLIENTS)]


class AdaptiveClient:
    def __init__(self, model, config, use_adaptive):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_adaptive = use_adaptive
        self.model = deepcopy(model).to(self.device)
        self.optimizer = None
        self.optimizer_scheduler = None
        self.optimizer_wrapper = None
        self.train_loader = None

        self.list_mask = [None for _ in range(len(self.model.prunable_layers))]
        if self.use_adaptive:
            self.accumulated_sgrad = dict()

    @abstractmethod
    def init_optimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_train_loader(self, *args, **kwargs):
        pass

    def main(self, is_adj_round):
        self.model.train()
        num_proc_data = 0

        lr = self.optimizer_wrapper.get_last_lr()

        accumulated_grad = dict()
        for _ in range(self.config.NUM_LOCAL_UPDATES):
            with torch.no_grad():
                for layer, mask in zip(self.model.prunable_layers, self.list_mask):
                    if mask is not None:
                        layer.weight *= mask
            inputs, labels = self.train_loader.get_next_batch()
            list_grad = self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))

            num_proc_data += len(inputs)

            for (key, param), g in zip(self.model.named_parameters(), list_grad):
                assert param.size() == g.size()  # only simulation
                if key in accumulated_grad.keys():
                    accumulated_grad[key] += param.grad  # g
                else:
                    accumulated_grad[key] = param.grad  # g

        with torch.no_grad():
            if self.use_adaptive:
                for key, grad in accumulated_grad.items():
                    if key in self.accumulated_sgrad.keys():
                        self.accumulated_sgrad[key] += (grad ** 2) * num_proc_data
                    else:
                        self.accumulated_sgrad[key] = (grad ** 2) * num_proc_data

            for layer, mask in zip(self.model.prunable_layers, self.list_mask):
                if mask is not None:
                    layer.weight *= mask

        self.optimizer_wrapper.lr_scheduler_step()

        if self.use_adaptive and is_adj_round:
            sgrad_to_upload = deepcopy_dict(self.accumulated_sgrad)
            self.accumulated_sgrad = dict()
        else:
            sgrad_to_upload = {}
        return self.model.state_dict(), num_proc_data, sgrad_to_upload, lr

    def load_mask(self, masks):
        self.list_mask = masks

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class AdaptiveFL(ABC):
    def __init__(self, args, config, server, client_list):
        self.config = config
        self.use_ip = args.initial_pruning
        self.use_adaptive = args.use_adaptive
        self.tgt_d, self.max_d = args.target_density, args.max_density
        self.max_round = config.MAX_ROUND
        self.server = server
        self.client_list = client_list

        self.list_loss, self.list_acc, self.list_est_time, self.list_model_size = [], [], [], []
        self.start_adj_round = None

    def main(self):
        len_pre_rounds = 0
        if self.use_ip:
            print("Starting initial pruning stage...")
            len_pre_rounds = self.server.initial_pruning(self.list_est_time, self.list_loss, self.list_acc,
                                                         self.list_model_size)
            print("Clients loading server model...")
            for client in self.client_list:
                client.load_state_dict(self.server.model.state_dict())
                client.load_mask([layer.mask for layer in self.server.model.prunable_layers])

        print("Starting further pruning stage...")
        start = timer()
        for idx in range(self.max_round):
            list_state_dict, list_num, list_accum_sgrad, list_last_lr = [], [], [], []
            is_adj_round = False
            if idx % self.config.EVAL_DISP_INTERVAL == 0:
                is_adj_round = self.check_adj_round(len_pre_rounds, idx)

            for client in self.client_list:
                sd, npc, grad, last_lr = client.main(is_adj_round)
                list_state_dict.append(sd)
                list_num.append(npc)
                list_accum_sgrad.append(grad)
                list_last_lr.append(last_lr)
            last_lr = list_last_lr[0]
            for client_lr in list_last_lr[1:]:
                assert client_lr == last_lr

            density_limit = None
            if self.max_d is not None:
                density_limit = self.max_d
            if self.tgt_d is not None:
                assert self.tgt_d <= self.max_d
                density_limit += (self.tgt_d - self.max_d) / self.max_round * idx

            list_mask, new_list_sd = self.server.main(idx, list_state_dict, list_num, last_lr, list_accum_sgrad, start,
                                                      self.list_loss, self.list_acc, self.list_est_time,
                                                      self.list_model_size, is_adj_round, density_limit)
            for client, new_sd in zip(self.client_list, new_list_sd):
                client.load_state_dict(new_sd)
                client.load_mask(list_mask)

    def check_adj_round(self, pre_rounds, idx):
        if not self.use_adaptive or len(self.list_acc) == 0:
            return False
        if len(self.list_acc) * self.config.EVAL_DISP_INTERVAL \
                < pre_rounds * self.config.EVAL_DISP_INTERVAL + self.config.ADJ_INTERVAL:
            return False
        elif self.start_adj_round is None:
            self.start_adj_round = idx
            print("Starting reconfiguration at round {}.".format(idx))
            return True
        else:
            return (idx - self.start_adj_round) % self.config.ADJ_INTERVAL == 0
