import argparse
import os
from copy import deepcopy
import torch
from utils.save_load import mkdir_save, load
from utils.functional import disp_num_params
from timeit import default_timer as timer

from abc import ABC, abstractmethod


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode',
                        help="r = reinit; rr = random reinit",
                        action='store',
                        dest='mode',
                        type=str,
                        default="r",
                        required=False)
    parser.add_argument('-t', '--targeted',
                        help="If use targeted final model",
                        action='store_true',
                        dest='targeted',
                        default=False,
                        required=False)
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


class ReinitServer(ABC):
    def __init__(self, args, config, model, save_interval=50):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.experiment_name = args.experiment_name
        self.save_path = os.path.join("results", config.EXP_NAME, args.experiment_name)
        self.save_interval = save_interval
        self.mode = args.mode
        assert self.mode in ["r", "rr"]

        self.model = model.to(self.device)
        self.adaptive_folder = "adaptive{}{}".format("_target" if args.targeted else "",
                                                     "_cs" if args.client_selection else "")
        init_model_path = os.path.join("results", config.EXP_NAME, self.adaptive_folder, "init_model.pt")
        final_model_path = os.path.join("results", config.EXP_NAME, self.adaptive_folder, "model.pt")
        final_model = load(final_model_path)

        # reinit
        if self.mode == "r":
            self.model = load(init_model_path).to(self.device)
            self.model.reinit_from_model(final_model)

        # random reinit, using different seed for initialization but same mask
        elif self.mode == "rr":
            for layer, final_layer in zip(self.model.prunable_layers, final_model.prunable_layers):
                layer.mask = final_layer.mask.clone().to(layer.mask.device)
        else:
            raise ValueError("Mode {} not supported".format(self.mode))

        with torch.no_grad():
            for layer in self.model.prunable_layers:
                layer.weight.mul_(layer.mask)

        disp_num_params(self.model)

        self.model.train()
        mkdir_save(self.model, os.path.join(self.save_path, "init_model.pt"))

        self.test_loader = None

        self.init_test_loader()
        self.init_clients()

    @abstractmethod
    def init_test_loader(self):
        pass

    @abstractmethod
    def init_clients(self):
        pass

    def main(self, idx, list_sd, list_num_proc, lr, start, list_loss, list_acc, list_est_time, list_model_size):
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


class ReinitClient(ABC):
    def __init__(self, config, model):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = deepcopy(model).to(self.device)

        self.optimizer = None
        self.optimizer_scheduler = None
        self.optimizer_wrapper = None
        self.train_loader = None

    @abstractmethod
    def init_optimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_train_loader(self, *args, **kwargs):
        pass

    def main(self):
        self.model.train()
        num_proc_data = 0
        lr = self.optimizer_wrapper.get_last_lr()
        for _ in range(self.config.NUM_LOCAL_UPDATES):
            inputs, labels = self.train_loader.get_next_batch()
            self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))

            num_proc_data += len(inputs)

        self.optimizer_wrapper.lr_scheduler_step()

        return self.model.state_dict(), num_proc_data, lr

    @torch.no_grad()
    def load_mask(self, masks):
        for layer, new_mask in zip(self.model.prunable_layers, masks):
            layer.mask = new_mask.clone().to(layer.mask.device)
            layer.weight.mul_(layer.mask)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class ReinitFL:
    def __init__(self, config, server, client_list):
        self.max_round = config.MAX_ROUND
        self.server = server
        self.client_list = client_list

        self.list_loss, self.list_acc, self.list_est_time, self.list_model_size = [], [], [], []

    def main(self):
        start = timer()
        for idx in range(self.max_round):
            list_state_dict, list_num, list_last_lr = [], [], []

            for client in self.client_list:
                sd, npc, last_lr = client.main()
                list_state_dict.append(sd)
                list_num.append(npc)
                list_last_lr.append(last_lr)
            last_lr = list_last_lr[0]
            for client_lr in list_last_lr[1:]:
                assert client_lr == last_lr

            list_mask, new_list_sd = self.server.main(idx, list_state_dict, list_num, last_lr, start, self.list_loss,
                                                      self.list_acc, self.list_est_time, self.list_model_size)
            for client, new_sd in zip(self.client_list, new_list_sd):
                client.load_state_dict(new_sd)
                client.load_mask(list_mask)
