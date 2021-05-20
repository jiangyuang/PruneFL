import argparse
import os
import torch
import numpy as np
from bases.fl.simulation.adaptive import AdaptiveServer, AdaptiveClient
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from control.algorithm import ControlModule

from configs.femnist import *
import configs.femnist as config

from utils.save_load import mkdir_save, load
from utils.functional import disp_num_params

from timeit import default_timer as timer


def check_is_adj_round(idx):
    return idx > 0 and idx % CONTROL_INTERVAL == 0


class FEMNISTAdaptiveServer(AdaptiveServer):
    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, test=True, flatten=False, one_hot=True, num_workers=8,
                                           pin_memory=True)

    def init_clients(self):
        list_usr = [[i for i in range(19 * j, 19 * (j + 1) if j != 9 else 193)] for j in range(10)]
        models = [self.model for _ in range(NUM_CLIENTS)]
        return models, list_usr

    def init_control(self):
        self.control = ControlModule(self.model, config=config)

    def save_exp_config(self):
        exp_config = {"exp_name": EXP_NAME, "seed": seed, "batch_size": CLIENT_BATCH_SIZE,
                      "num_local_updates": NUM_LOCAL_UPDATES, "mdd": MAX_DEC_DIFF, "init_lr": INIT_LR,
                      "chl": CONTROL_HALF_LIFE, "use_adaptive": self.use_adaptive, "client_selection": client_selection}
        if client_selection:
            exp_config["num_users"] = num_users
        mkdir_save(exp_config, os.path.join(self.save_path, "exp_config.pt"))

    def main(self, idx, list_sd, list_num_proc, lr, list_accumulated_sgrad, start, list_loss, list_acc, list_est_time,
             list_model_size, is_adj_round):
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
            # keep the same
            max_dec_diff = self.config.MAX_DEC_DIFF * (0.5 ** (MAX_ROUND_ADAPTIVE / self.config.ADJ_HALF_LIFE))
            self.control.adjust(max_dec_diff, density - dec_coeff * idx)
            print("Total alg time = ", timer() - alg_start)
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


class FEMNISTAdaptiveClient(AdaptiveClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)

    def init_train_loader(self, train_loader):
        self.train_loader = train_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-rounds',
                        help="How many extra rounds to run",
                        action='store',
                        dest='num_round',
                        type=int,
                        required=True)
    parser.add_argument('-c', '--client-selection',
                        help="If use client-selection",
                        action='store_true',
                        dest='client_selection',
                        default=False,
                        required=False)
    parser.add_argument('-e', '--exp-name',
                        help="Experiment name",
                        action='store',
                        dest='experiment_name',
                        type=str,
                        required=True)
    parser.add_argument('-s', '--seed',
                        help="The seed to use for the prototype",
                        action='store',
                        dest='seed',
                        type=int,
                        default=0,
                        required=False)

    return parser.parse_args()


def load_existing_model():
    return load(final_model_path)


if __name__ == "__main__":
    args = parse_args()
    experiment_name, client_selection, seed = args.experiment_name, args.client_selection, args.seed
    max_round = args.num_round
    model_save_interval = 1000
    dec_coeff = 0.00001
    save_interval = 50

    adaptive_folder = "adaptive_cs" if client_selection else "adaptive"
    final_model_path = os.path.join("results", EXP_NAME, adaptive_folder, "model.pt")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)  # Not setting separate seeds for clients
    list_loss, list_acc, list_est_time, list_model_size = [], [], [], []
    save_path = os.path.join("results", EXP_NAME, experiment_name)
    density = load("results/{}/{}/model.pt".format(EXP_NAME, adaptive_folder)).density()
    print("Init density = {}".format(density))

    server = FEMNISTAdaptiveServer(dev, experiment_name, config, load_existing_model, save_path, True)
    list_models, list_users = server.init_clients()

    num_users = 193
    client_list = [FEMNISTAdaptiveClient(list_models[idx], dev, config, True) for idx in range(NUM_CLIENTS)]
    for client in client_list:
        client.init_optimizer()

    if client_selection:
        all_train_loaders = [get_data_loader(EXP_NAME, train=True, train_batch_size=CLIENT_BATCH_SIZE, shuffle=True,
                                             flatten=False, one_hot=True, num_workers=8, user_list=[i],
                                             pin_memory=True) for i in range(num_users)]
    else:
        list_train_loaders = [get_data_loader(EXP_NAME, train=True, train_batch_size=CLIENT_BATCH_SIZE, shuffle=True,
                                              flatten=False, one_hot=True, num_workers=8, user_list=users,
                                              pin_memory=True) for users in list_users]
        for client, tl in zip(client_list, list_train_loaders):
            client.init_train_loader(tl)

    print("All initialized. Use adaptive = {}. "
          "Num users = {}. Client selection = {}. "
          "Seed = {}. Max round = {}.".format(True, num_users, client_selection, seed, max_round))

    start = timer()
    for idx in range(max_round):
        if idx > 0 and idx % model_save_interval == 0:
            mkdir_save(server.model, os.path.join(save_path, "model_{}.pt".format(idx)))

        list_state_dict, list_num, list_accum_sgrad, list_last_lr = [], [], [], []
        is_adj_round = check_is_adj_round(idx)

        if client_selection:
            selected_train_loaders = np.random.choice(num_users, NUM_CLIENTS, replace=False).tolist()
            for client, train_loader_id in zip(client_list, selected_train_loaders):
                client.init_train_loader(all_train_loaders[train_loader_id])

        for client in client_list:
            sd, npc, grad, last_lr = client.main(is_adj_round)
            list_state_dict.append(sd)
            list_num.append(npc)
            list_accum_sgrad.append(grad)
            list_last_lr.append(last_lr)
        last_lr = list_last_lr[0]
        for client_lr in list_last_lr[1:]:
            assert client_lr == last_lr
        list_mask, new_list_sd = server.main(idx, list_state_dict, list_num, last_lr, list_accum_sgrad, start, list_loss,
                                             list_acc, list_est_time, list_model_size, is_adj_round)
        for client, new_sd in zip(client_list, new_list_sd):
            client.load_state_dict(new_sd)
            client.load_mask(list_mask)
