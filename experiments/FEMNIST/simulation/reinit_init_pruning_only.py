import os
import torch
import numpy as np
from bases.fl.simulation.reinitialize import ReinitServer, ReinitClient, parse_args
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader

from bases.nn.models.leaf import Conv2
from configs.femnist import *
import configs.femnist as config

from utils.save_load import load

from timeit import default_timer as timer


class FEMNIST10ReinitServer(ReinitServer):
    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, test=True, flatten=False, one_hot=True, num_workers=8,
                                           pin_memory=True)

    def init_clients(self):
        list_usr = [[i for i in range(19 * j, 19 * (j + 1) if j != 9 else 193)] for j in range(10)]
        models = [self.model for _ in range(NUM_CLIENTS)]
        return models, list_usr


class FEMNISTReinitClient(ReinitClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)

    def init_train_loader(self, train_loader):
        self.train_loader = train_loader


if __name__ == "__main__":
    args = parse_args()
    mode, experiment_name = args.mode, args.experiment_name
    client_selection = False
    seed = args.seed
    assert mode in ["r", "rr"]
    mode_name = {"r": "reinit", "rr": "random_reinit"}
    adaptive_folder = "adaptive_init_only"

    init_model_path = os.path.join("results", EXP_NAME, adaptive_folder, "init_model.pt")
    final_model_path = os.path.join("results", EXP_NAME, adaptive_folder, "init_pruning_model.pt")
    save_interval = 50
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if mode == "rr":
        prev_config = load(os.path.join("results", EXP_NAME, adaptive_folder, "exp_config.pt"))
        seed = prev_config["seed"] + 1

    torch.manual_seed(seed)  # Not setting separate seeds for clients
    list_loss, list_acc, list_est_time, list_model_size = [], [], [], []
    save_path = os.path.join("results", EXP_NAME, experiment_name)

    server = FEMNIST10ReinitServer(dev, experiment_name, config, Conv2, save_path, save_interval, mode,
                                   init_model_path, final_model_path)
    list_models, list_users = server.init_clients()

    num_users = 193
    client_list = [FEMNISTReinitClient(list_models[idx], dev, config) for idx in range(NUM_CLIENTS)]
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

    max_round = MAX_ROUND_CONVENTIONAL_FL

    print("All initialized. Mode = {}. Client selection = {}. "
          "Seed = {}. Max round = {}.".format(mode_name[mode], client_selection, seed, max_round))

    start = timer()
    for idx in range(MAX_ROUND_CONVENTIONAL_FL):
        list_state_dict, list_num, list_last_lr = [], [], []
        if client_selection:
            selected_train_loaders = np.random.choice(num_users, NUM_CLIENTS, replace=False).tolist()
            for client, train_loader_id in zip(client_list, selected_train_loaders):
                client.init_train_loader(all_train_loaders[train_loader_id])

        for client in client_list:
            sd, npc, last_lr = client.main()
            list_state_dict.append(sd)
            list_num.append(npc)
            list_last_lr.append(last_lr)
        last_lr = list_last_lr[0]
        for client_lr in list_last_lr[1:]:
            assert client_lr == last_lr

        list_mask, new_list_sd = server.main(idx, list_state_dict, list_num, last_lr, start, list_loss, list_acc,
                                             list_est_time, list_model_size)
        for client, new_sd in zip(client_list, new_list_sd):
            client.load_state_dict(new_sd)
            client.load_mask(list_mask)
