import os
import torch
from bases.fl.simulation.adaptive import AdaptiveServer, AdaptiveClient, AdaptiveFL, parse_args
from bases.optim.optimizer import SGD
from torch.optim import lr_scheduler
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from control.algorithm import ControlModule

from bases.nn.models.resnet import resnet18
from configs.imagenet100 import *
import configs.imagenet100 as config

from utils.save_load import mkdir_save
from bases.vision.sampler import FLSampler


class INAdaptiveServer(AdaptiveServer):
    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="val", batch_size=200, num_workers=8, pin_memory=True)

    def init_clients(self):
        rand_perm = torch.randperm(NUM_TRAIN_DATA).tolist()
        indices = []
        len_slice = NUM_TRAIN_DATA // num_slices

        for i in range(num_slices):
            indices.append(rand_perm[i * len_slice: (i + 1) * len_slice])

        models = [self.model for _ in range(NUM_CLIENTS)]
        self.indices = indices
        return models, indices

    def init_control(self):
        self.control = ControlModule(self.model, config=config)

    def init_ip_config(self):
        self.ip_train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE,
                                               subset_indices=self.indices[0][:IP_DATA_BATCH * CLIENT_BATCH_SIZE],
                                               shuffle=True, num_workers=8, pin_memory=True)

        self.ip_test_loader = get_data_loader(EXP_NAME, data_type="val", batch_size=200, num_workers=8,
                                              pin_memory=True)

        ip_optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.ip_optimizer_wrapper = OptimizerWrapper(self.model, ip_optimizer)
        self.ip_control = ControlModule(model=self.model, config=config)

    def save_exp_config(self):
        exp_config = {"exp_name": EXP_NAME, "seed": args.seed, "batch_size": CLIENT_BATCH_SIZE,
                      "num_local_updates": NUM_LOCAL_UPDATES, "mdd": MAX_DEC_DIFF, "init_lr": INIT_LR,
                      "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY, "lrhl": LR_HALF_LIFE, "step_size": STEP_SIZE,
                      "ahl": ADJ_HALF_LIFE, "use_adaptive": self.use_adaptive,
                      "client_selection": args.client_selection}
        if args.client_selection:
            exp_config["num_users"] = num_users
        mkdir_save(exp_config, os.path.join(self.save_path, "exp_config.pt"))


class INAdaptiveClient(AdaptiveClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        self.optimizer_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=STEP_SIZE,
                                                       gamma=0.5 ** (STEP_SIZE / LR_HALF_LIFE))
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, self.optimizer_scheduler)

    def init_train_loader(self, tl):
        self.train_loader = tl


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    
    num_users = 100
    num_slices = num_users if args.client_selection else NUM_CLIENTS

    server = INAdaptiveServer(args, config, resnet18(num_classes=100))
    list_models, list_indices = server.init_clients()

    sampler = FLSampler(list_indices, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                        num_slices)
    print("Sampler initialized")

    train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                   sampler=sampler, num_workers=8, pin_memory=True)

    client_list = [INAdaptiveClient(list_models[idx], config, args.use_adaptive) for idx in range(NUM_CLIENTS)]
    for client in client_list:
        client.init_optimizer()
        client.init_train_loader(train_loader)

    print("All initialized. Experiment is {}. Use adaptive = {}. Use initial pruning = {}. Client selection = {}. "
          "Num users = {}. Seed = {}. Max round = {}. "
          "Target density = {}".format(EXP_NAME, args.use_adaptive, args.initial_pruning, args.client_selection,
                                       num_users, args.seed, MAX_ROUND, args.target_density))

    fl_runner = AdaptiveFL(args, config, server, client_list)
    fl_runner.main()
