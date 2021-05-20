import torch
from bases.fl.simulation.online import OnlineServer, OnlineClient, OnlineFL, parse_args
from bases.optim.optimizer import SGD
from torch.optim import lr_scheduler
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.nn.models.resnet import resnet18
from configs.imagenet100 import *
import configs.imagenet100 as config
from bases.vision.sampler import FLSampler
from bases.vision.load import get_data_loader


class INOnlineServer(OnlineServer):
    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="val", batch_size=200, num_workers=8, pin_memory=True)

    def init_clients(self):
        rand_perm = torch.randperm(NUM_TRAIN_DATA).tolist()
        indices = []
        len_slice = NUM_TRAIN_DATA // num_slices

        for i in range(num_slices):
            indices.append(rand_perm[i * len_slice: (i + 1) * len_slice])

        models = [self.model for _ in range(NUM_CLIENTS)]
        return models, indices


class INOnlineClient(OnlineClient):
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

    server = INOnlineServer(args, config, resnet18(num_classes=100))
    list_models, list_indices = server.init_clients()

    sampler = FLSampler(list_indices, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                        num_slices)
    print("Sampler initialized")

    train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                   sampler=sampler, num_workers=8, pin_memory=True)

    client_list = [INOnlineClient(config, list_models[idx]) for idx in range(NUM_CLIENTS)]
    for client in client_list:
        client.init_optimizer()
        client.init_train_loader(train_loader)

    print("All initialized. Experiment is {}. Max upload = {}. Client selection = {}. Num users = {}. Seed = {}. "
          "Max round = {}.".format(EXP_NAME, MAX_NUM_UPLOAD, args.client_selection, num_users, args.seed, MAX_ROUND))

    fl_runner = OnlineFL(config, server, client_list)
    fl_runner.main()
