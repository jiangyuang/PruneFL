import os
import torch
from bases.fl.simulation.online import OnlineServer, OnlineClient, OnlineFL, parse_args
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from bases.vision.sampler import FLSampler
from bases.nn.models import Conv4
from configs.celeba import *
import configs.celeba as config


class CelebAOnlineServer(OnlineServer):
    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=8, batch_size=100, shuffle=False,
                                           pin_memory=True)

    def init_clients(self):
        if args.client_selection:
            list_usr = [[i] for i in range(num_users)]
        else:
            nusr = num_users // NUM_CLIENTS  # num users for the first NUM_CLIENTS - 1 clients
            list_usr = [list(range(nusr * j, nusr * (j + 1) if j != NUM_CLIENTS - 1 else num_users)) for j in
                        range(NUM_CLIENTS)]
        models = [self.model for _ in range(NUM_CLIENTS)]
        return models, list_usr


class CelebAOnlineClient(OnlineClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)

    def init_train_loader(self, tl):
        self.train_loader = tl


def get_indices_list():
    train_meta = torch.load(os.path.join("datasets", "CelebA", "processed", "train_meta.pt"))
    cur_pointer = 0
    indices_list = []
    for ul in list_users:
        num_data = 0
        for user_id in ul:
            num_data += len(train_meta[user_id]["x"])
        indices_list.append(list(range(cur_pointer, cur_pointer + num_data)))
        cur_pointer += num_data

    return indices_list


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    
    num_user_path = os.path.join("datasets", "CelebA", "processed", "num_users.pt")
    if not os.path.isfile(num_user_path):
        get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False, num_workers=8,
                        pin_memory=True)
    num_users = torch.load(num_user_path)

    server = CelebAOnlineServer(args, config, Conv4())
    list_models, list_users = server.init_clients()

    sampler = FLSampler(get_indices_list(), MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                        NUM_CLIENTS)
    print("Sampler initialized")

    train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                   sampler=sampler, num_workers=8, pin_memory=True)

    client_list = [CelebAOnlineClient(config, list_models[idx]) for idx in range(NUM_CLIENTS)]
    for client in client_list:
        client.init_optimizer()
        client.init_train_loader(train_loader)

    print("All initialized. Experiment is {}. Max upload = {}. Client selection = {}. Num users = {}. Seed = {}. "
          "Max round = {}.".format(EXP_NAME, MAX_NUM_UPLOAD, args.client_selection, num_users, args.seed, MAX_ROUND))

    fl_runner = OnlineFL(config, server, client_list)
    fl_runner.main()
