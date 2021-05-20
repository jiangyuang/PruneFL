import argparse
import torch
import configs.femnist as config
import configs.network as network_config
from bases.nn.models.leaf import Conv2
from bases.vision.load import get_data_loader
from bases.optim.optimizer import SGD
from bases.fl.modules import Server
from bases.optim.optimizer_wrapper import OptimizerWrapper
from control.algorithm import ControlModule
from control.utils import ControlScheduler
from utils.functional import disp_num_params

from timeit import default_timer as timer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed',
                        help="The seed to use for both server and clients.",
                        action='store',
                        dest='seed',
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument('-r', '--resume',
                        help="Resume previous prototype",
                        action='store_true',
                        dest='resume',
                        default=False,
                        required=False)

    mutex = parser.add_mutually_exclusive_group(required=True)
    mutex.add_argument('-a', '--adaptive',
                       help="Use adaptive pruning",
                       action='store_true',
                       dest='use_adaptive')
    mutex.add_argument('-na', '--no-adaptive',
                       help="Do not use adaptive pruning",
                       action='store_false',
                       dest='use_adaptive')

    return parser.parse_args()


class FEMNISTServer(Server):
    def get_init_extra_params(self):
        self.check_client_to_sparse()
        return [([i for i in range(19 * j, 19 * (j + 1) if j != 9 else 193)], self.client_is_sparse) for j in range(10)]


if __name__ == "__main__":
    args = parse_args()
    seed, resume, use_adaptive = args.seed, args.resume, args.use_adaptive
    torch.manual_seed(seed)
    model = Conv2()
    test_loader = get_data_loader(config.EXP_NAME, data_type="test", num_workers=8, pin_memory=True)
    if use_adaptive:
        control_module = ControlModule(model, config=config)
        control_scheduler = ControlScheduler(init_max_dec_diff=config.MAX_DEC_DIFF,
                                             dec_half_life=config.ADJ_HALF_LIFE)
    else:
        control_module = None
        control_scheduler = None

    # Initial pruning
    ip_start = timer()
    print("Starting initial pruning")
    server_pruning_rounds = 1000
    server_adjust_interval = 50
    num_pre_batch = 10
    threshold = 0.1
    print("\tRounds = {}, num data = {}, interval = {}".format(server_pruning_rounds,
                                                               num_pre_batch * config.CLIENT_BATCH_SIZE,
                                                               server_adjust_interval))
    server_loader = get_data_loader(config.EXP_NAME, data_type="train", batch_size=config.CLIENT_BATCH_SIZE,
                                    shuffle=True, num_workers=8, user_list=[0], pin_memory=True)
    server_inputs, server_outputs = [], []
    for _ in range(num_pre_batch):
        inp, out = server_loader.get_next_batch()
        server_inputs.append(inp)
        server_outputs.append(out)

    server_optimizer = SGD(model.parameters(), lr=config.INIT_LR)
    server_optimizer_wrapper = OptimizerWrapper(model, server_optimizer)
    server_control = ControlModule(model=model, config=config)

    prev_density, prev_num, prev_ind = None, 5, []
    for server_i in range(1, server_pruning_rounds + 1):
        for server_inp, server_out in zip(server_inputs, server_outputs):
            list_grad = server_optimizer_wrapper.step(server_inp, server_out)
            for (key, param), g in zip(model.named_parameters(), list_grad):
                assert param.size() == g.size()
                server_control.accumulate(key, g ** 2)

        if server_i % server_adjust_interval == 0:
            server_control.adjust(config.MAX_DEC_DIFF)
            cur_density = disp_num_params(model)

            if prev_density is not None:
                prev_ind.append(abs(cur_density / prev_density - 1) <= threshold)
            prev_density = cur_density

            if len(prev_ind) >= prev_num and all(prev_ind[-prev_num:]):
                ip_end = timer()
                print("Early-stopping initial pruning at round {}. "
                      "Elapsed time = {}.".format(server_i, ip_end - ip_start))
                break
    else:
        ip_end = timer()
        print("Completed initial pruning. Elapsed time = {}.".format(ip_end - ip_start))

    server = FEMNISTServer(config, network_config, model, test_loader, seed, SGD, {"lr": config.INIT_LR}, use_adaptive,
                           use_evaluate=True, control=control_module, control_scheduler=control_scheduler,
                           resume=resume, init_time_offset=-(ip_end - ip_start))

    while True:
        terminate = server.main()
        if terminate:
            break
