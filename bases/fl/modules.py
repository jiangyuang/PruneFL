import os
from copy import deepcopy
from typing import Union, Type, List
from threading import Thread

import torch
from bases.fl.sockets import ServerSocket
from utils.save_load import mkdir_save, load
from bases.fl.messages import ServerToClientUpdateMessage, ServerToClientInitMessage
from abc import ABC, abstractmethod
from timeit import default_timer as timer

from bases.fl.sockets import ClientSocket
from bases.fl.messages import ClientToServerUpdateMessage
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.nn.linear import DenseLinear, SparseLinear
from utils.functional import copy_dict

__all__ = ["Server", "Client"]


def eval_model_async(eval_func, loader, list_loss, list_acc):
    loss, acc = eval_func(loader)
    list_loss.append(loss)
    list_acc.append(acc)


class ExpConfig:
    def __init__(self, exp_name: str, save_dir_name: str, seed: int, batch_size: int, num_local_updates: int,
                 optimizer_class: Type, optimizer_params: dict, lr_scheduler_class: Union[Type, None],
                 lr_scheduler_params: Union[dict, None], use_adaptive: bool):
        self.exp_name = exp_name
        self.save_dir_name = save_dir_name
        self.seed = seed
        self.batch_size = batch_size
        self.num_local_updates = num_local_updates
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_params = lr_scheduler_params
        self.use_adaptive = use_adaptive


class Server(ABC):
    def __init__(self, config, network_config, model, test_loader, seed, optimizer_class: Type, optimizer_params: dict,
                 use_adaptive, use_evaluate=True, lr_scheduler_class=None, lr_scheduler_params=None, control=None,
                 control_scheduler=None, resume=False, init_time_offset=0.):
        self.config = config
        self.use_adaptive = use_adaptive
        self.use_evaluate = use_evaluate
        self.max_round = self.config.MAX_ROUND_ADAPTIVE if use_adaptive else self.config.MAX_ROUND_CONVENTIONAL_FL
        if use_adaptive:
            assert control is not None
            assert control_scheduler is not None
            self.control = control
            self.control_scheduler = control_scheduler
        self.model = model
        self.test_loader = test_loader
        self.socket = ServerSocket(network_config.SERVER_ADDR, network_config.SERVER_PORT, config.NUM_CLIENTS)
        save_dir_name = self.get_save_dir_name()
        self.save_path = os.path.join("results", "exp_{}".format(config.EXP_NAME), save_dir_name, "server")
        # prototype config that clients need
        exp_config = ExpConfig(self.config.EXP_NAME, save_dir_name, seed, self.config.CLIENT_BATCH_SIZE,
                               self.config.NUM_LOCAL_UPDATES, optimizer_class, optimizer_params, lr_scheduler_class,
                               lr_scheduler_params, use_adaptive)

        self.list_loss = None
        self.list_acc = None
        self.list_time_stamp = None
        self.list_model_size = None
        self.start_time = None
        self.init_time_offset = init_time_offset
        self.round = None
        self.eval_thread = None

        self.client_is_sparse = False

        self.terminate = False

        self.initialize(exp_config, resume)

    @abstractmethod
    def get_init_extra_params(self) -> List[tuple]:
        pass

    def initialize(self, exp_config, resume):
        list_extra_params = self.get_init_extra_params()

        self.socket.wait_for_connections()

        if resume:
            print("Resuming server...")
            self.list_loss = load(os.path.join(self.save_path, "loss.pt"))
            self.list_acc = load(os.path.join(self.save_path, "accuracy.pt"))
            self.list_time_stamp = load(os.path.join(self.save_path, "time.pt"))
            self.list_model_size = load(os.path.join(self.save_path, "model_size.pt"))

            self.model = load(os.path.join(self.save_path, "model.pt"))

            num_loss_acc = len(self.list_loss)
            assert len(self.list_acc) == num_loss_acc

            num_evals = len(self.list_time_stamp)
            assert len(self.list_model_size) == num_evals

            if num_evals - num_loss_acc == 1:
                loss, acc = self.model.evaluate(self.test_loader)
                self.list_loss.append(loss)
                self.list_acc.append(acc)
            elif num_evals != num_loss_acc:
                raise RuntimeError("Cannot resume")

            self.round = (num_evals - 1) * self.config.EVAL_DISP_INTERVAL
            assert self.round >= 0
            self.start_time = timer() - self.list_time_stamp[-1]

            self.check_client_to_sparse()
            resume_param = (True, self.round + 1, self.client_is_sparse)
            list_params = [(idx, exp_config, self.model, list_extra_params[idx], resume_param) for idx in
                           range(self.config.NUM_CLIENTS)]
            resume_msgs_to_client = [ServerToClientInitMessage(init_params) for init_params in list_params]
            self.socket.init_connections(resume_msgs_to_client)

            self.round += 1

            print("Server resumed")
            print(self)

        else:
            self.list_loss = []
            self.list_acc = []
            self.list_time_stamp = []
            self.list_model_size = []
            self.start_time = timer() + self.init_time_offset
            self.round = 0
            mkdir_save(self.model, os.path.join(self.save_path, "init_model.pt"))
            self.model.eval()

            list_init_params = [(idx, exp_config, self.model, list_extra_params[idx], (False, 0, False)) for idx in
                                range(self.config.NUM_CLIENTS)]
            init_msgs_to_client = [ServerToClientInitMessage(init_params) for init_params in list_init_params]
            self.socket.init_connections(init_msgs_to_client)

            print("Server initialized")
            print(self)

    def get_save_dir_name(self):
        if not self.use_adaptive:
            return "conventional"
        else:
            mdd_100, chl = 100 * self.config.MAX_DEC_DIFF, self.config.ADJ_HALF_LIFE
            lrhl = self.config.LR_HALF_LIFE if hasattr(self.config, "LR_HALF_LIFE") else None
            assert mdd_100 - int(mdd_100) == 0
            return "mdd{}_chl{}_lrhl{}".format(int(mdd_100), lrhl, chl)

    def calc_model_params(self, display=False):
        sum_param_in_use = 0
        sum_all_param = 0
        for layer, layer_prefix in zip(self.model.param_layers, self.model.param_layer_prefixes):
            num_bias = 0 if layer.bias is None else layer.bias.nelement()
            layer_param_in_use = layer.mask.sum().int().item() + num_bias
            layer_all_param = layer.mask.nelement() + num_bias
            sum_param_in_use += layer_param_in_use
            sum_all_param += layer_all_param
            if display:
                print("\t{} remaining: {}/{} = {}".format(layer_prefix, layer_param_in_use, layer_all_param,
                                                          layer_param_in_use / layer_all_param))
        if display:
            print("\tTotal: {}/{} = {}".format(sum_param_in_use, sum_all_param, sum_param_in_use / sum_all_param))

        return sum_param_in_use

    def adjust_model(self, display=True):
        print("Running control algorithm")
        alg_start = timer()
        max_dec_diff = self.control_scheduler.max_dec_diff(self.round)
        self.control.adjust(max_dec_diff, None)
        print("Algorithm completed in {}s".format(timer() - alg_start))
        if display:
            print("New params:")
            self.calc_model_params(display=True)

        self.check_client_to_sparse()

    @torch.no_grad()
    def merge_accumulate_client_update(self, list_num_proc, list_state_dict, lr):
        total_num_proc = sum(list_num_proc)

        # merged_state_dict = dict()
        dict_keys = list_state_dict[0].keys()
        for state_dict in list_state_dict[1:]:
            assert state_dict.keys() == dict_keys

        # accumulate extra sgrad and remove from state_dict
        if self.use_adaptive and self.is_adj_round():
            prefix = "extra."
            for state_dict in list_state_dict:
                del_list = []
                for key, param in state_dict.items():
                    if key[:len(prefix)] == prefix:
                        sgrad_key = key[len(prefix):]
                        mask_0 = self.model.get_mask_by_name(sgrad_key) == 0.
                        dense_sgrad = torch.zeros_like(mask_0, dtype=torch.float)
                        dense_sgrad.masked_scatter_(mask_0, param)

                        # no need to divide by lr
                        self.control.accumulate(sgrad_key, dense_sgrad)
                        del_list.append(key)

                for del_key in del_list:
                    del state_dict[del_key]

        # accumulate sgrad and update server state dict
        server_state_dict = self.model.state_dict()
        for key in dict_keys:
            server_param = server_state_dict[key]
            avg_inc_val = None
            for num_proc, state_dict in zip(list_num_proc, list_state_dict):
                if state_dict[key].size() != server_state_dict[key].size():
                    mask = self.model.get_mask_by_name(key)
                    inc_val = server_param.masked_scatter(mask, state_dict[key]) - server_param
                else:
                    inc_val = state_dict[key] - server_param

                if avg_inc_val is None:
                    avg_inc_val = num_proc / total_num_proc * inc_val
                else:
                    avg_inc_val += num_proc / total_num_proc * inc_val

                # accumulate sgrad from parameters
                if self.use_adaptive and key in dict(self.model.named_parameters()).keys():
                    self.control.accumulate(key, ((inc_val / lr) ** 2) * num_proc)

            server_param.add_(avg_inc_val)

    def check_termination(self) -> bool:
        """
        For extra termination criterion, e.g. max time reached. True if terminate, else False.
        """
        return self.terminate

    def evaluate(self):
        if self.eval_thread is not None:
            self.eval_thread.join()
        t = Thread(target=eval_model_async,
                   args=(deepcopy(self.model).evaluate, self.test_loader, self.list_loss, self.list_acc))
        t.start()
        self.eval_thread = t

        elapsed_time = timer() - self.start_time
        self.list_time_stamp.append(elapsed_time)

        model_size = self.calc_model_params(display=False)
        self.list_model_size.append(model_size)

        len_loss = len(self.list_loss)
        len_acc = len(self.list_acc)
        assert len_loss == len_acc
        print("Evaluation at round #{}. "
              "Loss/acc (at round {}) = {}/{}. "
              "Elapsed time = {}".format(self.round,
                                         (len_acc - 1) * self.config.EVAL_DISP_INTERVAL if len_acc > 0 else "NaN",
                                         self.list_loss[-1] if len_acc > 0 else "NaN",
                                         self.list_acc[-1] if len_acc > 0 else "NaN",
                                         elapsed_time))

        self.save_exp()

    def is_adj_round(self, rd=None) -> bool:
        if rd is None:
            rd = self.round
        return self.use_adaptive and rd > 0 and rd % self.config.ADJ_INTERVAL == 0

    def is_one_before_adj_round(self) -> bool:
        return self.is_adj_round(self.round + 1)

    def check_client_to_sparse(self):
        if not self.client_is_sparse and self.model.density() <= self.config.TO_SPARSE_THR:
            self.client_is_sparse = True

    def clean_dict_to_client(self) -> dict:
        """
        Clean up state dict before processing, e.g. remove entries, transpose.
        To be overridden by subclasses.
        """
        clean_state_dict = copy_dict(self.model.state_dict())  # not deepcopy
        if self.client_is_sparse:
            for layer, prefix in zip(self.model.param_layers, self.model.param_layer_prefixes):
                key = prefix + ".bias"
                if isinstance(layer, DenseLinear) and key in clean_state_dict.keys():
                    clean_state_dict[key] = clean_state_dict[key].view((-1, 1))

        return clean_state_dict

    @torch.no_grad()
    def process_state_dict_to_client(self) -> dict:
        """
        Process state dict before sending to client, e.g. to cpu, to sparse, keep values only.
        if not self.client_is_sparse: send dense
        elif self.is_adj_round(): send full sparse state_dict
        else: send sparse values only
        To be overridden by subclasses.
        """
        clean_state_dict = self.clean_dict_to_client()
        if not self.client_is_sparse:
            return clean_state_dict

        if self.is_adj_round():
            for layer, prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
                # works for both layers
                key_w = prefix + ".weight"
                if key_w in clean_state_dict.keys():
                    weight = clean_state_dict[key_w]
                    w_mask = self.model.get_mask_by_name(key_w)
                    sparse_weight = (weight * w_mask).view(weight.size(0), -1).to_sparse()
                    clean_state_dict[key_w] = sparse_weight

        else:
            for prefix in self.model.prunable_layer_prefixes:
                key_w = prefix + ".weight"
                if key_w in clean_state_dict.keys():
                    clean_state_dict[key_w] = clean_state_dict[key_w].masked_select(self.model.get_mask_by_name(key_w))

        return clean_state_dict

    def save_exp(self):
        mkdir_save(self.list_loss, os.path.join(self.save_path, "loss.pt"))
        mkdir_save(self.list_acc, os.path.join(self.save_path, "accuracy.pt"))
        mkdir_save(self.list_time_stamp, os.path.join(self.save_path, "time.pt"))
        mkdir_save(self.list_model_size, os.path.join(self.save_path, "model_size.pt"))
        mkdir_save(self.model, os.path.join(self.save_path, "model.pt"))

    def main(self):
        assert not self.terminate
        msgs = self.socket.recv_update_msg_from_all()
        list_lr = [msg.lr for msg in msgs]
        list_num_proc = [msg.num_processed for msg in msgs]
        list_state_dict = [msg.state_dict for msg in msgs]

        lr = list_lr[0]
        for client_lr in list_lr[1:]:
            assert client_lr == lr

        self.merge_accumulate_client_update(list_num_proc, list_state_dict, lr)

        if self.use_evaluate and self.round % self.config.EVAL_DISP_INTERVAL == 0:
            self.evaluate()

        if self.is_adj_round():
            self.adjust_model()

        # if self.round % self.config.SAVE_INTERVAL == 0:
        #     self.save_exp()

        terminate = self.check_termination()
        if self.round >= self.max_round - 1:
            terminate = True

        state_dict_to_client = self.process_state_dict_to_client()
        client_adj = self.is_one_before_adj_round()
        to_sparse = self.client_is_sparse
        msg_to_clients = ServerToClientUpdateMessage((state_dict_to_client, client_adj, to_sparse, terminate))
        self.socket.send_msg_to_all(msg_to_clients)

        self.round += 1

        if terminate:
            self.socket.recv_ack_msg_from_all()
            self.socket.close()
            self.eval_thread.join()
            self.save_exp()
            self.terminate = True
            print("Task completed")

        return terminate

    def __repr__(self):
        return "Experiment = {}".format(self.config.EXP_NAME)


class Client(ABC):
    def __init__(self, network_config, max_try=100):
        self.network_config = network_config
        self.socket = ClientSocket(network_config.SERVER_ADDR, network_config.SERVER_PORT)
        self.train_loader = None

        init_msg = self.socket.init_connections(max_try)
        self.client_id = init_msg.client_id

        self.exp_config = init_msg.exp_config

        torch.manual_seed(self.exp_config.seed)

        # self.save_path = os.path.join("results", "exp_{}".format(self.exp_config.exp_name),
        #                               self.exp_config.save_dir_name, "client_{}".format(self.client_id))

        self.model = init_msg.model
        self.model.train()

        self.optimizer = self.exp_config.optimizer_class(params=self.model.parameters(),
                                                         **self.exp_config.optimizer_params)
        self.lr_scheduler = None
        if self.exp_config.lr_scheduler_class is not None:
            self.lr_scheduler = self.exp_config.lr_scheduler_class(optimizer=self.optimizer,
                                                                   **self.exp_config.lr_scheduler_params)
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, self.lr_scheduler)

        if self.exp_config.use_adaptive:
            self.dict_extra_sgrad = dict()
            self.accum_dense_grad = dict()

        self.is_adj_round = False
        self.is_sparse = False
        self.terminate = False
        self.parse_init_extra_params(init_msg.extra_params)

        resume, cur_round, resume_to_sparse = init_msg.resume_params
        self.initialize(resume, cur_round, resume_to_sparse)

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        param_dict = dict(self.model.named_parameters())
        buffer_dict = dict(self.model.named_buffers())
        for key, param in {**param_dict, **buffer_dict}.items():
            if key in state_dict.keys():
                if state_dict[key].size() != param.size():
                    # sparse param with value only
                    param._values().copy_(state_dict[key])
                elif state_dict[key].is_sparse:
                    # sparse param at adjustment round
                    # print(param, param.size(), state_dict[key].is_sparse, state_dict[key])
                    # param.zero_()
                    param.copy_(state_dict[key])
                    # param._indices().copy_(state_dict[key]._indices())
                    # param._values().copy_(state_dict[key]._values())
                    # need to reload mask in this case
                    param.mask.copy_(state_dict[key].mask)
                else:
                    param.copy_(state_dict[key])

    def initialize(self, resume, cur_round, resume_to_sparse):
        if resume:
            print("Resuming client...")
            # move optimizer to the right position
            for _ in range(cur_round * self.exp_config.num_local_updates):
                self.optimizer_wrapper.lr_scheduler_step()

            # move train loader to the right position
            remaining_batches = cur_round * self.exp_config.num_local_updates
            num_batches_epoch = len(self.train_loader)
            while remaining_batches >= num_batches_epoch:
                self.train_loader.skip_epoch()
                remaining_batches -= num_batches_epoch
            for _ in range(remaining_batches):
                self.train_loader.get_next_batch()

            if resume_to_sparse:
                self.convert_to_sparse()

            print("Client resumed")
        else:
            print("Client initialized")

    @abstractmethod
    def parse_init_extra_params(self, extra_params):
        # Initialize train_loader, etc.
        pass

    def cleanup_state_dict_to_server(self) -> dict:
        """
        Clean up state dict before process, e.g. remove entries, transpose.
        To be overridden by subclasses.
        """
        clean_state_dict = copy_dict(self.model.state_dict())  # not deepcopy
        if self.is_sparse:
            for layer, prefix in zip(self.model.param_layers, self.model.param_layer_prefixes):
                key = prefix + ".bias"
                if isinstance(layer, SparseLinear) and key in clean_state_dict.keys():
                    clean_state_dict[key] = clean_state_dict[key].view(-1)

            del_list = []
            del_suffix = "placeholder"
            for key in clean_state_dict.keys():
                if key.endswith(del_suffix):
                    del_list.append(key)

            for del_key in del_list:
                del clean_state_dict[del_key]

        return clean_state_dict

    @torch.no_grad()
    def process_state_dict_to_server(self) -> dict:
        """
        Process state dict before sending to server, e.g. keep values only, extra param in adjustment round.
        if not self.is_sparse: send dense
        elif self.adjustment_round: send sparse values + extra grad values
        else: send sparse values only
        To be overridden by subclasses.
        """
        clean_state_dict = self.cleanup_state_dict_to_server()

        if self.is_sparse:
            for key, param in clean_state_dict.items():
                if param.is_sparse:
                    clean_state_dict[key] = param._values()

        if self.is_adj_round:
            clean_state_dict.update(self.dict_extra_sgrad)
            self.dict_extra_sgrad = dict()

        return clean_state_dict

    def convert_to_sparse(self):
        self.model = self.model.to_sparse()
        old_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        self.optimizer = self.exp_config.optimizer_class(params=self.model.parameters(),
                                                         **self.exp_config.optimizer_params)
        if self.exp_config.lr_scheduler_class is not None:
            lr_scheduler_state_dict = deepcopy(self.lr_scheduler.state_dict())
            self.lr_scheduler = self.exp_config.lr_scheduler_class(optimizer=self.optimizer,
                                                                   **self.exp_config.lr_scheduler_params)
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
        self.optimizer.param_groups[0]["lr"] = old_lr
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, self.lr_scheduler)

        self.is_sparse = True

        print("Model converted to sparse")

    def accumulate_dense_grad_round(self):
        for key, param in self.model.named_parameters():
            if hasattr(param, "is_sparse_param"):
                if key in self.accum_dense_grad.keys():
                    self.accum_dense_grad[key] += param.dense.grad
                else:
                    self.accum_dense_grad[key] = param.dense.grad

    def accumulate_sgrad(self, num_proc_data):
        prefix = "extra."
        for key, param in self.accum_dense_grad.items():
            pkey = prefix + key
            if pkey in self.dict_extra_sgrad.keys():
                self.dict_extra_sgrad[pkey] += (param ** 2) * num_proc_data
            else:
                self.dict_extra_sgrad[pkey] = (param ** 2) * num_proc_data

            if self.is_adj_round:
                param_mask = dict(self.model.named_parameters())[key].mask == 0.
                self.dict_extra_sgrad[pkey] = self.dict_extra_sgrad[pkey].masked_select(param_mask)

    def main(self):
        num_proc_data = 0
        for _ in range(self.exp_config.num_local_updates):
            inputs, labels = self.train_loader.get_next_batch()
            self.optimizer_wrapper.step(inputs, labels)

            if self.exp_config.use_adaptive:
                self.accumulate_dense_grad_round()

            num_proc_data += len(inputs)

        if self.exp_config.use_adaptive:
            self.accumulate_sgrad(num_proc_data)
            self.accum_dense_grad = dict()

        lr = self.optimizer_wrapper.get_last_lr()

        state_dict_to_server = self.process_state_dict_to_server()
        msg_to_server = ClientToServerUpdateMessage((state_dict_to_server, num_proc_data, lr))
        self.socket.send_msg(msg_to_server)

        update_msg = self.socket.recv_update_msg()
        self.is_adj_round = update_msg.adjustment
        if not self.is_sparse and update_msg.to_sparse:
            self.convert_to_sparse()

        state_dict_received = update_msg.state_dict
        self.load_state_dict(state_dict_received)

        self.optimizer_wrapper.lr_scheduler_step()

        terminate = update_msg.terminate
        if terminate:
            self.socket.send_ack_msg()
            self.socket.close()
            self.terminate = True
            print("Task completed")

        return terminate
