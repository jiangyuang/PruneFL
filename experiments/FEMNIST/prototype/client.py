import configs.network as network_config
from bases.vision.load import get_data_loader
from bases.fl.modules import Client


class FEMNISTClient(Client):
    def parse_init_extra_params(self, extra_params):
        users = extra_params[0]
        self.train_loader = get_data_loader(self.exp_config.exp_name, data_type="train",
                                            batch_size=self.exp_config.batch_size, shuffle=True, num_workers=8,
                                            pin_memory=True, user_list=users)

        to_sparse = extra_params[1]
        if to_sparse:
            self.convert_to_sparse()


if __name__ == "__main__":
    client = FEMNISTClient(network_config)

    while True:
        terminate = client.main()
        if terminate:
            break
