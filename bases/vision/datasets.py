import os
from shutil import move
import fnmatch
import json
import numpy as np
import torch
from torchvision.datasets import VisionDataset, ImageFolder
from torchvision.datasets.folder import default_loader
from PIL import Image
import warnings

from configs.femnist import IMG_DIM


class FEMNIST(VisionDataset):
    """
    classes: 10 digits, 26 lower cases, 26 upper cases.
    We use torch.save, torch.load in this dataset
    """

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, user_list: list = None):
        super(FEMNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        """
        0 <= any user in user_list < total_users
        """
        self.train = train
        self.user_list = user_list

        if download:
            self.download()

        if not self._check_exists():
            raise FileNotFoundError("Dataset not found. You can use download=True to download it")

        self.total_num_users = torch.load(os.path.join(self.processed_folder, "num_users.pt"))

        if self.user_list is not None:
            self.num_users = len(self.user_list)
        else:
            self.user_list = [i for i in range(self.total_num_users)]
            self.num_users = self.total_num_users

        if self.train:
            self.data, self.targets = self.load(train=True)
        else:
            self.data, self.targets = self.load(train=False)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # Needs 0~255, uint8 scale
        img = Image.fromarray(np.uint8(255 * (1 - img.numpy())), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    @property
    def all_data_folder(self):
        return os.path.join(self.root, "femnist", "data", "raw_data")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.raw_folder, "train")) and
                os.path.exists(os.path.join(self.raw_folder, "test")) and
                os.path.exists(os.path.join(self.processed_folder, "num_users.pt")))

    def download(self):
        if self._check_exists():
            print("Data already downloaded.")
            return

        if os.path.isdir(self.raw_folder) and len(os.listdir(self.raw_folder)) != 0:
            self.process()
            return

        root = self.root
        if not os.path.isdir(root):
            os.mkdir(root)

        if not os.path.exists(self.all_data_folder):
            # download from https://github.com/TalwalkarLab/leaf/tree/master/data/femnist
            input_str = input("Downloading and processing data will take "
                              "approximately 10 to 30 minutes, and it consumes about 15GB of storage. Continue? [y/n]")
            if input_str.lower() in ["y", "yes"]:
                os.system(rf"git clone https://github.com/TalwalkarLab/leaf.git {root}/github_repo"
                          rf"&& cd {root}/github_repo/data/femnist"
                          r"&& ./preprocess.sh -s niid --sf 0.05 -k 0 -t sample"
                          r"&& cd ../../.."
                          r"&& mv github_repo/data/utils utils"
                          r"&& mv github_repo/data/femnist femnist"
                          r"&& rm -rf github_repo")
                os.makedirs(self.raw_folder, exist_ok=False)
                os.system(rf"cd {root}"
                          r"&& rm -r femnist/data/rem_user_data femnist/data/sampled_data"
                          r"&& mv femnist/data/test raw/ && mv femnist/data/train raw/")
            else:
                print("Exiting...")
                exit()
        else:
            if os.path.exists(os.path.join(root, "data", "rem_user_data")):
                os.system(rf"rm -r {root}/data/rem_user_data")
            if os.path.exists(os.path.join(root, "data", "sampled_data")):
                os.system(rf"rm -r {root}/data/sampled_data")
            if os.path.exists(os.path.join(root, "data", "train")):
                os.system(rf"rm -r {root}/data/train")
            if os.path.exists(os.path.join(root, "data", "test")):
                os.system(rf"rm -r {root}/data/test")
            if os.path.exists(os.path.join(root, "raw")):
                os.system(rf"rm -r {root}/raw")
            if os.path.exists(os.path.join(root, "processed")):
                os.system(rf"rm -r {root}/processed")

            os.makedirs(self.raw_folder, exist_ok=False)
            os.system(rf"cd {root}/femnist"
                      r"&& ./preprocess.sh -s niid --sf 0.05 -k 0 -t sample"
                      r"&& cd .."
                      r"&& rm -r femnist/data/rem_user_data femnist/data/sampled_data"
                      r"&& mv femnist/data/test raw/ && mv femnist/data/train raw/")

        self.process()

    def process(self):
        print("Processing data...")

        if not os.path.isdir(self.processed_folder):
            os.makedirs(self.processed_folder)

        total_users_train = 0
        list_train_f = [f for f in os.listdir(os.path.join(self.raw_folder, "train")) if
                        fnmatch.fnmatch(f, "*.json")]
        list_train_f.sort(key=lambda fname: int(fname[9:-28]))

        for filename in list_train_f:
            with open(os.path.join(self.raw_folder, "train", filename)) as file:
                data = json.load(file)
                for user_name, val in data["user_data"].items():
                    # key: user name
                    # val: dict {x: x_data, y: y_data}
                    x = torch.tensor(val["x"]).reshape((-1, *IMG_DIM))
                    y = torch.tensor(val["y"])

                    torch.save((x, y), os.path.join(self.processed_folder, "train_{}.pt".format(total_users_train)))
                    total_users_train += 1

        total_users_test = 0
        list_test_f = [f for f in os.listdir(os.path.join(self.raw_folder, "test")) if fnmatch.fnmatch(f, "*.json")]
        list_test_f.sort(key=lambda fname: int(fname[9:-27]))

        for filename in list_test_f:
            with open(os.path.join(self.raw_folder, "test", filename)) as file:
                data = json.load(file)
                for user_name, val in data["user_data"].items():
                    # key: user name
                    # val: dict {x: x_data, y: y_data}
                    x = torch.tensor(val["x"]).reshape((-1, *IMG_DIM))
                    y = torch.tensor(val["y"])

                    torch.save((x, y), os.path.join(self.processed_folder, "test_{}.pt").format(total_users_test))
                    total_users_test += 1

        assert total_users_train == total_users_test
        torch.save(total_users_train, os.path.join(self.processed_folder, "num_users.pt"))
        print("Done. {} users processed.".format(total_users_train))

    def load(self, train):
        if train:
            prf = "train"
        else:
            prf = "test"

        data_list, label_list = [], []
        for user_id in self.user_list:
            x, y = torch.load(os.path.join(self.processed_folder, "{}_{}.pt".format(prf, user_id)))
            data_list.append(x)
            label_list.append(y)
        return torch.cat(data_list, dim=0), torch.cat(label_list, dim=0)


class CelebA(VisionDataset):
    """
    The Leaf CelebA dataset. See "https://github.com/TalwalkarLab/leaf/tree/master/data/celeba" for details.
    We use torch.save, torch.load in this dataset.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, user_list: list = None):
        super(CelebA, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.user_list = user_list
        self.loader = default_loader

        if download:
            self.download()

        if not self._check_exists():
            raise FileNotFoundError("Dataset not found. You can use download=True to download it")

        self.total_num_users = torch.load(os.path.join(self.processed_folder, "num_users.pt"))

        if self.user_list is not None:
            self.num_users = len(self.user_list)
        else:
            self.user_list = list(range(self.total_num_users))
            self.num_users = self.total_num_users

        if self.train:
            self.img_paths, self.labels = self.load(train=True)
        else:
            self.img_paths, self.labels = self.load(train=False)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.img_paths[index], self.labels[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.img_paths)

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")

    @property
    def leaf_github_folder(self):
        return os.path.join(self.root, "github_repo")

    def _check_exists(self):
        return (os.path.isdir(os.path.join(self.raw_folder, "img_align_celeba")) and
                os.path.isfile(os.path.join(self.processed_folder, "num_users.pt")) and
                os.path.isfile(os.path.join(self.processed_folder, "train_meta.pt")) and
                os.path.isfile(os.path.join(self.processed_folder, "test_meta.pt"))
                )

    def download(self):
        if self._check_exists():
            print("Data already downloaded and processed.")
            return

        if os.path.isdir(self.raw_folder) and len(os.listdir(self.raw_folder)) != 0:
            self.process()
            return
        elif not os.path.isdir(self.raw_folder):
            os.makedirs(self.raw_folder, exist_ok=False)

        print("Please download the required files to the \"raw\" subdirectory following the instructions from "
              r"https://github.com/TalwalkarLab/leaf/tree/master/data/celeba. Exiting...")
        exit()

    def process(self):
        print("Processing data...")

        if not os.path.exists(os.path.join(self.raw_folder, "img_align_celeba")):
            pass

        if not os.path.isdir(self.processed_folder):
            os.makedirs(self.processed_folder)

        root = self.root
        assert not os.path.isdir(os.path.join(self.processed_folder, "train"))
        assert not os.path.isdir(os.path.join(self.processed_folder, "test"))
        if not os.path.exists(self.leaf_github_folder):
            os.system(rf"git clone https://github.com/TalwalkarLab/leaf.git "
                      rf"{root}/github_repo")
        os.makedirs(f"{root}/github_repo/data/celeba/data/raw", exist_ok=False)
        os.system(rf"mv {root}/raw/img_align_celeba {root}/github_repo/data/celeba/data/raw"
                  rf"&& mv {root}/raw/identity_CelebA.txt {root}/github_repo/data/celeba/data/raw"
                  rf"&& mv {root}/raw/list_attr_celeba.txt {root}/github_repo/data/celeba/data/raw"
                  rf"&& cd {root}/github_repo/data/celeba"
                  r"&& ./preprocess.sh -s niid --sf 1. -k 0 -t sample"
                  r"&& cd ../../.."
                  r"&& mv github_repo/data/celeba/data/train processed/"
                  r"&& mv github_repo/data/celeba/data/test processed/"
                  r"&& mv github_repo/data/celeba/data/raw/img_align_celeba raw/"
                  r"&& mv github_repo/data/celeba/data/raw/identity_CelebA.txt raw/"
                  r"&& mv github_repo/data/celeba/data/raw/list_attr_celeba.txt raw/"
                  r"&& rm -rf github_repo")

        # train data
        total_users_train = 0
        list_train_f = [f for f in os.listdir(os.path.join(self.processed_folder, "train")) if
                        fnmatch.fnmatch(f, "*.json")]
        assert len(list_train_f) == 1
        filename = list_train_f[0]

        renamed_users_train = []
        with open(os.path.join(self.processed_folder, "train", filename)) as file:
            data = json.load(file)
            sorted_user_name = sorted(data["user_data"].keys())
            for user_name in sorted_user_name:
                val = data["user_data"][user_name]
                renamed_users_train.append(val)

                total_users_train += 1

        # test data
        total_users_test = 0
        list_test_f = [f for f in os.listdir(os.path.join(self.processed_folder, "test")) if
                       fnmatch.fnmatch(f, "*.json")]
        assert len(list_test_f) == 1
        filename = list_test_f[0]

        renamed_users_test = []
        with open(os.path.join(self.processed_folder, "test", filename)) as file:
            data = json.load(file)
            sorted_user_name = sorted(data["user_data"].keys())
            for user_name in sorted_user_name:
                val = data["user_data"][user_name]
                renamed_users_test.append(val)

                total_users_test += 1

        assert total_users_train == total_users_test
        torch.save(total_users_train, os.path.join(self.processed_folder, "num_users.pt"))
        torch.save(renamed_users_train, os.path.join(self.processed_folder, "train_meta.pt"))
        torch.save(renamed_users_test, os.path.join(self.processed_folder, "test_meta.pt"))
        print("Done. {} users processed.".format(total_users_train))

    def load(self, train):
        if train:
            prf = "train"
        else:
            prf = "test"

        meta_data = torch.load(os.path.join(self.processed_folder, "{}_meta.pt".format(prf)))

        path_list, label_list = [], []
        for user_id in self.user_list:
            x, y = meta_data[user_id]["x"], meta_data[user_id]["y"]
            path_list.extend([os.path.join(self.raw_folder, "img_align_celeba", p) for p in x])
            label_list.extend(y)
        return path_list, label_list


class ImageNet100(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.parent_dir = root
        if download:
            self.download()

        if not self._check_exists():
            raise FileNotFoundError("Dataset does not exist.")

        data_folder = self.train_folder if train else self.val_folder
        super(ImageNet100, self).__init__(data_folder, transform, target_transform)

    def download(self):
        if self._check_exists():
            print("Data already downloaded and processed.")
            return

        if not os.path.isdir(self.root):
            os.mkdir(self.root)

        raise NotImplementedError("Download not supported")

    def _check_exists(self):
        return os.path.exists(self.train_folder) and os.path.exists(self.val_folder)

    @property
    def train_folder(self):
        return os.path.join(self.parent_dir, "train")

    @property
    def val_folder(self):
        return os.path.join(self.parent_dir, "val")


class TinyImageNet(ImageFolder):
    """
    200 classes. Each class has 500 training images, 50 validation images, and 50 test images
    We use torch.save, torch.load in this dataset
    """

    @property
    def train_folder(self):
        return os.path.join(self.parent_dir, "tiny-imagenet-200", "train")

    @property
    def test_folder(self):
        return os.path.join(self.parent_dir, "tiny-imagenet-200", "test")

    @property
    def val_folder(self):
        return os.path.join(self.parent_dir, "tiny-imagenet-200", "val")

    def __init__(self, root, data_type, transform=None, target_transform=None, download=False):
        assert data_type in ["train", "test", "val"]
        self.parent_dir = root
        if download:
            self.download()

        if not self._check_exists():
            raise FileNotFoundError("Dataset not found. You can use download=True to download it")

        if data_type == "train":
            path_to_data = self.train_folder
        elif data_type == "test":
            path_to_data = self.test_folder
            warnings.warn("No labels for test data")
        else:
            path_to_data = self.val_folder

        super(TinyImageNet, self).__init__(path_to_data, transform=transform, target_transform=target_transform)

    def _check_exists(self):
        return (os.path.exists(self.train_folder) and
                os.path.exists(self.test_folder) and
                not os.path.exists(os.path.join(self.val_folder, "images")))

    def download(self):
        if self._check_exists():
            print("Data already downloaded.")
            return

        if not os.path.isdir(self.root):
            os.mkdir(self.root)

        os.system(rf"cd {self.root}"
                  r"&& wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip"
                  r"&& unzip tiny-imagenet-200.zip")

        self.process()

    def process(self):
        print("Processing...")
        file_to_class = {}
        with open(os.path.join(self.val_folder, "val_annotations.txt"), 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                file_name, class_name = split_line[0], split_line[1]
                file_to_class[file_name] = class_name
                dir_path = os.path.join(self.val_folder, class_name)
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)

        image_folder_path = os.path.join(self.val_folder, "images")
        all_imgs = os.listdir(image_folder_path)

        for img_name in all_imgs:
            img_class = file_to_class[img_name]
            move(os.path.join(image_folder_path, img_name), os.path.join(self.val_folder, img_class, img_name))

        os.rmdir(image_folder_path)
        print("Done.")
