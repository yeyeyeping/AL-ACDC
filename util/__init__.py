import yaml
import os
import torch
import scipy.ndimage as ndimage
import numpy as np
import albumentations as A
from os.path import join
import random
from argparse import ArgumentParser
import time

import logging
from torch.utils.data import Dataset, SubsetRandomSampler


class SubsetSampler(SubsetRandomSampler):
    def __iter__(self):
        return iter(self.indices)


def save_query_plot(folder, labeled_percent, dice_list):
    import matplotlib.pyplot as plt
    with open(f"{folder}/result.txt", "w") as fp:
        fp.write("x:")
        fp.write(str(labeled_percent))
        fp.write("\ny:")
        fp.write(str(dice_list))
    plt.plot(labeled_percent, dice_list)
    plt.savefig(f"{folder}/result.jpg")


def get_samplers(data_num, initial_labeled, with_pseudo=False):
    initial_labeled = int(data_num * initial_labeled)
    data_indice = list(range(data_num))
    np.random.shuffle(data_indice)
    retval = (SubsetSampler(data_indice[:initial_labeled]), SubsetSampler(data_indice[initial_labeled:]))
    if with_pseudo:
        retval = (*retval, SubsetSampler([]))
    return retval


def get_largest_k_components(image, k=1):
    """
    Get the largest K components from 2D or 3D binary image.

    :param image: The input ND array for binary segmentation.
    :param k: (int) The value of k.

    :return: An output array with only the largest K components of the input.
    """
    dim = len(image.shape)
    if (image.sum() == 0):
        print('the largest component is null')
        return image
    if (dim < 2 or dim > 3):
        raise ValueError("the dimension number should be 2 or 3")
    s = ndimage.generate_binary_structure(dim, 1)
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    sizes_sort = sorted(sizes, reverse=True)
    kmin = min(k, numpatches)
    output = np.zeros_like(image)
    for i in range(kmin):
        labeli = np.where(sizes == sizes_sort[i])[0] + 1
        output = output + np.asarray(labeled_array == labeli, np.uint8)
    return output


def label_smooth(volume):
    [D, H, W] = volume.shape
    s = ndimage.generate_binary_structure(2, 1)
    for d in range(D):
        if (volume[d].sum() > 0):
            volume_d = get_largest_k_components(volume[d], k=5)
            if (volume_d.sum() < 10):
                volume[d] = np.zeros_like(volume[d])
                continue
            volume_d = ndimage.morphology.binary_closing(volume_d, s)
            volume_d = ndimage.morphology.binary_opening(volume_d, s)
            volume[d] = volume_d
    return volume


def get_dataloader_ISIC(config):
    data_dir = config["Dataset"]["data_dir"]
    batch_size = config["Dataset"]["batch_size"]
    num_worker = config["Dataset"]["num_workers"]
    from dataset.ACDCDataset import ISICDataset
    train_transform = A.Compose([
        A.Resize(1280, 1280),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(p=0.2),
        A.RandomCrop(1024, 1024),
        A.GaussNoise(0.005, 0, per_channel=False),
    ])
    test_transform = A.Compose([
        A.Resize(1280, 1280),
        A.RandomCrop(1024, 1024),
    ])

    dataset_train, dataset_val = ISICDataset(trainfolder=join(data_dir, "train"),
                                             transform=train_transform), \
        ISICDataset(trainfolder=join(data_dir, "test"), transform=test_transform)

    labeled_sampler, unlabeled_sampler = get_samplers(len(dataset_train), config["AL"]["initial_labeled"],
                                                      with_pseudo=False)

    dulabeled = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=batch_size,
                                            sampler=unlabeled_sampler,
                                            persistent_workers=True,
                                            pin_memory=True,
                                            prefetch_factor=num_worker,
                                            num_workers=num_worker)

    dlabeled = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=batch_size,
                                           sampler=labeled_sampler,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           prefetch_factor=num_worker,
                                           num_workers=num_worker)

    dval = torch.utils.data.DataLoader(dataset_val,
                                       batch_size=1,
                                       persistent_workers=True,
                                       pin_memory=True,
                                       prefetch_factor=num_worker,
                                       num_workers=num_worker)

    return {
        "labeled": dlabeled,
        "unlabeled": dulabeled,
        "test": dval
    }


def get_dataloader_ACDC(config, with_pseudo=False):
    data_dir = config["Dataset"]["data_dir"]
    batch_size = config["Dataset"]["batch_size"]
    num_worker = config["Dataset"]["num_workers"]
    from dataset.ACDCDataset import ACDCDataset2d, ACDCDataset3d
    train_transform = A.Compose([
        A.PadIfNeeded(256, 256),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(p=0.2),
        A.RandomCrop(192, 192),
        A.GaussNoise(0.005, 0, per_channel=False),
    ])
    dataset_train, dataset_val = ACDCDataset2d(trainfolder=join(data_dir, "train"),
                                               transform=train_transform), \
        ACDCDataset3d(folder=join(data_dir, "test"))
    labeled_sampler, *unlabeled_sampler = get_samplers(len(dataset_train), config["AL"]["initial_labeled"],
                                                       with_pseudo=with_pseudo)
    retval = {}
    if with_pseudo:
        from dataset.SphDataset import PseudoDataset2d
        unlabeled_sampler, pseudo_sampler = unlabeled_sampler
        dpseudo = torch.utils.data.DataLoader(PseudoDataset2d(datafolder=join(num_worker, "train"),
                                                              transform=train_transform),
                                              batch_size=batch_size,
                                              sampler=pseudo_sampler,
                                              persistent_workers=True,
                                              pin_memory=True,
                                              prefetch_factor=num_worker,
                                              num_workers=num_worker)
        retval["pseudo"] = dpseudo
    else:
        unlabeled_sampler = unlabeled_sampler[0]

    dulabeled = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=batch_size,
                                            sampler=unlabeled_sampler,
                                            persistent_workers=True,
                                            pin_memory=True,
                                            prefetch_factor=num_worker,
                                            num_workers=num_worker)

    dlabeled = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=batch_size,
                                           sampler=labeled_sampler,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           prefetch_factor=num_worker,
                                           num_workers=num_worker)

    dval = torch.utils.data.DataLoader(dataset_val,
                                       batch_size=1,
                                       persistent_workers=True,
                                       pin_memory=True,
                                       prefetch_factor=num_worker,
                                       num_workers=num_worker)

    return {
        **retval,
        "labeled": dlabeled,
        "unlabeled": dulabeled,
        "test": dval
    }


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * sigmoid_rampup(epoch, 80)


def read_yml(filepath):
    assert os.path.exists(filepath), "file not exist"
    with open(filepath) as fp:
        config = yaml.load(fp, yaml.FullLoader)
    return config


def random_seed(config):
    import torch.backends.cudnn as cudnn
    seed = config["Training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def init_logger(config):
    import sys
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    outputdir = config["Training"]["output_dir"]
    fh = logging.FileHandler(f"{outputdir}/{time.time()}.log")
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(f"Query Strategy: {outputdir}")
    logger.info(config)
    return logger


def parse_config():
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str,
                        default="config-ssl/config-ssl.yml")
    parser.add_argument("--strategy", "-s", type=str,
                        default="config-ssl/strategy.yml")
    args = parser.parse_args()

    config, all_strategy = read_yml(args.config), read_yml(args.strategy)

    config["Training"]["output_dir"] = config["AL"]["query_strategy"] if config["Training"]["output_dir"] is None \
        else config["Training"]["output_dir"]

    os.makedirs(config["Training"]["output_dir"], exist_ok=True)

    config["Training"]["checkpoint_dir"] = os.path.join(config["Training"]["output_dir"], "checkpoint")
    os.makedirs(config["Training"]["checkpoint_dir"], exist_ok=True)
    config["all_strategy"] = all_strategy
    return config


if __name__ == '__main__':
    print(build_strategy("MaxEntropy"))
