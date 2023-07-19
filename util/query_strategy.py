import albumentations as A
from os.path import join
import random
from pymic.util.evaluation_seg import binary_dice
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.jitfunc as f
from torch import nn
from sklearn.metrics import pairwise_distances
from torch.nn import functional as F
from util import SubsetSampler


# todo: 前向过程应由trainer负责，这样可以避免不同模型输出不一样的问题


class LimitSortedList(object):

    def __init__(self, limit, descending=False) -> None:
        self.descending = descending
        self.limit = limit
        self._data = []

    def reset(self):
        self._data.clear()

    @property
    def data(self):
        return map(lambda x: int(x[0]), self._data)

    def extend(self, idx_score):
        assert isinstance(idx_score, (torch.Tensor, np.ndarray, list, tuple))
        idx_score = list(idx_score)
        self._data.extend(idx_score)
        if len(self._data) > self.limit:
            self._data = sorted(self._data, key=lambda x: x[1], reverse=self.descending)[:self.limit]


class QueryStrategy(object):

    def __init__(self, dataloader: DataLoader) -> None:
        super().__init__()
        self.unlabeled_dataloader = dataloader["unlabeled"]
        self.labeled_dataloader = dataloader["labeled"]

    def select_dataset_idx(self, query_num):
        raise NotImplementedError

    def convert2img_idx(self, ds_idx, dataloader: DataLoader):
        return [dataloader.sampler.indices[img_id] for img_id in ds_idx]

    def sample(self, query_num):
        dataset_idx = self.select_dataset_idx(query_num)
        img_idx = self.convert2img_idx(dataset_idx, self.unlabeled_dataloader)
        self.labeled_dataloader.sampler.indices.extend(img_idx)
        # 注意这里不可以用index来移除，因为pop一个之后，原数组就变换了
        # for i in dataset_idx:
        #     self.unlabeled_dataloader.sampler.indices.pop(i)
        for item in img_idx:
            self.unlabeled_dataloader.sampler.indices.remove(item)


class RandomQuery(QueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)

    def sample(self, query_num):
        np.random.shuffle(self.unlabeled_dataloader.sampler.indices)
        self.labeled_dataloader.sampler.indices.extend(self.unlabeled_dataloader.sampler.indices[:query_num])
        del self.unlabeled_dataloader.sampler.indices[:query_num]


class SimpleQueryStrategy(QueryStrategy):
    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        assert "trainer" in kwargs
        assert "descending" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model
        self.descending = kwargs["descending"]

    def compute_score(self, model_output):
        raise NotImplementedError

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        self.model.eval()
        device = next(iter(self.model.parameters())).device
        q = LimitSortedList(limit=query_num, descending=self.descending)
        for batch_idx, (img, _) in enumerate(self.unlabeled_dataloader):
            img = img.to(device)
            output = self.model(img)
            score = self.compute_score(output).cpu()
            assert len(score) == len(img), "shape mismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entropy = torch.column_stack([torch.arange(offset, offset + len(img)), score])
            q.extend(idx_entropy)
        return q.data


class MaxEntropy(SimpleQueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, descending=True, **kwargs)

    def compute_score(self, model_output):
        model_output, _ = model_output
        return f.max_entropy(model_output.softmax(dim=1))


class MarginConfidence(SimpleQueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, descending=False, **kwargs)

    def compute_score(self, model_output):
        model_output, _ = model_output
        return f.margin_confidence(model_output.softmax(dim=1))


class LeastConfidence(SimpleQueryStrategy):
    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, descending=False, **kwargs)

    def compute_score(self, model_output):
        model_output, _ = model_output
        return f.least_confidence(model_output.softmax(dim=1))
