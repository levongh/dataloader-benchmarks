from src.dataloaders.base import DataLoader
from src.datasets.coco.index import CocoDatasets
from src.datasets.coco.base import LABEL_DICT, core_transform
from indra import Loader
from src.libraries.hub3 import filter_by_class
from functools import partial

DATASET = CocoDatasets["hub3"]


def identity(x, y):
    t = {}
    t["categories"] = y["labels"]
    t["boxes"] = y["boxes"]
    return x, t


def aux(mode, sample):
    image = sample["images"]
    return core_transform(mode, identity, image, sample)


def collate_fn(batch):
    return tuple(zip(*batch))


class Hub3Loader(DataLoader):
    transform = None

    def _get(self, mode, **kwargs):

        # kwargs["num_workers"] = 0 # increased performance
        # self.transform = partial(aux, mode)

        if self.remote:
            dataset = DATASET.get_remote(mode=mode, transforms=None)
        else:
            dataset = DATASET.get_local(mode=mode, transforms=None)

        if self.filtering:
            FC = [LABEL_DICT[c] for c in self.filtering_classes]
            dataset = filter_by_class(dataset, FC)

        loader = Loader(
            dataset,
            # transform_fn=self.transform_hub,
            transform_fn=partial(aux, mode),
            distributed=self.distributed,
            collate_fn=collate_fn,
            **kwargs
        )
        return loader

    def get_train_loader(self, **kwargs):
        return self._get("train", **kwargs)

    def get_val_loader(self, **kwargs):
        return self._get("val", **kwargs)

    # def transform_hub(self, sample):
    #     sample["images"] = self.transform(sample["images"])
    #     return sample
