import logging

from pathlib import Path
from src.datasets.base import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from src.datasets.random.base import get_random


class FFCVDataset(Dataset):

    def __init__(self):
        super().__init__("random", "ffcv")

    def generate_locally(self, mode="train", transforms=None):

        path = super().generate_locally(mode, transforms)
        if not path:
            return None
        path.mkdir(parents=True, exist_ok=True)

        random = get_random(mode, download=True, transform=transforms)

        write_path = path / f"random-{mode}.beton"
        # Pass a type for each data field
        writer = DatasetWriter(
            write_path,
            {
                # Tune options to optimize dataset size, throughput at train-time
                "image": RGBImageField(write_mode='jpg'),
                "label": IntField(),
            },
        )
        # Write dataset
        writer.from_indexed_dataset(random)

    def generate_remotely(self, mode="train", transforms=None):
        pass

    def get_local(self, transforms=None):
        """
        Does not seem like FFCV has a get dataset 
        """
        pass

    def get_remote(self, transforms=None):
        pass
