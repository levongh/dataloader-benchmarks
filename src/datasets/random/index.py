from src.datasets.random.ffcv import FFCVDataset
from src.datasets.random.pytorch import PytorchDataset
from src.datasets.random.hub import HubDataset
from src.datasets.random.squirrel import SquirrelDataset
from src.datasets.random.hub3 import Hub3Dataset
from src.datasets.random.torchdata import TorchdataDataset
from src.datasets.random.webdataset import WebdatasetDataset

RandomDatasets = {
    "ffcv": FFCVDataset(),
    "pytorch": PytorchDataset(),
    "hub": HubDataset(),
    "webdataset": WebdatasetDataset(),
    "torchdata": TorchdataDataset(),
    "squirrel": SquirrelDataset(),
    "hub3": Hub3Dataset(),
}