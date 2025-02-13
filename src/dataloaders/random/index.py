from src.dataloaders.random.ffcv import FFCVLoader
from src.dataloaders.random.pytorch import PytorchLoader
from src.dataloaders.random.hub import HubLoader
from src.dataloaders.random.squirrel import SquirrelLoader
from src.dataloaders.random.webdataset import WebdatasetLoader
from src.dataloaders.random.torchdata import TorchdataLoader
from src.dataloaders.random.hub3 import Hub3Loader

RandomLoaders = {
    "pytorch": PytorchLoader,
    "ffcv": FFCVLoader,
    "hub": HubLoader,
    "webdataset": WebdatasetLoader,
    "torchdata": TorchdataLoader,
    "squirrel": SquirrelLoader,
    "hub3": Hub3Loader,
}
