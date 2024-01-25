# === your unchanged research code

from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch


# task function
def eval_model(
        model: torch.nn.Module,
        dataloader: DataLoader,
):
    ...

# with the following helper functions:
def make_cifar_loader(batch_size: int = 64) -> DataLoader:
    dataset = CIFAR10(root="data", train=True, transform=ToTensor(), download=False)
    return DataLoader(dataset, batch_size=batch_size)

def make_mnist_loader(batch_size: int = 64) -> DataLoader:
    dataset = MNIST(root="data", train=True, transform=ToTensor(), download=False)
    return DataLoader(dataset, batch_size=batch_size)

# ==== now add hydra

from hydra_zen import store, builds

# define a configuration group (contains exchangeable configs for a dataloader)
data_store = store(group="dataloader")
# store configs generated based on the signature of our make_loader functions
# builds creates a dataclass configuration class based on the signature
# `populate_full_signature` tells hydra to build a Config that includes all parameters
# of the make_cifar_loader function.
CifarConfig = builds(make_cifar_loader, populate_full_signature=True)
# which can then be added to our store
data_store(CifarConfig, name="cifar")
# same for mnist
# we can also overwrite defaults right here
MnistConfig = builds(make_mnist_loader, batch_size=128, populate_full_signature=True)
data_store(MnistConfig, name="mnist")

# create entrypoint config, by inspecting the task_function
MainConfig = builds(
    eval_model,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"dataloader": "cifar"},  # default dataloader is cifar
        {"model": "load_model"}
    ]
)
# store the entrypoint config
store(MainConfig, name="eval_model")

store.add_to_hydra_store()  # cross the bridge from hydra-zen to hydra

# first we need a configurable target.
# lets create a model loading function for this:
def load_model(model_name : str) -> torch.nn.Module:
    model = torch.hub.load('pytorch/vision', model_name, pretrained=True)
    model.eval()
    return model

model_store = store(load_model, group="model")

store.add_to_hydra_store()  # cross the bridge from hydra-zen to hydra


# ==== add a cli

if __name__ == "__main__":
    from hydra_zen import zen

    # expose cli
    zen(eval_model).hydra_main(
        config_name="eval_model",  # refers to the name of the MainConfig
        version_base="1.1",
    )
