import os

import torch
import wandb
from hydra import TaskFunction
from hydra.conf import HydraConf
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from hydra_zen import store, builds
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor


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


# first we need a configurable target.
# let's create a model loading function for this:
def load_model(model_name: str) -> torch.nn.Module:
    model = torch.hub.load('pytorch/vision', model_name, pretrained=True)
    model.eval()
    return model


model_store = store(load_model, group="model")

# create entrypoint config, by inspecting the task_function
MainConfig = builds(
    eval_model,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"dataloader": "cifar"},  # default dataloader is cifar
        {"model": "load_model"}  # model is defined through load_model (which is a function, that needs to be configured manually)
    ]
)
# store the entrypoint config
store(MainConfig, name="eval_model")


class WandBCallback(Callback):
    def __init__(self, *, project: str, entity: str, job_type: str) -> None:
        self.project = project
        self.entity = entity
        self.job_type = job_type

    def on_job_start(self, config: DictConfig, task_function: TaskFunction) -> None:
        wandb.init(
            project=self.project,
            entity=self.entity,
            job_type=self.job_type,
            # save the config as metadata for the W&B run
            config={**dict(config)},
            # hydra already creates an output folder.
            # We can put everything in the same place.
            dir=os.getcwd(),
        )

    def on_job_end(self, config: DictConfig, job_return: JobReturn) -> None:
        assert job_return.status == JobStatus.COMPLETED, "job did not complete!"
        results = job_return.return_value
        wandb.log({"results": results})
        wandb.finish()


store(
    HydraConf(
        callbacks={
            "wandb": builds(WandBCallback, entity="<wandb_username>",
                            project="learn_hydra_wandb", populate_full_signature=True),
        },
    ),
    name="config",
    group="hydra"
)

store.add_to_hydra_store()  # cross the bridge from hydra-zen to hydra (only need this once, at the end)

if __name__ == "__main__":
    from hydra_zen import zen

    # expose cli based on config name and task function
    zen(eval_model).hydra_main(
        config_name="eval_model",  # refers to the name of the MainConfig
        version_base="1.1",
    )
