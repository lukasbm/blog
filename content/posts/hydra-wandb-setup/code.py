import torch
from hydra_zen import store, builds
from torch.utils.data import DataLoader


def eval_model(
        model: torch.nn.Module,
        dataloader: DataLoader,
):
    print(model)
    print(dataloader)


def make_cifar_loader(batch_size: int = 64) -> DataLoader:
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor

    dataset = CIFAR10(root="data", train=True, transform=ToTensor(), download=True)
    return DataLoader(dataset, batch_size=batch_size)


def make_mnist_loader(batch_size: int = 64) -> DataLoader:
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor

    dataset = MNIST(root="data", train=True, transform=ToTensor(), download=True)
    return DataLoader(dataset, batch_size=batch_size)


data_store = store(group="dataloader")
data_store(builds(make_cifar_loader, populate_full_signature=True), name="cifar")
data_store(builds(make_mnist_loader, populate_full_signature=True), name="mnist")

from torchvision.models import ResNet, AlexNet

model_store = store(group="model")
model_store(builds(ResNet, populate_full_signature=True), name="resnet")
model_store(builds(AlexNet, populate_full_signature=True), name="alexnet")

# create entrypoint config, by inspecting the task_function
Config = builds(
    eval_model,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"dataloader": "cifar"}
    ]
)
# store the entrypoint config
store(Config, name="eval_model")

store.add_to_hydra_store(overwrite_ok=True)  # cross the bridge from hydra-zen to hydra

if __name__ == "__main__":
    from hydra_zen import zen

    # expose cli
    zen(eval_model).hydra_main(
        config_name="eval_model",
        version_base="1.1",
        config_path=None
    )
