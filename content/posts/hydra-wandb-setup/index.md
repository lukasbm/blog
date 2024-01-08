+++
title = "A Cleaner Approach To AI Research"
author = "Lukas Böhm"
description = "Using tools from MLOps to accelerate your AI research"
draft = true
+++

Many researches are often presented with having to evaluate a new model on
multiple datasets or sets of hyperparameters.
Research code is famously known for having bad quality due to it's nature of being thrown away after the publication.
Often times no reproducible environment is set up and everything is crammed into a hand-full of scripts, or even worse, jupyter notebooks.
These overloaded scripts often try do everything from data manipulation, training and evaluation at once,
but more often just ends up as a giant pile of spaghetti nobody wants to deal with.
By the time you write your paper and create the graphs and figures your entire workflow
has become so messy that your productivity grinds to a painful halt.

If you want to pride yourself as a good researcher that publishes high quality, reproducible code,
you might want to stick around as we explore how we can employ two technologies to transform your workflow, without having to rewrite anything!

## Current Problems

Often it is assumed that there has to be a trade-off between working fast and employing best practices.
Naturally, most people prefer to work fast, thus we end up in this dire situation.

![What if i told you there wasn't a tradeoff between working fast and clean](./what-if-i-told-you.jpg)

What if i told you there wasn't a tradeoff between working fast and clean.

Currently most research repositories consist of a very rigid codebase that is not very adaptable to change, i.e. new parameters that have an effect on the behaviour.
Meanwhile the run configurations are stored in dozens of bash scripts,
that have endless amounts of duplication and would have to be rewritten if the code changed.

Another common issue is the data handling, especially the output of said scripts.
They usually lack any form of versioning or metadata,
making the entire postprocessing step significanctly harder.

To battle these issues we will explore Hydra-Zen (a lightweights Hydra wrapper) and Weights&Biases (wandb).

[Hydra](https://hydra.cc) is an open source tool by facebook for quickly building hierarchical configurations without any boilerplate.
Basic tools for configuration like python's `argparse` library require boilerplate to map from the arguments, to the objects or functions they configure and parametrize.
Hydra, skips this step by mapping a set of yaml configuration file directly to a class or function. "Set of yaml files" sound suspiciously like the aforementioned problem of duplicated scripts that have to be rewritten if the code changes. Yeah, thats exactly the problem with hydra, hence we use [Hydra-Zen](https://mit-ll-responsible-ai.github.io/hydra-zen/), a pure python solution to configuration.

Our other tool, Weights&Biases, is used for versioning the outputs and annotating them with metadata.
As a MLOps tool it closely follows the style of well known CI&CD platforms, where runs are connected to the artifacts they generate for better reproducibility.
In our case *runs* relate to the models we are training or the data we are processing and *artifacts* relate to the output they produce like model checkpoints, plots, tables, etc. 

## Solution

To make this guide more practical, assume the following task function:
```python
# in eval_model.py
def eval_model(
        model: torch.nn.Module,
        dataloader: torch.utils.data.Dataloader,
): ...
```

with the following helper functions:
```python
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
```


Making this function configurable in the traditional way would require endless amounts of if-else statements. Let's see how we can clean this up.

We start by installing the required dependencies:
```bash
pip install hydra-zen hydra-core wandb
```

Let's start a new file, to show that you can gradually add better configurations:
```python
from hydra_zen import store, builds
from eval_model import eval_model, make_mnist_loader, make_cifar_loader

# define a configuration group (contains exchangeable configs for dataloader)
data_store = store(group="dataloader")
# generated based on the signature of our make_loader functions
data_store(builds(make_cifar_loader, populate_full_signature=True), name="cifar")
data_store(builds(make_mnist_loader, populate_full_signature=True), name="mnist")

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
```

The main concept of hydra is the *store*. An abstraction layer that only exists in the configuration phase. Everything we add to stores is configurable, either by code or via a CLI.
Once we launch our script, the store automagically populate the desired target objects and run the task functions.

In the code above TODO TODO TODO

We can see how it looks by adding a CLI:
```python
if __name__ == "__main__":
    from hydra_zen import zen

    # expose cli
    zen(eval_model).hydra_main(
        config_name="eval_model",
        version_base="1.1",
        config_path=None
    )
```
If we run `python eval_script.py --help` we will get an overview of our configuration as it will be initialized. It should look something like:
```python
== Configuration groups ==
Compose your configuration from those groups (group=option)

dataloader: cifar, mnist


== Config ==
Override anything in the config (foo.bar=value)

_target_: __main__.train_model
model: ???
dataloader:
  _target_: __main__.make_cifar_loader
  batch_size: 64
```

We can see that dataloader is already assigned to be instantiated by make_cifar_loader with a batch size of 64. However we can easily overwrite this nested value by running `python eval_script.py dataloader.batch_size=128 --help`.
The value for model is still missing, which makes it impossible to run the script.
Let's extend the script for a `model_store`:



## Glimpse into the Future


## References

<!-- TODO: use markdown footnotes -->

# Comments
<!-- 
<script src="https://giscus.app/client.js"
        data-repo="lukasbm/blog"
        data-repo-id="R_kgDOLBREVQ"
        data-category="General"
        data-category-id="DIC_kwDOLBREVc4CcOfk"
        data-mapping="title"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="top"
        data-theme="preferred_color_scheme"
        data-lang="en"
        crossorigin="anonymous"
        async>
</script> -->
