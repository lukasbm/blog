+++
title = "A Cleaner Approach To AI Research"
author = "Lukas Böhm"
description = "Using tools from MLOps to accelerate your AI research"
+++

Many researches are often presented with having to evaluate a new model on
multiple datasets or sets of hyperparameters.
Research code is famously known for having bad quality due to it's nature of being thrown away after the publication.
Often times no reproducible environment is set up and everything is crammed into a hand-full of scripts, or even worse, jupyter notebooks.
These overloaded scripts often try do everything from data manipulation, training and evaluation at once,
but more often just ends up as a giant pile of spaghetti nobody wants to deal with.
By the time you write your paper and create the graphs and figures your entire workflow
has become so messy that your productivity grinds to a painful halt.

If this sounds familiar, you might want to stick around as we explore how we can employ two technologies to transform your workflow, without having to rewrite anything!

## Current Problems

Often it is assumed that there has to be a trade-off between working fast and employing best practices.
Naturally, most people prefer to work fast, thus we end up in this dire situation.

![What if i told you there wasn't a tradeoff between working fast and clean](./what-if-i-told-you.jpg)

What if i told you there wasn't a tradeoff between working fast and clean.

Currently most research repositories consist of a very rigid codebase that is not very adaptable to change, i.e. new modules or parameters that have an effect on the behaviour.
Meanwhile the run configurations are stored in dozens of bash scripts,
that have endless amounts of duplication and would have to be rewritten if the code changed.

Another common issue is the data handling, especially the output of said scripts.
They usually lack any form of versioning or metadata,
making the postprocessing significanctly harder.

To battle these issues we will explore Hydra-Zen (a lightweights Hydra wrapper) and Weights&Biases (wandb).

[Hydra](https://hydra.cc) is an open source tool by facebook for quickly building hierarchical configurations without any boilerplate.
Basic tools for configuration like python's `argparse` library require boilerplate to map from the arguments, to the objects or functions they configure and parametrize.
Hydra, skips this step by mapping a set of yaml configuration file directly to a class or function. "Set of yaml files" sound suspiciously like the aforementioned problem of duplicated scripts that have to be rewritten if the code changes. Yeah, thats exactly the problem with hydra, hence we use [Hydra-Zen](https://mit-ll-responsible-ai.github.io/hydra-zen/), a pure python solution to configuration.

The other tool, [Weights&Biases](https://wandb.ai/), is used for versioning the outputs and annotating them with metadata.
As a MLOps tool it closely follows the style of well known CI&CD platforms, where runs are connected to the artifacts they generate for better reproducibility.
In our case *runs* relate to the models we are training or the data we are processing and *artifacts* relate to the output they produce like model checkpoints, plots, tables, etc.
These can be linked together to build a completely reproducible pipeline

## Configuration with Hydra

To make this guide more practical, assume the following task function.
This is our entrypoint we want to make configurable.

```python
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor
import torch

# task function
def eval_model(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
) -> EvalResults: ...

# with the following helper functions:
def make_cifar_loader(batch_size: int = 64) -> DataLoader:
    dataset = CIFAR10(root="data", train=True, transform=ToTensor(), download=True)
    return DataLoader(dataset, batch_size=batch_size)

def make_mnist_loader(batch_size: int = 64) -> DataLoader:
    dataset = MNIST(root="data", train=True, transform=ToTensor(), download=True)
    return DataLoader(dataset, batch_size=batch_size)
```

Making this function configurable in the traditional way would require endless amounts of if-else statements. Let's see how we can clean this up.

We start by installing the required dependencies:
```bash
pip install hydra-zen hydra-core
```

Now we just have to add some code to build a hierarchical configuration with it:
```python
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
        {"dataloader": "cifar"}
    ]
)
# store the entrypoint config
store(MainConfig, name="eval_model")

store.add_to_hydra_store()  # cross the bridge from hydra-zen to hydra
```

The main concept of hydra is the *store*. An abstraction layer that only exists in the configuration phase. Everything we add to stores is configurable, either by code or via a CLI.
Once we launch our script, the store automagically populates the desired target objects and run the task functions with the appropriate parameters.
In our case we generate a configuration group "dataloader" which can be used to populate any parameter named "dataloader".

We can test things out by adding a CLI:
```python
if __name__ == "__main__":
    from hydra_zen import zen

    # expose cli
    zen(eval_model).hydra_main(
        config_name="eval_model",  # refers to the name of the MainConfig
        version_base="1.1",
    )
```
We run `python code.py --help` to get an overview of our configuration as it will be initialized.
It should look something like:
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

We can see that dataloader is already assigned to be instantiated by make_cifar_loader with a batch size of 64.
However we can easily overwrite this nested value by running `python code.py dataloader.batch_size=128 --help`.
This is the power of hierarchical configurations!
But the value for model is still missing (denoted by `???`), which makes it impossible to run the script.
Let's extend the script for a `model_store`:

```python
# first we need a configurable target.
# lets create a model loading function for this:
def load_model(model_name : str) -> torch.nn.Module:
    model = torch.hub.load('pytorch/vision', model_name, pretrained=True)
    model.eval()
    return model

model_store = store(load_model, group="model")
```

You might notice this looks different than what we did with the datasets before.
That is because we are no longer just storing preconfigured configs,
but rather a completely configurable object.
This saves us some time as we don't have to store every option as a possible preset,
but rather have configured a function that can take in any model name!

Also don't forget to update your hydra defaults and hydra store:
```python
MainConfig = builds(
    eval_model,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"dataloader": "cifar"},  # default dataloader is cifar
        {"model": "load_model"}
    ]
)
store(MainConfig, name="eval_model")

store.add_to_hydra_store()  # cross the bridge from hydra-zen to hydra
```

We can finally use all of this to run our code:
```
python code.py model.model_name=alexnet dataloader=mnist dataloader.batch_size=512 
```

After you run the code, you might notice a new directory appead: `outputs`.
This is where all of your outputs along with metadata and log files reside now.
It is precicely ordered by date and time, so you can find the run immediately.
Lets look at an example directory:
```
<date>
   |- <time>
   |    |- .hydra
   |    |     |- config.yaml  # this is the yaml representation of our MainConfig
   |    |     |- hydra.yaml  # this is the internal hydra configuration
   |    |     |- overrides.yaml  # this is the overrides to the previous two
   |    |- _implementations.log  # the logged outputs
   |    |- ... possible other output files
```




TODO also write avbout the output dir hydra generated

TODO and how to configure hydra itself


## Experiment Management with Weights&Biases

We start by installing the required dependencies:
```bash
pip install wandb
```

### Combine with Hydra-Zen

We can leverage another powerful feature of Hydra: Callbacks.

Instead of having to wrap every task function in W&B runs, we can define a callback that sets up everything for us.

```python
import os
from typing import Any

import wandb
from hydra import TaskFunction
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from omegaconf import DictConfig

from src.helpers.wandb import save_results, create_artifact_name


class WandBCallback(Callback):
    def __init__(self, *, project, entity, job_type) -> None:
        self.project = project
        self.entity = entity
        self.job_type = job_type

    def on_job_start(
            self, config: DictConfig, task_function: TaskFunction
    ) -> None:
        wandb.init(
            project=self.project,
            entity=self.entity,
            job_type=self.job_type,
            # save the hydra config as metadata for the W&B run
            config={**dict(config)},
            # hydra already creates an output folder.
            # We can put everything in the same place.
            dir=os.getcwd(),  
        )

    def on_job_end(
            self, config: DictConfig, job_return: JobReturn
    ) -> None:
        assert job_return.status == JobStatus.COMPLETED
        do_something_with_the_job_results
        save_results(job_return.return_value, artifact_name=create_artifact_name(config))
        wandb.finish()
```

<!-- TODO: describe this callback -->

To include the callback we have to add it do the main HydraConfig:
```python
store(
    HydraConf(
        callbacks={
            "wandb": builds(WandBCallback, entity="<wandb_username>",
                project="<wandb_project_name>", job_type="eval_model",
                populate_full_signature=True),
        },
    ),
    name="config",
    group="hydra"
)
```


## Glimpse into the Future

I hope this introduction served you well in layout out how we can improve our AI research workflow.
Both Hydra and W&B are complex systems that offer significanly more functionality for every use case.

A crucial aspect for many researchers is to run the code on a compute cluster.
In these HPC Clusters you usually install your code in a directory and then use a bash script to submit a job
into a management system like SLURM.
Fortunately Hydra has a powerful plugin system, that offers everything from custom logging to sweeping hyperparameter search.

For slurm you might want to look at [this plugin](https://hydra.cc/docs/plugins/submitit_launcher/). \
🔜 I am currently testing out this plugin myself and will likely be writing a small follow-up for it.

Furthermore, both Hydra and WandB have an offline mode in case your compute nodes are isolated.

<!-- ## References -->

<!-- TODO: use markdown footnotes -->

# Comments

Please leave a comment if you have any questions or remarks 😃

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
</script>
