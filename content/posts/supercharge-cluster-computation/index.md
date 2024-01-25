+++
title = "Supercharge your Cluster Computation Using HydraZen"
description = "Use HydraZen with a plugin to automatically submit jobs to slurm without having to write shell scripts"
author = "Lukas Böhm"
draft = true
categories = ['Guides']
tags = ['hydra', 'hydra-zen', 'slurm', 'SLURM', 'submitit', 'python', 'research']
+++

In [a previous guide](../hydra-wandb-setup) we explored how to use Hydra(Zen) and Weights&Biases to speed up our AI research workflow.
At the end of the last article I mentioned that I'm going to write a follow up for [SLURM](https://slurm.schedmd.com/overview.html), a popular job management system for computer clusters, once I've gotten comfortable with it.
Well, now that I've spent some time with it i can confidently say that it's a huge timesaver with a low setup cost.

When running a python script on a High performance Compute (HPC) cluster, we usually need to write a shell script that sets up the environment, specifies where to find the source files and allocates the resources (CPU Cores, GPUs, etc.).
The scripts usually look something like this:
```bash
#!/bin/bash -l
#
# Parameters
#SBATCH --gres=gpu:1  # request one gpu
#SBATCH --time=03:00:00  # time limit 3 hours
#SBATCH --job-name=important_calculations
#SBATCH --array=0-14%15
#SBATCH --output=/some/complicated/path/%A_%a/output.log
#SBATCH --error=/some/complicated/path/%A_%a/error.log

# Setup
source venv/bin/activate
export WANDB_MODE=offline

# commands
srun --unbuffered python3 -u code.py params=value
```

This is just a minimal example that could be dozens of lines longer.
These bash scripts are often required to be duplicated multiple times for different setups and parameters which causes them to be a main source for errors and boilerplate.
We would also need to specify a location for the output and error files manually, which makes tracking the runs more complicated.

To de-deduplicate and improve our workflow, let's see how we can turn the previous code snippet into this equivalent command call:
```bash
python code.py +setup=hpc_gpu params=value --multirun
```

Furthermore, this one-liner also takes care of placing the output files in the hydra-run directory and setting up the job array for multirun.

## Getting Started

We again, start by adding the required dependencies to our project.
Besides the requirements from last time (`wandb`, `hydra-zen`, `hydra-core`) we need these:
```
pip install hydra-submitit-launcher submitit
```

[Submit It](https://github.com/facebookincubator/submitit) by itself is a python library for submitting jobs to SLURM.
Here we are installing the the hydra plugin that ships with two new job launchers:
- `submitit_local`: This can be used for local job running and testing. Has a reduced configuration set.
- `submitit_slurm`: Actually launch the SLURM job with all it's parameters (as usually specified via `#SBATCH`.)


## Configuration

Configuration works as usual, we create a Config any way we want (`make_config`, `dataclass`, `builds`),
add it to the store and select the desired configuration group, maybe even include some overwrites.

In this example we want to create two configurations. One that requests a multiple GPUs for a short time, and one that requests multiple CPU nodes for a long time (batch processing).
As we are dealing with the Job launcher we need to overwrite the internal Hydra Config.
In Hydra-Zen this will look like this:
```python
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf, LocalQueueConf

store(
    SlurmQueueConf(
        gres="gpu:4",
        timeout_min=30,
        setup=[  # setup commands to run before the job starts
            "source venv/bin/activate",
            "export WANDB_MODE=offline",
        ],
    ),
    name="submitit_slurm_custom_gpu",
    group="hydra/launcher",
)

store(
    SlurmQueueConf(
        timeout_min=180,
        nodes=6,
        setup=[  # setup commands to run before the job starts
            "source venv/bin/activate",
            "export WANDB_MODE=offline",
        ],
    ),
    name="submitit_slurm_custom_batch",
    group="hydra/launcher",
)
```

Here overwrite the default hydra config to set the timeout and resource requirements
To see which settings have been applied and what other settings are available, we can inspect it using the `--cfg all` flag which shows the config (similar to `--help`, but also including the hydra config) as it will be used in the script.
We can narrow the output down to a specific subconfig using the `-p` flag.
```
python calculations.py hydra/launcher=submitit_slurm --cfg all -p hydra.launcher
```

Now this would already work but there is one other improvement we can do.
Often times running code on the HPC requires storing the results on other drives.
Instead of specifying the output location and launcher seperately each time we run the script,
we can create global presets:
```python
# exclude from hierarchical config by setting package to _global_
setup_store = store(group="setup", package="_global_")

# hacky way to get the main config
# store itself is a dict of dicts, the None key is the main config and the inner dict should only have a single value
conf = list(store[None].values())[0]

# create a new config which changes the location of the outputs
store(
    SweepDir(dir=os.path.join("some", "new", "basedir", "${now:%Y-%m-%d}", "${now:%H-%M-%S}")),
    name="hpc",
    group="hydra/sweep",
)

# create global presets based on the configs
setup_store(
    make_config(
        hydra_defaults=["_self_", {"/hydra/sweep": "hpc"}, {"override /hydra/launcher": "submitit_slurm_custom_gpu"}],
        bases=(conf,),  # has to inherit from the main config
    ),
    name="hpc_gpu",
)

setup_store(
    make_config(
        hydra_defaults=["_self_", {"/hydra/sweep": "hpc"}, {"override /hydra/launcher": "submitit_slurm_custom_batch"}],
        bases=(conf,),  # has to inherit from the main config
    ),
    name="hpc_batch",
)
```

This allows us to set up everything we need for hpc as a single paramter: `+setup=hpc_gpu`.
Notice the `+` at the beginning. This means we append to the defaults list, as setup is not included there yet.

Lastly we just need to add the obligatory main function with a dummy task function:
```python
@store(name="submitit_tutorial")
def task_fn(params):
    pass


store.add_to_hydra_store(overwrite_ok=True)  # overwrite required for callbacks

if __name__ == "__main__":
    from hydra_zen import zen

    # expose cli based on config name and task function
    zen(task_fn).hydra_main(
        config_name="submitit_tutorial",  # refers to the name of the MainConfig
        version_base="1.1",
    )
```


## Usage

Now that was a lot of code snippets, let's actually use it!
As mentioned in the beginning we can run the following command:
```bash
python code.py +setup=hpc_gpu params=value --multirun
```

Running this creates a new run directory in the desired location.
If you look really close you will see that there's a new hidden folder included now called `.submitit`.
This is where everything related to slurm ends up: The submission script, the log files whose location we no longer have to specify manually, etc.

Now that we have our new launcher set up, let's see how easy it is to launch jobs.

One note beforehand: The submitit launcher only works in multirun mode.
So make sure to always set the `--multirun` flag even if you are only planning one run.

## Background

<!-- Explain how jobs are launched (diagram?) -->


## Caveats

<!-- Signal handling -->


## Conclusion
