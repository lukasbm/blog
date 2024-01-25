+++
title = "Supercharge your Cluster Computation Using HydraZen"
description = "Use HydraZen with a plugin to automatically submit jobs to slurm without having to write shell scripts"
author = "Lukas Böhm"
draft = true
categories = ['Guides']
tags = ['hydra', 'hydra-zen', 'python', 'research']
+++

In [a previous guide](../hydra-wandb-setup) we explored how we can use Hydra(Zen) and Weights&Biases to speed up our AI research workflow.
At the end of the last article i mentioned that I'm going to write a follow up for [SLURM](https://slurm.schedmd.com/overview.html), a popular job management system for computer clusters, once I've gotten comfortable with it.
Well, now that I've spent some time with it i can confidently say that it's a huge timesaver without a high setup cost.

When running a python script on a High performance Compute (HPC) cluster, we usually need to write a shell script that sets up the environment, specifies where to find the source files and allocates the resources (CPU Cores, GPUs, etc.).
The scripts usually look something like this:
```bash
#!/bin/bash -l
#
# Parameters
#SBATCH --gres=gpu:1  # request one gpu
#SBATCH --time=03:00:00  # time limit 3 hours
#SBATCH --job-name=important_calculations
#SBATCH --output=<output_file>
#SBATCH -error=<error_file>

# Setup
source venv/bin/activate
export WANDB_MODE=offline


# commands
srun --unbuffered python3 calculations.py params1
srun --unbuffered python3 calculations.py params2
```

This is just a minimal example that could be dozens of lines longer.
I specifically also omitted the command line arguments for the python script.
These bash scripts are often required to be duplicated multiple times for different setups and parameters which causes them to be a main source for errors and boilerplate.
We would also need to specify a location for the output and error files.

To de-deduplicate and improve our workflow, let's see how we can turn the previous code snippet into this equivalent representation:
```bash
python calculations.py \
    hydra/launcher=submitit_slurm_custom \
    params=params1,params2 \
    --multirun
```

As you can see this is significantly shorter.
Furthermore, this one-liner also takes care of placing the output files in the hydra-run directory and setting up the *job array* for multirun.

## Getting Started

We again, start by adding the required dependencies to our project.
Besides the requirements from last time (`wandb`, `hydra-zen`, `hydra-core`) we need this one:
```
pip install hydra-submitit-launcher
```

[Submit It](https://github.com/facebookincubator/submitit) by itself is a python library for submitting jobs to SLURM.
Here we are installing the the hydra plugin that ships with two new job launchers:
- `submitit_local`: This can be used for local job running and testing. Has a reduced configuration set.
- `submitit_slurm`: Actually launch the SLURM job with all it's parameters (as usually specified by `#SBATCH`.)


## Configuration

Configuration works as usual, we createt a Config any way we want (`make_config`, `dataclass`, `builds`),
add it to the store and the select the desired configuration group, maybe even include some overwrites.

As we are dealing with the Job launcher we need to overwrite the internal Hydra Config.
In Hydra-Zen this will look like this:
```python
store(
    HydraConf(
        launcher={
            "timeout_min": 180,
            "gres": "gpu:1",
        }
    ),
    name="config_slurm",
    group="hydra"
)
```

Here overwrite the default hydra config to set the timeout and resource requirements
To see which settings have been applied and what other settings are available, we can inspect it using the `--cfg all` flag which shows the config (similar to `--help`, but also including the hydra config) as it will be used in the script.
We can narrow the output down to a specific subconfig using the `-p` flag.
```
python calculations.py hydra/launcher=submitit_slurm --cfg all -p hydra.launcher
```

This is a simple hack to change the default params for the `submitit_slurm` launcher.
To create exchangeable pre-set configs for the launcher we need to be more specific and overwrite it completely:
```python
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf, LocalQueueConf

store(
    SlurmQueueConf(
        gres="gpu:1",
        nodes=1,
        timeout_min=180,
        additional_parameters={  # more sbatch parameters (not included in SlurmQueueConf)
            "clusters": "gpu",
        },
        setup=[  # setup commands to run before the job starts
            "source venv/bin/activate",
            "export WANDB_MODE=offline",
        ],
    ),
    name="submitit_slurm_small_job",  # preset name
    group="hydra/launcher",  # add to launcher group config
    provider="submitit_launcher",  # specifiy that it belong to the launcher
)

store(
    SlurmQueueConf(
        gres="gpu:4",
        nodes=4,
        timeout_min=30,
        additional_parameters={  # more sbatch parameters (not included in SlurmQueueConf)
            "clusters": "gpu",
        },
        setup=[  # setup commands to run before the job starts
            "source venv/bin/activate",
            "export WANDB_MODE=offline",
        ],
    ),
    name="submitit_slurm_big_job",  # preset name
    group="hydra/launcher",  # add to launcher group config
    provider="submitit_launcher",  # specifiy that it belong to the launcher
)
```


## Usage

Now that we have our new launcher set up, let's see how easy it is to launch jobs.

One note beforehand: The submitit launcher only works in multirun mode.
So make sure to always set the `--multirun` flag even if you are only planning one run.

