+++
title = "Supercharge your Cluster Computation Using Hydra"
author = "Lukas Böhm"
description = "Use Hydra Plugings to automatically submit jobs to slurm without having to write shell scripts"
draft = True
+++

In [a previous guide](../hydra-wandb-setup) we explored how we can use Hydra(Zen) and Weights&Biases to speed up our AI research workflow.
At the end of the last article i mentioned that I'm going to write a follow up for [SLURM](https://slurm.schedmd.com/overview.html), a popular job management system for computer clusters, once I've gotten comfortable with it.
Well, now that I've spent some time with it, we can dive right into it.

When running a python script on a High performance computer cluster, we usually need to write a shell script that sets up the environment, specifies where to find the source files and allocates the resources (CPU Cores, GPUs, etc.).
The scripts usually look something like this:
```bash
#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --time=03:00:00
#SBATCH --output=/tmp/%j_out.log
#SBATCH --error=/tmp/%j_err.log

set -euo pipefail # makes the script stop if any command fails

cd <source_directory>

# finicky python conda setup
module load python3.10
set +eu
eval "$(conda shell.bash hook)" || true
conda activate project-env
set -eu

srun --unbuffered python3 calculations.py
```

This is just a minimal example that could be dozens of lines longer.
I specifically also omitted the command line arguments for the python script.
These bash scripts are often required to be duplicated multiple times for different setups and parameters which causes them to be a main source for errors and boilerplate.

To de-deduplicate and improve our workflow, let's see how we can turn the previous code snippet into this equivalent representation: (FIXME: is the sbatch stuff included here?)
```bash
python calculations.py --multirun hydra/launcher=submitit_slurm
```

## Getting Started

We again, start by adding the required dependencies to our project.
Besides the requirements from last time (`wandb`, `hydra-zen`, `hydra-core`) we need this one:
```
pip install hydra-submitit-launcher
```

[Submit It](https://github.com/facebookincubator/submitit) by itself is a python library for submitting jobs to SLURM.
Here we are installing the the hydra plugin that ships with two new job launchers:
- `submitit_local`: This can be used for local job running and testing. Therefore has a reduced configuration set
- `submitit_slurm`: Actually launch the SLURM job with all it's parameters (as usually specified by `#SBATCH`)




We can explore the config as follows:
```
python calculations.py hydra/launcher=submitit_slurm --cfg hydra -p hydra.launcher 
```



## Usage

Now that we have our new launcher set up, let's see how easy it is to launch jobs.

One note beforehand: The submitit launcher only works in multirun mode.
So make sure to always set the `--multirun` flag even if you are only planning one run.

