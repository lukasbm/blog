import signal

from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from hydra_zen import make_config
from hydra_zen import store

# store the configures hydra launchers

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

# create global setups for the launchers

setup_store = store(group="setup", package="_global_")

# hacky way to get the main config
# store is a dict of dicts, the None key is the main config and the inner dict should only have a single value
conf = list(store[None].values())[0]

setup_store(
    make_config(
        hydra_defaults=["_self_", {"/hydra/sweep": "hpc"}, {"/hydra/run": "hpc"},
                        {"override /hydra/launcher": "submitit_slurm_custom_gpu"}],
        bases=(conf,),  # has to inherit from the main config
    ),
    name="hpc_gpu",
)

setup_store(
    make_config(
        hydra_defaults=["_self_", {"/hydra/sweep": "hpc"}, {"/hydra/run": "hpc"},
                        {"override /hydra/launcher": "submitit_slurm_custom_batch"}],
        bases=(conf,),  # has to inherit from the main config
    ),
    name="hpc_batch",
)

# hacky way to stop runs from crashing

signal.signal(signal.SIGUSR1, signal.SIG_IGN)  # ignore SIGUSR1
signal.signal(signal.SIGUSR2, signal.SIG_IGN)  # ignore SIGUSR2
signal.signal(signal.SIGCONT, signal.SIG_IGN)  # ignore SIGCONT
signal.signal(signal.SIGTERM, signal.SIG_IGN)  # ignore SIGTERM
signal.signal(signal.SIGHUP, signal.SIG_IGN)  # ignore SIGTSTP


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
