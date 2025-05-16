from pathlib import Path
import pickle
from external.koutlis2024.utils import train_one_experiment, get_transforms

device = "cuda:0"
workers = 12
epochss = [1, 3, 5]
epochs_reduce_lr = [6, 11]
backbones = [("ViT-L/14", 1024)]
factors = [0.2, 0.4]
nprojs = [2, 4]
proj_dims = [512, 1024]
batch_sizes = [128]
lrs = [1e-3]
experiments = []
for backbone in backbones:
    for factor in factors:
        for nproj in nprojs:
            for proj_dim in proj_dims:
                for batch_size in batch_sizes:
                    for lr in lrs:
                        experiments.append(
                            {
                                "training_set": "elsad3",
                                "backbone": backbone,
                                "factor": factor,
                                "nproj": nproj,
                                "proj_dim": proj_dim,
                                "batch_size": batch_size,
                                "lr": lr,
                                "savpath": f"koutlis2024_training/{backbone[0].replace('/', '-')}_{factor}_{nproj}_{proj_dim}_{batch_size}_{lr}",
                            }
                        )
transforms_train, transforms_val, transforms_test = get_transforms()
for experiment in experiments:
    train_one_experiment(
        experiment=experiment,
        epochss=epochss,
        epochs_reduce_lr=epochs_reduce_lr,
        transforms_train=transforms_train,
        transforms_val=transforms_val,
        transforms_test=transforms_test,
        workers=workers,
        device=device,
        without=None,
        store=True,
    )

log_files = sorted(Path("koutlis2024_training").rglob("epoch=5_log.pickle"))
print(len(log_files))
for log_file in log_files:
    with open(log_file, "rb") as f:
        log = pickle.load(f)
        print(log_file.parent.name, log["results"])  # prints results after 1, 3, and 5 epochs