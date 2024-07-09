import torch
import wandb

wandb.login()

sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [2]},
        "epochs": {"values": [1000]},
        "lr": {"values":[1e-4,5e-4,1e-3,5e-3,1e-2]},
        "weight_decay": {"values":[1e-5,5e-5,1e-4,5e-4,1e-3]},
        "momentum": {"distribution": "uniform", "max": 0.99, "min": 0.9},
        "optimizer":{"values": ["sgd", "adamw"]},
        "scheduler":{"values":["cosine", "poly"]}
    },
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="project-name")

wandb.agent(sweep_id=sweep_id, count=5)


