# hyperparameter tuning using nested cross-validation

import itertools
import json

from  experiment import experiment

# 1. experiment: ideal num_workers --> with optimal_workers.py
# 2. experiment: ideal batch_size -> 8  epoch: 50
# 2. experiment: compare models   -> 2  epoch: 100

EXPERIMENT = "experiment-batchsize-"

search_space = {
    "lr": [1e-4],  # +  1e-3
    "model_name": [ "newUNet"], # + "baselineUNet"
    "num_filters": [64], # + 32, 128
    "num_depth": [5], # + 4, 6
    "activation_fn": ["relu"],  # + , "leaky_relu", "sigmoid"
    "loss_fn": ["focal"],  # + dice, bce
    # "focal_gamma": [1,1.5], # not sure about these values, default is 2
    # "bce_weight": [None, 1.5], # not sure about these values, default is None
    "batch_size": [4, 8, 16, 32, 64, 128],
    "num_workers": [5],
    "optimizer_name": ["adam"],  # + sgd
    # "data_augmentation": [[0, 0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], # random horizontal flip, random vertical flip, random rotation, random zoom , random brightness etc. # GENERATED
    # "scheduler": ["cosine", "step"], # cosine annealing lr, step lr
    # "scaler": [True, False],  # automatic mixed precision
    # "gradient_clipping": [True, False], # gradient clipping
    # "model-version" : ["may", "september"], # tutorial or my own implementation
}

default_params = {
    "lr": 1e-4,
    "model_name": "newUNet",
    "num_filters": 64,
    "num_depth": 5,
    "activation_fn": "relu",
    "loss_fn": "bce",
    "batch_size": 16,
    "num_workers": 0,
    "max_epochs": 100,
    "optimizer_name": "adam",
    "scaler_name": "True",
    "bce_weight": None,
    "focal_gamma": 2}

# returns the string representation of the experiment, it will be used to name the directories of checkpoints and writers
def experiment_name(params):
    # add to the name only if there is more than one value
    name = ""
    for key in params:
        if len(search_space[key]) > 1:
            name += f"{key}-{params[key]}--"
    name = name[:-2]
    return name


def main():
    # create cross product of all hyperparameters

    keys, values = zip(*search_space.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(len(experiments))
    print(experiments[1])

    print(experiment_name(experiments[1]))

    # Run each experiment
    all_results = []
    for params in experiments:

        # Run the experiment, set params, if they are not set use default values
        f1_score, run_time = experiment(
            experiment_name=experiment_name(params),
            lr=params.get("lr", default_params["lr"]),
            model_name=params.get("model_name", default_params["model_name"]),
            num_filters=params.get("num_filters", default_params["num_filters"]),
            num_depth=params.get("num_depth", default_params["num_depth"]),
            activation_fn=params.get("activation_fn", default_params["activation_fn"]),
            loss_fn=params.get("loss_fn", default_params["loss_fn"]),
            batch_size=params.get("batch_size", default_params["batch_size"]),
            num_workers=params.get("num_workers", default_params["num_workers"]),
            max_epochs=params.get("max_epochs", default_params["max_epochs"]),
            optimizer_name=params.get("optimizer_name", default_params["optimizer_name"]),
            scaler_name=params.get("scaler_name", default_params["scaler_name"]),
            bce_weight=params.get("bce_weight", default_params["bce_weight"]),
            focal_gamma=params.get("focal_gamma", default_params["focal_gamma"]),
        )

        

        result = {
            "params": params,
            "f1_score": f1_score,
            "run_time": run_time,
        }
        all_results.append(result)
    
    # Save results
    with open(EXPERIMENT+"results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    
    # Print the best hyperparameters
    best_result = max(all_results, key=lambda x: x["f1_score"])
    print("Best hyperparameters:")
    print(best_result["params"])
    print(f"F1 score: {best_result['f1_score']}")
    print(f"Run time: {best_result['run_time']:.2f} seconds")

    return


if __name__ == "__main__":
    main()
