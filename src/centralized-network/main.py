from ray import tune
from functools import partial
from train import train, test_accuracy
from resnet50 import ResNet
from data.provider import get_train_validation_data
from data.augmentation import get_transform
from ray.tune.schedulers import ASHAScheduler
import torch, torch.nn as nn
from ray.tune import CLIReporter
import os
from argparse import ArgumentParser



def main(normalization, max_num_epochs = 15, num_samples=9):
    """
    train / eval loop for HP tuning
    using transform on eval / test sets 
    an other module will take care of the network training loop
    """

    # we will see later for momentum as it requires more advanced config space and I'm not super familiar with raytune grid search
    config = {
        "optimizer" : tune.grid_search(["SGD", "Adam"]),
        "lr" : tune.loguniform(1e-4, 1e-2),
        "weightdecay" : tune.uniform(0, 0.1)
    } # maybe change some things here



    scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)
        
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])

    transform = get_transform()

    

    trainloader, valloader = get_train_validation_data(transform=transform) # change parameters when data augmentation pipeline will be done

    result = tune.run(
        tune.with_parameters(train, trainloader=trainloader, valloader=valloader, n_epochs=max_num_epochs, normalization=normalization),
        resources_per_trial={"cpu": 2, "gpu": 1},
        num_samples=num_samples,
        config=config,
        scheduler=scheduler,
        progress_reporter=reporter

    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]))

    best_trained_model = ResNet(normalization=normalization)
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, transform=transform)
    print("Best trial test set accuracy: {}".format(test_acc))

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--normalization", required=True, type=str, choices=["batch", "group"])
    args = parser.parse_args()

    main(args.normalization)