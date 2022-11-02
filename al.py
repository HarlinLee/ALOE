"""
Active learning with features extracted from pretrained model.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from argparse import ArgumentParser
import numpy as np
import utils

def main(args):
    acq_methods = ["uncertainty", "random"]

    print("Loading data...", flush=True)
    train_data = np.load(os.path.join(args.features_dir, f"train_{args.dataset}.npz"))
    train_X = train_data["features"]
    train_labels = train_data["labels"]

    validation_data = np.load(os.path.join(args.features_dir, f"validation_{args.dataset}.npz"))
    validation_X = validation_data["features"]
    validation_labels = validation_data["labels"]

    test_data = np.load(os.path.join(args.features_dir, f"test_{args.dataset}.npz"))
    test_X = test_data["features"]
    test_labels = test_data["labels"]

    X = np.concatenate((train_X, validation_X), axis=0)
    labels = np.concatenate((train_labels, validation_labels), axis=0)
    n_classes = len(np.unique(labels))

    # Create train and validation masks
    train_mask = np.zeros(X.shape[0], dtype=bool)
    train_mask[:train_X.shape[0]] = True
    validation_mask = np.zeros(X.shape[0], dtype=bool)
    validation_mask[train_X.shape[0]:] = True

    train_idx_all = np.where(train_mask)[0]

    for acq in acq_methods:
        print(f"Acquisition Function = {acq.upper()}", flush=True)

        # Select initial training set -- should be same for each method
        train_ind = np.array([], dtype=np.int16)
        for c in np.sort(np.unique(labels)):
            # Ensure the chosen points are in the correct subset of the dataset
            c_ind = np.intersect1d(np.where(labels == c)[0], train_idx_all) 
            rng = np.random.default_rng(args.seed) # For reproducibility
            train_ind = np.append(train_ind, rng.choice(c_ind, args.num_per_class, replace=False))

        # Run active learing with current acqusition function
        _, _, model = utils.active_learning_loop(X, n_classes, 
            train_ind=train_ind, labels=labels, num_iter=args.iters, method=acq, 
            train_idx_all=train_idx_all, validation_mask=validation_mask, by_class=args.by_class)
        
        _, test_accuracy = model.evaluate(test_X, test_labels, verbose=0)
        print(f"Test set accuracy: {test_accuracy}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="ALOE.")
    parser.add_argument("--features_dir", type=str, default="features", 
        help="directory containing audio representations")
    parser.add_argument("--dataset", type=str, default="speech_commands", 
        help="dataset to run active learning on")
    parser.add_argument("--by_class", default=False, action="store_true", 
        help="flag to indicate if select n examples per (pseudo) \
        class labeling at each active learning iteration")
    parser.add_argument("--num_per_class", type=int, default=5, 
        help="number of initially labeled points per class")
    parser.add_argument("--iters", type=int, default=100, 
        help="number of active learning iterations")
    parser.add_argument("--seed", type=int, default=1, 
        help="random seed.")
    args = parser.parse_args()

    main(args)
