"""
Helper functions for active learning. 
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np
import graphlearning as gl
import tensorflow as tf

def create_classifier(input_shape, n_classes, learning_rate=0.001):
    """
    Creates a linear classification model to be trained using features from a pretrained encoder.
    Params:
        - input_shape: shape of numpy array of features extracted from a pretrained model.
        - n_classes: number of classes
    Output:
        - model: tf.keras.model.Model object, trained linear classifier
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(n_classes, activation=None))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

def acquisition_function(candidate_inds, u):
    """
    Main function for computing acquisition function values. All available methods are callable from this function.
    Params:
        - candidate_inds: (N,) numpy array, indices to calculate the acquisition function on. 
        - u: (N x n_c) numpy array, current classifier output.
    Output:
        - vals: (len(candidate_inds), ) numpy array, acquisition function values on the specified candidate_inds
    """
    # Smallest margin acquistion
    u_sort = np.sort(u[candidate_inds])
    vals = 1.-(u_sort[:,-1] - u_sort[:,-2])
    return vals

def active_learning_loop(X, n_classes, train_ind, labels, num_iter, 
    method, train_idx_all=None, validation_mask=None, by_class=False,
    epochs=100):
    """
    Function for handling overall active learning iteration process 
        (a) compute/update classifier and (b) select query points via acquisition function values.
    Adapted from: https://github.com/jwcalder/MSTAR-Active-Learning
    Params:
        - X: (N x d) numpy array, data matrix
        - n_classes: int, number of classes
        - train_ind: (num_init_labeled,) numpy array, initially labeled index set
        - labels: (num_init_labeled,) numpy array, labels (classifications) for initially labeled index set, train_ind
        - num_iter: int, number of active learning iterations to perform
        - method: str, string to specify which acquisition function to use for selecting the query set
        - train_idx_all: (-1,) numpy array, indices of possible points to choose for active learning.
        - validation_mask: (N,) boolean mask, identifies indices of validation set points for evaluating 
            the accuracy of the model at each iteration
        - by_class: bool, flag to indicate whether or not to sample points sequentially regardless of class 
            labeling (by_class=False) or in a batch with one per class (by_class=True) at each iteration.
        - epochs: int, number epochs for training linear classifier.
    Output:
        - train_ind: (num_init_labeled + num_iter,) numpy array, indices of all labeled points resulting from 
            active learning process. (Includes initially labeled points' indices)
        - accuracy: (num_iter + 1,) numpy array, accuracies of model at each step of the active learning 
            process (including prior to any active learning queries)
    """

    assert method in ["random","uncertainty"]

    if train_idx_all is None:
        print("WARNING: You have set train_idx_all to None, which assumes ALL points are available \
            for active learning queries.")

    # Instantiate accuracy array
    al_accuracy = np.array([])

    # Run active learning iterations
    for i in range(num_iter + 1):
        if i > 0:
            if train_idx_all is None: # Assume that all unlabeled indices are potential candidates for active learning
                candidate_inds = np.delete(np.arange(len(labels)), train_ind)
            else:  # Candidate inds must be subset of given "train_idx_all"
                candidate_inds = np.setdiff1d(train_idx_all, train_ind)

            # Acquisition function calculation
            if method == "random":
                if by_class:
                    pseudo_labels = pred_labels[candidate_inds]
                    train_ind = np.append(train_ind, candidate_inds[gl.trainsets.generate(pseudo_labels, rate=1)])
                else:
                    train_ind = np.append(train_ind, np.random.choice(candidate_inds))
            else:
                obj_vals = acquisition_function(candidate_inds, u)
                if by_class:
                    pseudo_labels = pred_labels[candidate_inds]
                    new_train_inds = np.array([], dtype=int)
                    for c in np.unique(pseudo_labels):
                        c_mask = pseudo_labels == c
                        new_train_inds = np.append(new_train_inds, 
                            (candidate_inds[c_mask])[np.argmax(obj_vals[c_mask])])
                    train_ind = np.append(train_ind, new_train_inds)
                else:
                    new_train_ind = candidate_inds[np.argmax(obj_vals)]
                    train_ind = np.append(train_ind, new_train_ind)

        tf.keras.backend.clear_session()
        model = create_classifier(input_shape=X.shape[1], n_classes=n_classes)
        model.fit(X[train_ind,:], labels[train_ind], 
            epochs=epochs, verbose=0)
        u = model.predict(X, verbose=0)
        pred_labels = np.argmax(u, axis=1)

        if validation_mask is None:
            accuracy = gl.ssl.ssl_accuracy(pred_labels, labels, len(train_ind))
        else:
           accuracy = np.mean(labels[validation_mask] == pred_labels[validation_mask])
        al_accuracy = np.append(al_accuracy, accuracy)

    return train_ind, al_accuracy, model
    