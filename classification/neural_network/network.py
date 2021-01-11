import numpy as np
from neupy import layers, algorithms
from neupy.exceptions import StopTraining
from .load_neupy import ConjugateGradient_Custom

# Stop training if last 5 epoch have differences less than 0.0001
def on_epoch_end(optimizer):
    epoch = optimizer.last_epoch
    errors = optimizer.errors.train
    if epoch - 5 >= 0:
        if (
            np.absolute(errors[epoch - 5] - errors[epoch - 4]) < 0.0001
            and np.absolute(errors[epoch - 4] - errors[epoch - 3]) < 0.0001
            and np.absolute(errors[epoch - 3] - errors[epoch - 2]) < 0.0001
            and np.absolute(errors[epoch - 2] - errors[epoch - 1]) < 0.0001
        ):
            raise StopTraining("Stopping condition fulfilled")


def initialize_model(n_features, n_classes, neurons, bp_prop, show_logs):
    network = layers.join(layers.Input(n_features))
    for i in neurons:
        network = layers.join(network, layers.Sigmoid(i))
    network = layers.join(network, layers.Sigmoid(n_classes))
    if isinstance(bp_prop, float):
        model = algorithms.GradientDescent(
            batch_size=None,
            network=network,
            step=bp_prop,
            verbose=show_logs,
            signals=on_epoch_end,
        )
    else:
        model = ConjugateGradient_Custom(
            network=network,
            update_function=bp_prop,
            verbose=show_logs,
            signals=on_epoch_end,
        )
    return model
