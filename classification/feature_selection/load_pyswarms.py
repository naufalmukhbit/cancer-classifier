import numpy as np
import pyswarms as ps
from ..utils import calculate_performance

def f(x, network, X, y, alpha=0.88):
    objective = np.empty(x.shape[0])

    # Calculate objective function per-particle
    for i in range(x.shape[0]):
        P = calculate_performance(
            y.argmax(axis=1), network.predict(X * x[i]).argmax(axis=1)
        )
        objective[i] = (alpha * (1.0 - P)) + (
            (1.0 - alpha) * (np.count_nonzero(x[i]) / X.shape[1])
        )

    return objective


def reduce_dimensionality(network, X, y, epochs, n_particles, options):
    optimizer = ps.discrete.BinaryPSO(
        n_particles=n_particles, dimensions=X.shape[1], options=options
    )

    # Perform optimization
    cost, pos = optimizer.optimize(f, network=network, X=X, y=y, iters=epochs)
    return pos
