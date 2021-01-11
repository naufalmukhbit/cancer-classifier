import numpy as np
from copy import deepcopy

import dataset
from classification.neural_network import initialize_model
from classification.feature_selection import reduce_dimensionality
from classification.utils import k_validation_split, split_output, calculate_performance

def evaluate_algorithm(
    k, data, bpso_options, n_particles, bp_prop, neurons, show_logs
):
    folds = k_validation_split(k, data)
    for i in range(len(folds)):
        print("Iteration", i + 1, "of", len(folds), "folds.")
        trainset = deepcopy(folds)
        testset = trainset.pop(i)
        trainset = np.concatenate(tuple([i for i in trainset]))

        train_X, train_y = split_output(trainset)
        test_X, test_y = split_output(np.array(testset))

        model = initialize_model(
            train_X.shape[1], train_y.shape[1], neurons, bp_prop, False
        )
        model.train(train_X, train_y, epochs=1000)
        pos = reduce_dimensionality(
            model,
            test_X,
            test_y,
            epochs=100,
            n_particles=n_particles,
            options=bpso_options,
        )

        new_model = initialize_model(
            np.count_nonzero(pos), train_y.shape[1], neurons, bp_prop, show_logs
        )
        train_X = train_X[:, pos == 1]
        test_X = test_X[:, pos == 1]

        new_model.train(train_X, train_y, epochs=100)

        y_predicted = new_model.predict(test_X).argmax(axis=1)
        print("Predicted :", y_predicted)
        print("Actual    :", test_y.argmax(axis=1))
        print("Selected features:", np.count_nonzero(pos))
        print()
        scores.append(calculate_performance(test_y.argmax(axis=1), y_predicted))
        feats.append(np.count_nonzero(pos))
    return scores, feats


if __name__ == "__main__":
    data = np.concatenate(dataset.colon_tumor())
    # data = np.concatenate(dataset.lung_cancer())
    # data = np.concatenate(dataset.breast_cancer())
    # data = np.concatenate(dataset.ovarian_cancer())
    # data = np.concatenate(dataset.prostate_cancer())


    # c1: Cognitive Parameter
    # c2: Social Parameter
    # w: Inertia Parameter
    # k: Number of neighbors
    # p: P-norm to use. 1 for sum-of-absolute, 2 for Euclidean distance
    bpso_options = {"c1": 1, "c2": 2, "w": 0.5, "k": 50, "p": 1}

    # # Use Learning Rate for Standard BP
    # bp_prop = 0.5

    # Use CG Method for CG
    cg_methods = ["powell_beale", "flecther_reeves", "polak_ribiere"]
    bp_prop = cg_methods[0]

    scores, feats = evaluate_algorithm(
        5,
        data,
        bpso_options=bpso_options,
        n_particles=50,
        bp_prop=bp_prop,
        neurons=[45],
        show_logs=True,
    )
    print("Scores            :", scores)
    print("Selected features :", feats)
    print("Final score       :", (sum(scores) / len(scores)) * 100)