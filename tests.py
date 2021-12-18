import numpy as np
import pydot

from trees.SimpleDecisonTree import SimpleDecisionTree as SDT

if __name__ == "__main__":
    tree = SDT(feature_split_intervals=3)

    x = np.array([
        [0, 1],
        [2, 1],
        [1.5, 2],
        [1.3, 1.5],
        [0.8, 1],
        [2, 1.5],
        [5, 4],
        [6, 4],
        [5, 6],
        [6, 6],
    ])
    y = np.array([
        [0],
        [1],
        [0],
        [0],
        [0],
        [1],
        [2],
        [2],
        [2],
        [2],
    ])

    tree.fit(x, y)

    print(tree._feature_interval_class_amount)
    print(tree._feature_average_class_amount)
    print(tree._decision_tree)

    print("TESTS")
    for i, xample in enumerate(x):
        print(tree.predict(xample), y[i])

    for n in tree:
        print(n)

    tree.make_dot()
