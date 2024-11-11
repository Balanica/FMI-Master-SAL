from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from numpy.random import uniform
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


def ex2(dimension):
    box = ()
    if dimension == 2:
        box = ((10, 0), (0, 10))
    elif dimension == 3:
        box = ((0, 10, 0), (10, 0, 10))

    X, y = make_blobs(n_samples=500, n_features=dimension, cluster_std=1.0, centers=2, center_box=box)

    # print(X)
    # print(y)
    neurons = [[32, 16], [64, 32], [128, 64]]
    bins = [10, 20, 30]
    for i in range(3):
        iforest = IForest(contamination=0.02)
        loda = LODA(contamination=0.02, n_bins=bins[i])
        dif = DIF(contamination=0.02, hidden_neurons=neurons[i])

        iforest.fit(X)
        loda.fit(X)
        dif.fit(X)

        ud = uniform(low=-10, high=20, size=(1000, dimension))

        score_iForest = iforest.decision_function(ud)
        score_dif = dif.decision_function(ud)
        score_loda = loda.decision_function(ud)

        if dimension == 2:
            fig, ax = plt.subplots(nrows=1, ncols=3)
            ax[0].scatter(ud[:, 0], ud[:, 1], c=score_iForest, cmap=plt.cm.RdYlBu)
            ax[0].title.set_text('IForest')
            ax[1].scatter(ud[:, 0], ud[:, 1], c=score_dif, cmap=plt.cm.RdYlBu)
            ax[1].title.set_text('DIF')
            ax[2].scatter(ud[:, 0], ud[:, 1], c=score_loda, cmap=plt.cm.RdYlBu)
            ax[2].title.set_text('LODA')
        elif dimension == 3:
            fig, ax = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(0, 20))
            # fig = plt.figure(figsize=(12, 12))
            # ax = fig.add_subplot(nrows=1, ncols=3, projection = '3d')
            ax[0].scatter(ud[:, 0], ud[:, 1], ud[:, 2], c=score_iForest, cmap=plt.cm.RdYlBu)
            ax[0].title.set_text('IForest')
            ax[1].scatter(ud[:, 0], ud[:, 1], ud[:, 2], c=score_dif, cmap=plt.cm.RdYlBu)
            ax[1].title.set_text('DIF')
            ax[2].scatter(ud[:, 0], ud[:, 1], ud[:, 2], c=score_loda, cmap=plt.cm.RdYlBu)
            ax[2].title.set_text('LODA')
        plt.show()


# add dimension 2 for 2D or 3 for 3D
ex2(2)
