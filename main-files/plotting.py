import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

def plotting(W, classes, labels=['Politics', 'Sports', 'Sci-Tech', 'Other']):
    plt_colors = ['b', 'c', 'y', 'm']
    n_cls = 4

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt_inds = [0 for i in range(n_cls)]
    for cls in range(n_cls):
        plt_inds[cls] = [i for i, c in enumerate(classes) if int(c)==cls]
        # print(plt_inds[cls])
        ax.scatter(xs=W[plt_inds[cls], 0], ys=W[plt_inds[cls], 1],
            zs=W[plt_inds[cls],2], c=plt_colors[cls], label=labels[cls])

    ax.legend()
    plt.show()

    return 1
