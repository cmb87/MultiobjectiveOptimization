import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def coordinatesSplitter(X):
    if X.ndim == 2:
        x, y = X[:, 0], X[:, 1]
    elif X.ndim == 1:
        x, y = X[0], X[1]
    else:
        logging.info("Datatype not understood")
        return
    return x, y


def rosenbrock(X):
    x, y = coordinatesSplitter(X)
    a, b = 1, 100

    rspns = np.zeros((X.shape[0], 2))
    rspns[:, 0] = (a - x) ** 2 + b * (y - x ** 2) ** 2
    rspns[:, 1] = ((x - 2) ** 2 + (y + 2) ** 2) ** 0.5
    return rspns


def rosenbrockContour(imax=30, jmax=30, xbounds=[(-2, 2), (-2, 2)]):
    x = np.linspace(xbounds[0][0], xbounds[0][1], imax)
    y = np.linspace(xbounds[1][0], xbounds[1][1], jmax)
    xx, yy = np.meshgrid(y, x)
    rspns = rosenbrock(np.vstack((xx.reshape(-1), yy.reshape(-1))).T)

    # [:,0].reshape(imax,jmax)
    return [xx, yy, rspns[:, 0].reshape(imax, jmax)], [[0, 0]]


def rosenbrockContourConstrained(
    imax=30, jmax=30, xbounds=[(-2, 2), (-2, 2)], cradius=1.3
):
    x = np.linspace(xbounds[0][0], xbounds[0][1], imax)
    y = np.linspace(xbounds[1][0], xbounds[1][1], jmax)
    xx, yy = np.meshgrid(y, x)
    rspns = rosenbrock(np.vstack((xx.reshape(-1), yy.reshape(-1))).T)

    phi = np.linspace(0, 2 * np.pi, 30)
    return (
        [xx, yy, rspns[:, 0].reshape(imax, jmax)],
        [[2 + cradius * np.cos(phi), -2 + cradius * np.sin(phi)]],
    )


def binhAndKorn(X):
    x, y = coordinatesSplitter(X)
    rspns = np.zeros((X.shape[0], 4))

    rspns[:, 0] = 4 * x ** 2 + 4 * y ** 2
    rspns[:, 1] = (x - 5) ** 2 + (y - 5) ** 2

    rspns[:, 2] = (x - 5) ** 2 + y ** 2
    rspns[:, 3] = (x - 8) ** 2 + (y + 3) ** 2

    return rspns


# Visualization ###
def animateSwarm(Xcine, Ycine, contourfct, xbounds, store=False):

    itermax = Xcine.shape[0]
    [X, Y, Z], constraints = contourfct(xbounds=xbounds)

    fig = plt.figure()
    ax = plt.axes()

    sc = ax.contourf(X, Y, Z, 30)
    ax.contour(X, Y, Z, 30, colors="k", linewidths=1)
    plt.colorbar(sc)

    for const in constraints:
        plt.plot(const[0], const[1], "r--")

    lines, colors = {}, plt.cm.get_cmap("RdBu")(np.linspace(0, 1, Xcine.shape[1]))
    lines["title"] = ax.text(X.mean(), Y.max(), "Iter: 0")
    for color, p in zip(colors, range(Xcine.shape[1])):
        lines["particleBody_{}".format(p)] = ax.plot(
            Xcine[0, p, 0], Xcine[0, p, 1], "-", color=color
        )[0]
        lines["particleHead_{}".format(p)] = ax.plot(
            Xcine[0, p, 0], Xcine[0, p, 1], "^", color=color
        )[0]

    def update(i, x, lines):
        lines["title"].set_text("Iter: {}".format(i + 1))
        for p in range(x.shape[1]):
            lines["particleBody_{}".format(p)].set_data(
                [x[i, p, 0], x[i + 1, p, 0]], [x[i, p, 1], x[i + 1, p, 1]]
            )
            lines["particleHead_{}".format(p)].set_data(x[i + 1, p, 0], x[i + 1, p, 1])

    ani = animation.FuncAnimation(
        fig,
        update,
        (itermax - 1),
        fargs=(Xcine, lines),
        interval=10 * (itermax - 1),
        blit=False,
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    if store:
        ani.save(
            os.path.join("./", "swarm_opti_{}.gif".format(1)), writer="mencoder", fps=15
        )
        plt.close()
    else:
        plt.show()


# Visualization ###
def animateSwarm2(Xcine, Ycine, xbounds, ybounds, store=False):

    itermax = Xcine.shape[0]
    fig = plt.figure()
    ax = plt.axes(xlim=ybounds[0], ylim=ybounds[1])

    lines, colors = {}, plt.cm.get_cmap("RdBu")(np.linspace(0, 1, Ycine.shape[1]))
    lines["title"] = ax.text(Ycine[0, :, 0].mean(), Ycine[0, :, 1].max(), "Iter: 0")
    for color, p in zip(colors, range(Xcine.shape[1])):
        lines["particleBody_{}".format(p)] = ax.plot(
            Ycine[0, p, 0], Ycine[0, p, 1], "-", color=color
        )[0]
        lines["particleHead_{}".format(p)] = ax.plot(
            Ycine[0, p, 0], Ycine[0, p, 1], "^", color=color
        )[0]

    def update(i, x, lines):
        lines["title"].set_text("Iter: {}".format(i + 1))
        for p in range(x.shape[1]):
            lines["particleBody_{}".format(p)].set_data(
                [x[i, p, 0], x[i + 1, p, 0]], [x[i, p, 1], x[i + 1, p, 1]]
            )
            lines["particleHead_{}".format(p)].set_data(x[i + 1, p, 0], x[i + 1, p, 1])

    ani = animation.FuncAnimation(
        fig,
        update,
        (itermax - 1),
        fargs=(Ycine, lines),
        interval=10 * (itermax - 1),
        blit=False,
    )
    # ani.save(os.path.join(self.dir,f"modal_result_mode_{mid}.gif"),
    # writer = 'mencoder', fps=15)
    # plt.close()
    plt.xlabel("F1(x)")
    plt.ylabel("F2(x)")
    plt.grid(True)
    if store:
        ani.save(
            os.path.join("./", "swarm_opti_{}.gif".format(1)), writer="mencoder", fps=15
        )
        plt.close()
    else:
        plt.show()
