# import the libraries
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import lstsq


def parse_cmdline_args():
    """Parse command line arguments, define the parameters"""

    try:  # try converting directly to values
        return [
            int(sys.argv[1]) if len(sys.argv) >= 2 else 25,
            int(sys.argv[2]) if len(sys.argv) >= 3 else 25,
            int(sys.argv[3]) if len(sys.argv) >= 4 else 8,
            int(sys.argv[4]) if len(sys.argv) >= 5 else 1500,
        ]
    except ValueError:  # otherwise, parse keyword arguments
        parser = argparse.ArgumentParser(description="Resistor simulation")

        parser.add_argument(
            "-x", "--Nx", type=int, default=25, help="size along x axis"
        )
        parser.add_argument(
            "-y", "--Ny", type=int, default=25, help="size along y axis"
        )
        parser.add_argument(
            "-r", "--radius", type=int, default=8, help="radius of central lead"
        )
        parser.add_argument(
            "-n", "--Niter", type=int, default=1500, help="number of iterations to do"
        )

        args = parser.parse_args()

        return args.Nx, args.Ny, args.radius, args.Niter


def contruct_phi(Nx, Ny, radius):
    """Construct and plot phi, return phi, area inside radius and x,y coordinates"""

    phi = np.zeros((Ny, Nx))  # allocate the potential array
    X, Y = np.meshgrid(np.linspace(-0.5, 0.5, Nx), np.linspace(-0.5, 0.5, Ny))

    # scaled radius = 0.35 for default args, 5% margin for floating point errors
    ii = np.where(X ** 2 + Y ** 2 <= (1.05 * radius / (min(Nx, Ny) - 1)) ** 2)
    phi[ii] = 1.0  # initialize the potential array

    return phi, ii, X, Y


def contour_plot(phi, ii, X, Y):
    """Obtain a contour plot of the potential"""

    plt.figure(figsize=(8, 8))
    plt.title("Contour Plot of the Potential", fontsize=14)
    plt.xlabel("$x$", fontsize=14)
    plt.ylabel("$y$", fontsize=14)
    plt.clabel(plt.contour(X, Y, phi))
    plt.scatter(X[0, ii[1]], Y[ii[0], 0], color="r", label="$V=1$")
    plt.legend()
    plt.grid()
    plt.show()


def iterate(phi, ii, Niter):
    """Perform Niter no. of iterations on phi, return phi and errors"""

    errors = np.zeros(Niter)

    for k in range(Niter):
        oldphi = phi.copy()

        # updating the potential
        phi[1:-1, 1:-1] = (  # Poisson update interior points to average of neighbors
            phi[1:-1, :-2] + phi[1:-1, 2:] + phi[:-2, 1:-1] + phi[2:, 1:-1]
        ) / 4

        # boundary conditions
        phi[1:-1, 0] = phi[1:-1, 1]  # left
        phi[1:-1, -1] = phi[1:-1, -2]  # right
        phi[-1, :] = phi[-2, :]  # top
        phi[ii] = 1.0  # central area corresponding to electrodes to 1

        errors[k] = np.max(np.abs(phi - oldphi))

    return phi, errors


def plot_errors(errors, loglog=False, individual_points=False, plot_fits=False):
    plt.figure(figsize=(8, 8))
    plt.title("Error vs Iteration Number", fontsize=14)
    plt.xlabel("$N_{iter}$", fontsize=14)
    plt.ylabel("$|$error$|$", fontsize=14)

    n, err = np.arange(len(errors)), errors
    if individual_points:
        n, err = n[::50], err[::50]

    if loglog:
        plt.loglog(n, err, "o" if individual_points else "-", label="errors")
    else:
        plt.semilogy(n, err, "o" if individual_points else "-", label="errors")

    if plot_fits:
        fit1 = lstsq(np.c_[np.ones(len(errors)), n], np.log(errors))[0]
        fit2 = lstsq(np.c_[np.ones(len(errors)), n][500:], np.log(errors)[500:])[0]

        plt.semilogy(np.exp(fit1[0] + fit1[1] * n), "r", lw=3, label="fit1")
        plt.semilogy(np.exp(fit2[0] + fit2[1] * n), "g", label="fit2")

    plt.legend()
    plt.grid()
    plt.show()


def surface_plot(phi, X, Y):
    """Do a 3D plot of the potential"""

    ax = plt.figure(figsize=(8, 8)).add_subplot(projection="3d")  # new recommended way
    plt.title("The 3-D surface plot of the potential", fontsize=14)
    plt.xlabel("$x$", fontsize=14)
    plt.ylabel("$y$", fontsize=14)
    # ax = p3.Axes3D(plt.figure(figsize=(8, 8)))  # old way to do 3D plot
    plt.colorbar(ax.plot_surface(X, Y, phi, rstride=1, cstride=1, cmap=plt.cm.jet))
    plt.show()


def plot_currents(phi, X, Y):
    """Obtain the currents"""

    Jx, Jy = np.zeros_like(phi), np.zeros_like(phi)
    Jx[:, 1:-1] = (phi[:, :-2] - phi[:, 2:]) / 2
    Jy[1:-1, :] = (phi[:-2, :] - phi[2:, :]) / 2

    plt.figure(figsize=(8, 8))
    plt.title("Vector Plot of the Current Flow", fontsize=14)
    plt.xlabel("$x$", fontsize=14)
    plt.ylabel("$y$", fontsize=14)
    plt.quiver(X, Y, Jx, Jy, scale=4)
    plt.scatter(X[0, ii[1]], Y[ii[0], 0], color="r", label="$V=1$")
    plt.legend()
    plt.grid()
    plt.show()


Nx, Ny, radius, Niter = parse_cmdline_args()  # define the parameters
phi, ii, X, Y = contruct_phi(Nx, Ny, radius)  # allocate and initialize the potential
contour_plot(phi, ii, X, Y)  # ii denotes points inside the radius

phi, errors = iterate(phi, ii, Niter)  # perform the iteration

plot_errors(errors, loglog=True)  # graph the results
plot_errors(errors)
plot_errors(errors, individual_points=True)
plot_errors(errors, plot_fits=True)

surface_plot(phi, X, Y)  # surface plot of potential
contour_plot(phi, ii, X, Y)  # contour plot of the potential

plot_currents(phi, X, Y)  # vector plot of currents
