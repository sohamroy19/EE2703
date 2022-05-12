import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt


x = np.random.rand(100)
X = fft(x)
y = ifft(X)
np.c_[x, y]
print("Maximum Absolute Error for Random Data: ", np.abs(x - y).max())


x = np.linspace(0, 2 * np.pi, 128)
y = np.sin(5 * x)
Y = fft(y)

plt.figure()
plt.subplot(2, 1, 1)
plt.title("Spectrum of $\sin(5t)$ without Phase Wrapping")
plt.ylabel("$|Y|$")
plt.plot(np.abs(Y), lw=2)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.xlabel("$\omega$")
plt.ylabel("Phase of $Y$")
plt.plot(np.unwrap(np.angle(Y)), lw=2)
plt.grid(True)

plt.savefig("Assignment_08/LaTeX/eg1.png")


def plotter(w, Y, title, lim, out, xlabel="$\omega$", ylabels=("$|Y|$", r"$\angle Y$")):
    plt.figure(figsize=(8, 10))
    plt.subplot(2, 1, 1)
    plt.title(title, size=16)
    plt.ylabel(ylabels[0], size=14)
    plt.plot(w, abs(Y), lw=2)
    plt.xlim(-lim, lim)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.xlabel(xlabel, size=14)
    plt.ylabel(ylabels[1], size=14)
    ii = np.where(abs(Y) > 1e-3)
    plt.plot(w[ii], np.unwrap(np.angle(Y[ii])), "go", lw=2)
    plt.xlim(-lim, lim)
    plt.grid(True)

    plt.savefig("Assignment_08/LaTeX/" + out)


x = np.linspace(0, 2 * np.pi, 128, endpoint=False)
w = np.linspace(-64, 64, 128, endpoint=False)

Y = fftshift(fft(np.sin(5 * x))) / 128
plotter(w, Y, "Spectrum of $\sin(5t)$", 10, "eg2", "$k$")

Y = fftshift(fft((1 + 0.1 * np.cos(x)) * np.cos(10 * x))) / 128
plotter(w, Y, "Spectrum of $(1 + 0.1\cos(t))\cdot\cos(10t)$", 15, "eg3")


x = np.linspace(-4 * np.pi, 4 * np.pi, 512, endpoint=False)
w = np.linspace(-64, 64, 512, endpoint=False)

Y = fftshift(fft((1 + 0.1 * np.cos(x)) * np.cos(10 * x))) / 512
plotter(w, Y, "Spectrum of $(1 + 0.1\cos(t))\cdot\cos(10t)$ (Improved)", 15, "eg4")

Y = fftshift(fft(np.sin(x) ** 3)) / 512
plotter(w, Y, "Spectrum of $\sin^3(t)$", 15, "q2a")

Y = fftshift(fft(np.cos(x) ** 3)) / 512
plotter(w, Y, "Spectrum of $\cos^3(t)$", 15, "q2b")

Y = fftshift(fft(np.cos(20 * x + 5 * np.cos(x)))) / 512
plotter(w, Y, "Spectrum of $\cos(20t + 5\cos(t))$", 30, "q3")


T = 2 * np.pi
N = 128
iter_n = 0
error = None
threshold = 1e-6  # 6 decimals of precision

while error is None or error > threshold:
    t = np.linspace(-T / 2, T / 2, N, endpoint=False)
    w = np.linspace(-np.pi, np.pi, N, endpoint=False) * N / T
    y = np.exp(-0.5 * t ** 2)
    Y = fftshift(fft(ifftshift(y))) * T / (2 * np.pi * N)

    Y_true = np.exp(-0.5 * w ** 2) / np.sqrt(2 * np.pi)
    error = np.max(np.abs(Y - Y_true))

    T *= 2
    N *= 2
    iter_n += 1
    print(f"Iteration {iter_n}:   Total Error = {error:.2e}")

T /= 2
N /= 2

print(f"Samples = {int(N)},   Time Period = {int(T / np.pi)} pi")

plotter(w, Y, "Spectrum of Approximated Gaussian", 5, "q4a")
plotter(w, Y_true, "Spectrum of True Gaussian", 5, "q4b")


plt.show()
