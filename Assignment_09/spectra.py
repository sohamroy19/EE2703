import numpy as np
from numpy.fft import fft, fftshift

import matplotlib.pyplot as plt
from matplotlib import cm


""" Examples """

t = np.linspace(-np.pi, np.pi, 64, endpoint=False)
dt = t[1] - t[0]
f_max = 1 / dt
y = np.sin(np.sqrt(2) * t)
y[0] = 0  # the sample corresponding to -tmax should be set zero
y = fftshift(y)  # make y start with y(t=0)
Y = fftshift(fft(y)) / 64
w = np.linspace(-np.pi * f_max, np.pi * f_max, 64, endpoint=False)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w, np.abs(Y), lw=2)
plt.xlim([-10, 10])
plt.ylabel(r"$|Y|$", size=16)
plt.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)$")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y), "ro", lw=2)
plt.xlim([-10, 10])
plt.ylabel(r"Phase of $Y$", size=16)
plt.xlabel(r"$\omega$", size=16)
plt.grid(True)
plt.savefig("Assignment_09/LaTeX/eg1.png")

t1 = np.linspace(-np.pi, np.pi, 64, endpoint=False)
t2 = np.linspace(-3 * np.pi, -np.pi, 64, endpoint=False)
t3 = np.linspace(np.pi, 3 * np.pi, 64, endpoint=False)
# y = np.sin(np.sqrt(2) * t)
plt.figure()
plt.plot(t1, np.sin(np.sqrt(2) * t1), "b", lw=2)
plt.plot(t2, np.sin(np.sqrt(2) * t2), "r", lw=2)
plt.plot(t3, np.sin(np.sqrt(2) * t3), "r", lw=2)
plt.ylabel(r"$y$", size=16)
plt.xlabel(r"$t$", size=16)
plt.title(r"$\sin\left(\sqrt{2}t\right)$")
plt.grid(True)
plt.savefig("Assignment_09/LaTeX/eg2.png")

t1 = np.linspace(-np.pi, np.pi, 64, endpoint=False)
t2 = np.linspace(-3 * np.pi, -np.pi, 64, endpoint=False)
t3 = np.linspace(np.pi, 3 * np.pi, 64, endpoint=False)
y = np.sin(np.sqrt(2) * t1)
plt.figure()
plt.plot(t1, y, "bo", lw=2)
plt.plot(t2, y, "ro", lw=2)
plt.plot(t3, y, "ro", lw=2)
plt.ylabel(r"$y$", size=16)
plt.xlabel(r"$t$", size=16)
plt.title(r"$\sin\left(\sqrt{2}t\right)$ with $t$ wrapping every $2\pi$")
plt.grid(True)
plt.savefig("Assignment_09/LaTeX/eg3.png")

t = np.linspace(-np.pi, np.pi, 64, endpoint=False)
dt = t[1] - t[0]
f_max = 1 / dt
y = t
y[0] = 0  # the sample corresponding to -tmax should be set zero
y = fftshift(y)  # make y start with y(t=0)
Y = fftshift(fft(y)) / 64
w = np.linspace(-np.pi * f_max, np.pi * f_max, 64, endpoint=False)
plt.figure()
plt.semilogx(np.abs(w), 20 * np.log10(np.abs(Y)), lw=2)
plt.xlim([1, 10])
plt.ylim([-20, 0])
plt.xticks([1, 2, 5, 10], ["1", "2", "5", "10"], size=16)
plt.ylabel(r"$|Y|$ (dB)", size=16)
plt.title(r"Spectrum of a digital ramp")
plt.xlabel(r"$\omega$", size=16)
plt.grid(True)
plt.savefig("Assignment_09/LaTeX/eg4.png")

t1 = np.linspace(-np.pi, np.pi, 64, endpoint=False)
t2 = np.linspace(-3 * np.pi, -np.pi, 64, endpoint=False)
t3 = np.linspace(np.pi, 3 * np.pi, 64, endpoint=False)
n = np.arange(64)
wnd = fftshift(0.54 + 0.46 * np.cos(2 * np.pi * n / 63))
y = np.sin(np.sqrt(2) * t1) * wnd
plt.figure()
plt.plot(t1, y, "bo", lw=2)
plt.plot(t2, y, "ro", lw=2)
plt.plot(t3, y, "ro", lw=2)
plt.ylabel(r"$y$", size=16)
plt.xlabel(r"$t$", size=16)
plt.title(r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$")
plt.grid(True)
plt.savefig("Assignment_09/LaTeX/eg5.png")

t = np.linspace(-np.pi, np.pi, 64, endpoint=False)
dt = t[1] - t[0]
f_max = 1 / dt
n = np.arange(64)
wnd = fftshift(0.54 + 0.46 * np.cos(2 * np.pi * n / 63))
y = np.sin(np.sqrt(2) * t) * wnd
y[0] = 0  # the sample corresponding to -tmax should be set zero
y = fftshift(y)  # make y start with y(t=0)
Y = fftshift(fft(y)) / 64
w = np.linspace(-np.pi * f_max, np.pi * f_max, 64, endpoint=False)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w, np.abs(Y), lw=2)
plt.xlim([-8, 8])
plt.ylabel(r"$|Y|$", size=16)
plt.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y), "ro", lw=2)
plt.xlim([-8, 8])
plt.ylabel(r"Phase of $Y$", size=16)
plt.xlabel(r"$\omega$", size=16)
plt.grid(True)
plt.savefig("Assignment_09/LaTeX/eg6.png")

t = np.linspace(-4 * np.pi, 4 * np.pi, 256, endpoint=False)
dt = t[1] - t[0]
f_max = 1 / dt
n = np.arange(256)
wnd = fftshift(0.54 + 0.46 * np.cos(2 * np.pi * n / 256))
y = np.sin(np.sqrt(2) * t)
y = y * wnd
y[0] = 0  # the sample corresponding to -tmax should be set zero
y = fftshift(y)  # make y start with y(t=0)
Y = fftshift(fft(y)) / 256
w = np.linspace(-np.pi * f_max, np.pi * f_max, 256, endpoint=False)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(w, np.abs(Y), lw=2)
plt.xlim([-8, 8])
plt.ylabel(r"$|Y|$", size=16)
plt.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(w, np.angle(Y), "ro", lw=2)
plt.xlim([-8, 8])
plt.ylabel(r"Phase of $Y$", size=16)
plt.xlabel(r"$\omega$", size=16)
plt.grid(True)
plt.savefig("Assignment_09/LaTeX/eg7.png")

""" End of Examples """


def spectrum(f, n, lim, windowing=True, t=None):
    """Evaluates the DFT spectrum of a function f(t)."""
    if t is None:
        t, dt = np.linspace(-lim, lim, n, endpoint=False, retstep=True)
    else:
        dt = t[1] - t[0]
    f_max = 1 / dt
    w = np.linspace(-np.pi * f_max, np.pi * f_max, n, endpoint=False)
    y = f(t)
    if windowing:
        y *= fftshift(0.54 + 0.46 * np.cos(2 * np.pi * np.arange(n) / n))
    y[0] = 0  # the sample corresponding to -tmax should be set zero
    y = fftshift(y)  # make y start with y(t=0)
    Y = fftshift(fft(y)) / n

    return w, Y


def plotter(w, Y, title, lim, out, xlabel="$\omega$", ylabels=("$|Y|$", r"$\angle Y$")):
    """Plots the passed DFT spectrum."""
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title(title)
    plt.ylabel(ylabels[0])
    plt.plot(w, np.abs(Y), lw=2)
    plt.xlim(-lim, lim)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabels[1])
    phase = np.angle(Y)
    phase[np.where(np.abs(Y) < 3e-3)] = 0
    plt.plot(w, phase, "ro", lw=2)
    plt.xlim(-lim, lim)
    plt.grid(True)

    plt.savefig("Assignment_09/LaTeX/" + out)


def estimate_params(w, Y):
    """Estimates the parameters omega and delta of cos(omega*t + delta)."""
    ii = np.where(w > 0)
    omega = np.sum(np.abs(Y[ii]) ** 2 * w[ii]) / np.sum(np.abs(Y[ii]) ** 2)
    i = np.argmin(np.abs(w - omega))
    delta = np.angle(Y[i])

    return omega, delta


w, Y = spectrum(lambda t: np.cos(0.86 * t) ** 3, 64 * 4, 4 * np.pi, False)
plotter(w, Y, "Spectrum of $\cos^3(\omega_0 t)$ without a Hamming window", 3, "q2a")

w, Y = spectrum(lambda t: np.cos(0.86 * t) ** 3, 64 * 4, 4 * np.pi)
plotter(w, Y, "Spectrum of $\cos^3(\omega_0 t)$ with a Hamming window", 3, "q2b")

w, Y = spectrum(lambda t: np.cos(1.5 * t + 0.5), 128, np.pi)
plotter(w, Y, "Spectrum of $\cos(\omega_0 t + \delta)$", 3, "q3")
print("Estimated w_0 = {:f}, delta = {:f} without noise".format(*estimate_params(w, Y)))

w, Y = spectrum(lambda t: np.cos(1.5 * t + 0.5) + np.random.randn(128) / 10, 128, np.pi)
plotter(w, Y, "Spectrum of $\cos(\omega_0 t + \delta) with noise$", 3, "q4")
print("Estimated w_0 = {:f}, delta = {:f} with noise".format(*estimate_params(w, Y)))

w, Y = spectrum(lambda t: np.cos(16 * t * (1.5 + t / (2 * np.pi))), 1024, np.pi, False)
plotter(w, Y, "Spectrum of chirped signal without a Hamming window", 60, "q5a")

w, Y = spectrum(lambda t: np.cos(16 * t * (1.5 + t / (2 * np.pi))), 1024, np.pi)
plotter(w, Y, "Spectrum of chirped signal with a Hamming window", 60, "q5b")


Y_mag = np.zeros((16, 64))
for i, t in enumerate(np.split(np.linspace(-np.pi, np.pi, 1024, endpoint=False), 16)):
    w, Y = spectrum(lambda t: np.cos(16 * t * (1.5 + t / (2 * np.pi))), 64, np.pi, False, t)
    Y_mag[i] = np.abs(Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

t, dt = np.linspace(-np.pi, np.pi, 1024, endpoint=False, retstep=True)
f_max = 1 / dt
t = t[::64]
w = np.linspace(-f_max * np.pi, f_max * np.pi, 64, endpoint=False)
t, w = np.meshgrid(t, w)

surf = ax.plot_surface(w, t, Y_mag.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.ylabel("Frequency ($\omega$)")
plt.xlabel("Time ($t$)")
plt.savefig("Assignment_09/LaTeX/q6")


plt.show()
