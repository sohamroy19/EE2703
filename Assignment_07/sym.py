import sympy as sym
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

s = sym.symbols("s")


def lowpass(Vi, R1=10000, R2=10000, C1=1e-9, C2=1e-9, G=1.586):
    A = sym.Matrix(
        [
            [0, 0, 1, -1 / G],
            [-1 / (1 + s * R2 * C2), 1, 0, 0],
            [0, -G, G, 1],
            [-1 / R1 - 1 / R2 - s * C1, 1 / R2, 0, s * C1],
        ]
    )
    b = sym.Matrix([0, 0, 0, -Vi / R1])
    V = A.inv() * b

    return A, b, V


def highpass(Vi, R1=10000, R3=10000, C1=1e-9, C2=1e-9, G=1.586):
    A = sym.Matrix(
        [
            [0, -1, 0, 1 / G],
            [s * C2 * R3 / (s * C2 * R3 + 1), 0, -1, 0],
            [0, G, -G, 1],
            [-s * C2 - 1 / R1 - s * C1, 0, s * C2, 1 / R1],
        ]
    )
    b = sym.Matrix([0, 0, 0, -Vi * s * C1])
    V = A.inv() * b

    return A, b, V


def to_num_den(expr):
    num, den = expr.as_numer_denom()
    num = [float(i) for i in sym.Poly(num, s).all_coeffs()]
    den = [float(i) for i in sym.Poly(den, s).all_coeffs()]
    return num, den


def plot(title, xlabel, ylabel, x, ys, labels=None, plotting_fn=plt.plot):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for y in ys:
        plotting_fn(x, y)
    if labels is not None:
        plt.legend(labels)
    plt.grid(True)


title = "Lowpass Filter Magnitude Response"

A, b, V = lowpass(1)
V_o = V[3]
print(V_o)
ww = np.logspace(0, 8, 801)
ss = 1j * ww
H = sym.lambdify(s, V_o, "numpy")
HH = H(ss)
plot(title, "$\omega (rad/s)$", "$|H(j\omega)|$", ww, [abs(HH)], plotting_fn=plt.loglog)


title = "Lowpass Filter Step Response"

V_o = lowpass(1 / s)[2][3]
H = sp.lti(*to_num_den(V_o))
t = np.linspace(0, 5e-3, 10000)
v = sp.impulse(H, T=t)[1]
plot(title, "$t (s)$", "$V_o (V)$", t, [v])


title = "Lowpass Filter Input Response"

V_o = lowpass(1)[2][3]
H = sp.lti(*to_num_den(V_o))
t = np.linspace(0, 5e-3, 100000)
input = (np.sin(2000 * np.pi * t) + np.cos(2e6 * np.pi * t)) * (t > 0)
v = sp.lsim(H, U=input, T=t)[1]
plot(title, "$t (s)$", "$V (V)$", t, [input, v], ["Input", "Output"])


title = "Highpass Filter Magnitude Response"

A, b, V = highpass(1)
V_o = V[3]
print(V_o)
ww = np.logspace(0, 8, 801)
ss = 1j * ww
H = sym.lambdify(s, V_o, "numpy")
HH = H(ss)
plot(title, "$\omega (rad/s)$", "$|H(j\omega)|$", ww, [abs(HH)], plotting_fn=plt.loglog)


title = "Highpass Damped Response for High Frequency"

V_o = highpass(1)[2][3]
H = sp.lti(*to_num_den(V_o))
t = np.linspace(0, 1e-4, 1000)
input = np.cos(2 * np.pi * 1e8 * t) * np.exp(-5e4 * t) * (t > 0)
v = sp.lsim(H, U=input, T=t)[1]
plot(title, "$t (s)$", "$V (V)$", t, [input, v], ["Input", "Output"])


title = "Highpass Damped Response for Low Frequency"

t = np.linspace(0, 1, 1000)
input = np.cos(20 * np.pi * t) * np.exp(-5 * t) * (t > 0)
v = sp.lsim(H, U=input, T=t)[1]
plot(title, "$t (s)$", "$V (V)$", t, [input, v], ["Input", "Output"])


title = "Highpass Step Response"

V_o = highpass(1 / s)[2][3]
H = sp.lti(*to_num_den(V_o))
t = np.linspace(0, 5e-3, 10000)
v = sp.impulse(H, T=t)[1]
plot(title, "$t (s)$", "$V_o (V)$", t, [v])

plt.show()
