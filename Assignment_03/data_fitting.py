# %%
from pylab import *
from scipy.linalg import lstsq
from scipy.special import jn

# %% [markdown]
# ##### Constants

# %%
N = 101  # no of data points
K = 9    # no of sets of data with varying noise

A_true, B_true = 1.05, -0.105 # true values of A and B

DATAFILE = "fitting.dat"

# %% [markdown]
# ### 1. Generate the data points

# %%
""" run generate_data.py to generate the data """

# %% [markdown]
# ### 2. Load data

# %%
raw_data = loadtxt(DATAFILE)

Time = raw_data[:, 0]
F = raw_data[:, 1:]

# %% [markdown]
# ### 3. The function and the noise

# %%
def g(t, A=A_true, B=B_true):
    return A * jn(2, t) + B * t


F_true = g(Time)
Sigma = logspace(-1, -3, K)  # vector of stdevs of noise

# %% [markdown]
# ### 4. Plot the data

# %%
figure(figsize=(9, 7))
grid(True)
title("Q4: Data to be fitted to theory", size=16)
xlabel("$t$   $\longrightarrow$", size=16)
ylabel("$f(t)+noise$   $\longrightarrow$", size=16)
plot(Time, F)
plot(Time, F_true, color='black', lw=2)
legend([f"$\sigma_{i + 1}$ = {s:.3f}" for i, s in enumerate(Sigma)] + ["True Value"])
show()

# %% [markdown]
# ### 5. Plot the first column with error bars

# %%
figure(figsize=(9, 7))
grid(True)
title("Q5: Data points for $\sigma = 0.10$ along with exact function", size=16)
xlabel("$t$   $\longrightarrow$", size=16)
ylabel("$f(t)+noise$   $\longrightarrow$", size=16)
errorbar(Time[::5], F[::5, 0], Sigma[0], fmt="ro")
plot(Time, F_true, color='black', lw=2)
legend(["$f(t)$", "Errorbar"])
show()

# %% [markdown]
# ### 6. Confirm that the two vectors are equal

# %%
M = c_[jn(2, Time), Time]
assert allclose(F_true.reshape(N, 1), matmul(M, [[A_true], [B_true]]))

# %% [markdown]
# ### 7. Mean Squared Error for various A and B

# %%
# 0 to 2 in steps of 0.1 (including endpoint)
numA = int((2 - 0) / 0.1) + 1
A = linspace(0, 2, numA)

# -0.2 to 0 in steps of 0.01 (including endpoint)
numB = int((0 - -0.2) / 0.01) + 1
B = linspace(-0.2, 0, numB)

eps = zeros((numA, numB))
for i in range(numA):
    for j in range(numB):
        eps[i][j] = mean((F[:, 0] - g(Time, A[i], B[j])) ** 2)

# %% [markdown]
# ### 8. Plot the MSE

# %%
figure(figsize=(9, 7))
grid(True)
title("Q8: Contour plot of $\epsilon_{ij}$", size=16)
xlabel("$A$   $\longrightarrow$", size=16)
ylabel("$B$   $\longrightarrow$", size=16)
clabel(contour(A, B, eps, 15))
plot([A_true], [B_true], "ro")
annotate("Exact location", xy=(A_true, B_true), size=16)
show()

# %% [markdown]
# ### 9. Obtain best estimate of A and B

# %%
print("Best estimate:   A = {}, B = {}".format(*lstsq(M, F[:, 0])[0]))

# %% [markdown]
# ### 10. Plot the error in A and B for different stdev of noise

# %%
Aerr, Berr = abs(lstsq(M, F)[0] - [[A_true], [B_true]])

figure(figsize=(9, 7))
grid(True)
title("Q10: Variation of error with noise", size=16)
xlabel("$Noise$ $standard$ $deviation$   $\longrightarrow$", size=16)
ylabel("$Error$   $\longrightarrow$", size=16)
plot(Sigma, Aerr, 'o', linestyle="dashed")
plot(Sigma, Berr, 'o', linestyle="dashed")
legend(["Aerr", "Berr"])
show()

# %% [markdown]
# ### 11. Replot using log-log scale

# %%
figure(figsize=(9, 7))
grid(True)
title("Q11: Variation of error with noise", size=16)
xlabel("$\sigma_n$   $\longrightarrow$", size=16)
ylabel("$Error$   $\longrightarrow$", size=16)
xscale("log")
yscale("log")
errorbar(Sigma, Aerr, Sigma, fmt="o")
errorbar(Sigma, Berr, Sigma, fmt="o")
legend(["Aerr", "Berr"])
show()


