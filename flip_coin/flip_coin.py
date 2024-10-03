import numpy as np
import matplotlib.pyplot as plt

# define prior probabilities: p(C)
C0 = 0.4
C1 = 0.6
# This means we assume the coin has a 60% chance of being biased, and has a 40% chance of being fair.
# We also suppose that a biased coin has an 80% chance of flipping to heads. (that is why line 20 has a 0.8)

# the number of heads
num = 7

# define loss function: λ
Lambda00 = 0
Lambda01 = 10
Lambda10 = 5
Lambda11 = 0

# define likelihood function: p(x|C)
def likelihood(num, theta):
    return theta**num * (1-theta)**(10-num)

# calculate posterior probabilities: p(C|x)
def posterior(num, C0, C1):
    likelihood_0 = likelihood(num, 0.5)
    likelihood_1 = likelihood(num, 0.8)
    numerator_0 = likelihood_0 * C0
    numerator_1 = likelihood_1 * C1

    # p(x) = p(x|C=1)*p(C=1) + p(x|C=0)*p(C=0)
    evidence = numerator_1 + numerator_0
    
    # p(C|x) = p(C)*p(x|C) / p(x)
    posterior_0 = numerator_0 / evidence
    posterior_1 = numerator_1 / evidence
    return posterior_0, posterior_1

# calculate expected risk: R(a|x)
def risk(L1, L2, p1, p2):
    return L1*p1+L2*p2

if __name__ == "__main__":
    posterior_0, posterior_1 = posterior(num, C0, C1)
    print(f"Posterior probability of fair coin: {posterior_0:.3f}")
    print(f"Posterior probability of biased coin: {posterior_1:.3f}")
    print()

    print("According to the table of loss function...")
    risk_0 = risk(Lambda00, Lambda01, posterior_0, posterior_1)
    risk_1 = risk(Lambda10, Lambda11, posterior_0, posterior_1)
    print(f"Expected risk of fair coin: {risk_0:.3f}")
    print(f"Expected risk of biased coin: {risk_1:.3f}")

    # plot classifier
    x_values = np.arange(11)
    posterior_1_values = [posterior(x, C0, C1)[1] for x in x_values]
    plt.plot(x_values, posterior_1_values, label="Posterior probability of biased coin")
    plt.axhline(y=0.5, color="r", linestyle="--", label="Decision boundary")
    plt.xlabel("Number of heads")
    plt.ylabel("Posterior probability")
    plt.legend()
    plt.show()