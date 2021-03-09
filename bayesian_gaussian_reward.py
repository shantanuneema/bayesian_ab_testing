import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

NUM_TRIALS = 2000
BANDIT_MEANS = [1, 2, 3]
np.random.seed(1)

class Bandit:

    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.N = 0
        self.predicted_mean = 0
        self.sum_x = 0
        self.tau = 1
        self.lambda_ = 1
        
    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean
    
    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.predicted_mean
    
    def update(self, x):
        self.lambda_ += self.tau
        self.sum_x += x
        self.predicted_mean = self.tau*self.sum_x / self.lambda_
        self.N += 1
        
def plot_(bandits, trial):
    plt.figure()
    x = np.linspace(-3, 6, 200)
    for b in bandits:
        y = norm.pdf(x, b.predicted_mean, np.sqrt(1/b.lambda_))
        plt.plot(x, y, label = f"real mean: {b.true_mean:.4f},num plays: {b.N}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()
    
def experiment():
    bandits = [Bandit(m) for m in BANDIT_MEANS]
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = np.zeros(NUM_TRIALS)
    
    for i in range(NUM_TRIALS):

        # use Thompson-Sampling to select next bandit
        j = np.argmax([b.sample() for b in bandits])
        
        # plot the posteriors
        if i in sample_points:
            plot_(bandits, i)

        # pull the arm for the bandit with largest sample
        x = bandits[j].pull()
        bandits[j].update(x)
        rewards[i] = x
        
    print("total reward earned:", rewards.sum())
    print("overall win rate", rewards.sum()/NUM_TRIALS)
    print("number of times selected each bandit", [b.N for b in bandits])

if __name__ == "__main__":
    experiment()