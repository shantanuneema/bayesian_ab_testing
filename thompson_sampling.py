import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

NUM_TRIALS = 2000
EPS = 0.1 # epsilon
BANDIT_PROBS = [0.2, 0.5, 0.75] # win rates for bandits

class Bandit:
    
    def __init__(self, p):
        self.p = p
        self.a = 1.
        self.b = 1.
        self.N = 0. # Number of samples
        
    def pull(self):
        return np.random.random() < self.p
    
    def sample(self):
        return np.random.beta(self.a, self.b)
    
    def update(self, x):
        self.N += 1
        self.a += x
        self.b += 1 - x
        
def plot_(bandits, trial):
    plt.figure()
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label = f"real p: {b.p:.4f}, win rate: {b.a - 1}/{b.N}")
    plt.title(f"Bandit distributions after {trial} trials")

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBS]
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
        rewards[i] = x
        bandits[j].update(x)
        
    print("total reward earned:", rewards.sum())
    print("overall win rate", rewards.sum()/NUM_TRIALS)
    print("number of times selected each bandit", [b.N for b in bandits])

if __name__ == "__main__":
    experiment()