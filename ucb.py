import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
EPS = 0.1 # epsilon
BANDIT_PROBS = [0.2, 0.5, 0.75] # win rates for bandits

class Bandit:
    
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0.
        self.N = 0. # Number of samples
        
    def pull(self):
        return np.random.random() < self.p
    
    def update(self, x):
        self.N += 1
        self.p_estimate = ((self.N - 1)*self.p_estimate + x)/self.N
    
    def ucb(mean, n, nj):
        return mean + np.sqrt(2*np.log(n)/nj)
    
def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBS]
    rewards = np.zeros(NUM_TRIALS)
    total_plays = 0
    
    # initialize the bandit (once)
    for j in range(len(bandits)):
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)

    for i in range(NUM_TRIALS):

        # use epsilon-greedy to select next bandit
        j = np.argmax([ucb(b.p_estimate, total_plays, b.N) for b in bandits])

        # pull the arm for the bandit with largest sample
        x = bandits[j].pull()
        total_plays += 1
        rewards[i] = x
        bandits[j].update(x)
        
        cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

    plt.plot(cumulative_average)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBS))
    plt.xscale("log")
    
    for b in bandits:
        print("mean estimate:", b.p_estimate)

    print("total reward earned", rewards.sum())
    print("overall win rate", rewards.sum()/NUM_TRIALS)
    print("number of times selected each bandit", [b.N for b in bandits])
    
if __name__ == "__main__":
    experiment()