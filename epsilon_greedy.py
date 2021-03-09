# Epsilon-greedy starter code

import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
EPS = 0.1 # epsilon
BANDIT_PROBS = [0.2, 0.5, 0.75] # win rates for bandits

class Bandit:
    
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0 # Number of samples
        
    def pull(self):
        return np.random.random() < self.p
    
    def update(self, x):
        self.N += 1
        self.p_estimate = ((self.N - 1)*self.p_estimate + x)/self.N
        
    def experiment():
        bandits = [Bandit(p) for p in BANDIT_PROBS]
        rewards = np.zeros(NUM_TRIALS)
        num_times_explored = 0
        num_times_exploited = 0
        num_optimal = 0
        
        optimal_j = np.argmax([b.p for b in bandits])
        print("optimal_j:", optimal_j)
        
        for i in range(NUM_TRIALS):
            
            # use epsilon-greedy to select next bandit
            if np.random.random() < EPS:
                num_times_explored += 1
                j = bandits.index(np.random.choice(bandits))
            else:
                num_times_exploited += 1
                j = np.argmax([b.p for b in bandits])
                
            if j == optimal_j:
                num_optimal += 1
                
            # pull the arm for the bandit with largest sample
            x = Bandit(bandits[j].p).pull()
            
            # update the rewards log
            rewards[i] = x
            
            # update the distribution for the bandit whose arm we just pulled
            Bandit(bandits[j]).update(x)

        # for b in bandits:
        #     print("mean estimate:", Bandit(b).p_estimate)

        print("total reward earned", rewards.sum())
        print("overall win rate", rewards.sum()/NUM_TRIALS)
        print("num_times_explored", num_times_explored)
        print("num_times_exploited", num_times_exploited)
        print("num times selected optimal bandit", num_optimal)

        cummulative_rewards = np.cumsum(rewards)
        win_rates = cummulative_rewards/(np.arange(NUM_TRIALS) + 1)
        plt.figure(figsize = (10,8))
        plt.plot(win_rates)
        plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBS))
        plt.ylim(0,1)
        plt.show()
    
if __name__ == "__main__":
    Bandit.experiment()