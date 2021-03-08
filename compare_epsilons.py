import numpy as np
import matplotlib.pyplot as plt

class BanditArm:
    
    def __init__(self, m):
        self.m = m
        self.m_estimate = 0
        self.N = 0
        
    def pull(self):
        return np.random.randn() + self.m
    
    def update(self, x):
        self.N += 1
        self.m_estimate = (1 - 1/self.N)*self.m_estimate + 1/self.N*x
        
def run_experiment(m1, m2, m3, eps, N):
    
    bandits = [BanditArm(m1), BanditArm(m2), BanditArm(m3)]
    
    means = np.array([m1, m2, m3])
    true_best = np.argmax(means)
    count_suboptimal = 0
    data = np.empty(N)
    
    for i in range(N):
        # epsilon greedy
        p = np.random.random()
        if p < eps:
            j = np.random.choice(len(bandits))
        else:
            j = np.argmax([b.m_estimate for b in bandits])
            
        x = bandits[j].pull()
        bandits[j].update(x)
        
        if j != true_best:
            count_suboptimal += 1

        data[i] = x
    
    cummulative_average = np.cumsum(data) / (np.arange(N) + 1)
    
    plt.plot(cummulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()
    
    for b in bandits:
        print(b.m_estimate)
        
    print("percent suboptimal for epsilon = %s" % eps, count_suboptimal/N)
    
    return cummulative_average

if __name__ == "__main__":
    m1, m2, m3 = 1.5, 2.5, 3.5
    c_0p1 = run_experiment(m1, m2, m3, 0.1, 100000)
    c_0p05 =run_experiment(m1, m2, m3, 0.05, 100000)
    c_0p01 = run_experiment(m1, m2, m3, 0.01, 100000)
    
    plt.plot(c_0p1, label = "eps = 0.1")
    plt.plot(c_0p05, label = "eps = 0.05")
    plt.plot(c_0p01, label = "eps = 0.01")
    plt.legend()
    plt.xscale("log")
    plt.show()