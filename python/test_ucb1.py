from arms.bernoulli import BernoulliArm
from arms.normal import NormalArm
from algorithms.epsilon_greedy.standard import EpsilonGreedy
from algorithms.softmax.standard import Softmax
from algorithms.ucb.ucb1 import UCB1
from algorithms.exp3.exp3 import Exp3
from testing_framework.tests import test_algorithm
from core import ind_max

import random

random.seed(1)
means = [0.1, 0.1, 0.1, 0.1, 0.9]
n_arms = len(means)
random.shuffle(means)
arms = map(lambda mu: BernoulliArm(mu), means)
print("Best arm is " + str(ind_max(means)))

algo = UCB1([], [])
algo.initialize(n_arms)
results = test_algorithm(algo, arms, 5000, 250)

f = open("algorithms/ucb/ucb1_results.tsv", "w")

for i in range(len(results[0])):
    f.write("\t".join([str(results[j][i])
            for j in range(len(results))]) + "\n")

f.close()
