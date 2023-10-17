from algorithms.hedge import Hedge
from arms.bernoulli import BernoulliArm
from testing_framework.tests import test_algorithm
from core import ind_max
import random

random.seed(1)
means = [0.1, 0.1, 0.1, 0.1, 0.9]
n_arms = len(means)
random.shuffle(means)
arms = map(lambda mu: BernoulliArm(mu), means)
print("Best arm is " + str(ind_max(means)))

f = open("algorithms/hedge/hedge_results.tsv", "w")

for eta in [.5, .8, .9, 1, 2]:
    algo = Hedge(eta, [], [])
    algo.initialize(n_arms)
    results = test_algorithm(algo, arms, 5000, 250)
    for i in range(len(results[0])):
        f.write("\t".join([str(results[j][i])
                for j in range(len(results))]) + "\n")

f.close()
