import csv
import os
import matplotlib.pyplot as plt

# Read data from csv file
for n_p in [50, 100, 150, 300, 500]:
    plt.clf()
    for n_c in [100, 300, 600, 900, 1200, 1500, 2000]:
        filename = "results/{0}_{1}_{2}_{3}_{4}_{5}_log.csv".format("REX", "JGG", 1000, 0, n_c, n_p)
        if not os.path.exists(filename):
            continue
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            data = []
            for x in list(reader)[:1000]:
                if x[0] != "time":
                    data.append(float(x[1]))
            plt.plot(data, label="n_p={0}, n_c={1}".format(n_p, n_c))
        plt.legend()
        plt.ylim(0, 5000)
        plt.xlabel("Generation")
        plt.ylabel("Minimum")
    plt.savefig("results/{0}_{1}_{2}_{3}_{4}_graph.png".format("REX", "JGG", 1000, 0, n_p))
    plt.show()
