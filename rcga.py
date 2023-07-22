import numpy as np
import math
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures

# BLX-α 交叉
def blx_alpha_one(x1, x2, pc, alpha):
    c1 = np.zeros(DIM, dtype=np.float64)
    c2 = np.zeros(DIM, dtype=np.float64)
    # 交叉率pcの確率で交叉を行う
    crossover_index = np.random.choice([True, False], size=DIM, p=[pc, 1 - pc])
    
    for i in range(DIM):
        if crossover_index[i]:
            # c1, c2 の各次元について、x1, x2 の値の小さい方から
            # (1 + 2 * alpha) 倍した値から alpha 倍した値を引く
            c1[i] = np.random.uniform(
                min(x1[i], x2[i]) - alpha * abs(x1[i] - x2[i]), 
                max(x1[i], x2[i]) + alpha * abs(x1[i] - x2[i])
            )
            c2[i] = np.random.uniform(
                min(x1[i], x2[i]) - alpha * abs(x1[i] - x2[i]), 
                max(x1[i], x2[i]) + alpha * abs(x1[i] - x2[i])
            )
        else:
            c1[i] = x1[i]
            c2[i] = x2[i]

    return c1, c2

def blx_alpha(x_parents, nc, alpha=0.5):
    child = np.zeros((nc, DIM), dtype=np.float64)
    child_values = np.zeros(nc, dtype=np.float64)
    for i in range(0, nc, 2):
        # ランダムに2つの個体を選択し、交叉率Pcの確率で交叉を行う
        crossover_x = np.random.randint(0, n_p, 2)
        child[i], child[i+1] = blx_alpha_one(x_parents[crossover_x[0]], x_parents[crossover_x[1]], Pc, alpha)
        child_values[i] = rosenbrock(child[i])
        child_values[i+1] = rosenbrock(child[i+1])
    return child, child_values

# 単峰性正規分布交叉
def UNDX(p1, p2, p3, alpha=0.5, beta=0.5):
    m = (p1 + p2) / 2
    e = (p2 - p1) / np.abs(p2 - p1)
    
    for i in range(DIM):
        s1 = alpha * abs(p1[i] - p2[i])
        # p1, p2 の直線と p3 の距離
        d2 = np.linalg.norm(np.cross(p3 - p1, p3 - p2)) / np.linalg.norm(p2 - p1)
        print(d2)

# 多親交叉
def REX_thread(x_parents, x_g, parents_n, start, window_size):
    t_children = np.zeros((window_size, DIM), dtype=np.float64)
    t_children_values = np.zeros(window_size, dtype=np.float64)
    
    sigma = np.sqrt(1 / (parents_n))
    for i in range(window_size):
        # 平均0, 分散sigmaの正規分布に従う乱数を生成
        xi = np.random.normal(0, sigma, parents_n)
        #xi = np.random.uniform(-sigma, sigma, parents_n)
            
        # 各親間の距離 * xi
        s = np.zeros(DIM, dtype=np.float64)
        for j in range(parents_n):
            s += xi[j] * (x_parents[j] - x_g)

        # 子個体を生成
        t_children[i] = x_g + s
        t_children_values[i] = rosenbrock(t_children[i])
    return t_children, t_children_values, start, window_size

def REX(x_parents, parents_n, children_n):
    x_children = np.zeros((children_n, DIM), dtype=np.float64)
    x_children_values = np.zeros(children_n, dtype=np.float64)

    # 親の重心を求める
    x_g = np.average(x_parents, axis=0)
    
    window_size = 50
    with ThreadPoolExecutor(max_workers=8) as e:
        # 並列処理
        futures_list = [e.submit(REX_thread, x_parents, x_g, parents_n, start, min(window_size, children_n - start)) for start in range(0, children_n, window_size)]
        for future in futures.as_completed(fs=futures_list):
            t_children, t_children_values, start, window_size = future.result()
            x_children[start:start+window_size] = t_children
            x_children_values[start:start+window_size] = t_children_values
    
    return x_children, x_children_values

# 途中経過を csv に出力
def output_csv(g, x, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow([g+1])
        for i in range(CELL):
            writer.writerow(x[i])

# 途中経過を csv から読み込み
def input_csv(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        x = np.zeros((CELL, DIM), dtype=np.float64)
        g = int(lines[0])-1
        for i in range(CELL):
            strx = lines[i+1].split(",")
            for j in range(DIM):
                x[i][j] = float(strx[j])
        return g, x

# 評価関数
def rosenbrock(x):
    sum = 0
    for i in range(1, DIM):
        sum += 100 * (x[0] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return sum
    #return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

# dim = 50
DIM = 50
# cell = 1000
CELL = 1000
# 交叉率
Pc = 0.7
# 交叉個体数
n_c = 600
# 各ステップにおける親世代の置き換え数
n_p = 100

# x をランダムに初期化
x = np.zeros((CELL, DIM), dtype=np.float64)
for i in range(CELL):
    x[i] = np.random.uniform(-2.048, 2.048, DIM)

# 引数
# 総ステップ数の読み込み
steps = 10000
g = -1
if len(sys.argv) >= 2:
    steps = int(sys.argv[1])
# 途中経過の読み込み
if len(sys.argv) >= 3:
    filename = "rex_{0}.csv".format(sys.argv[2])
    g, x = input_csv(filename)

# 遺伝的アルゴリズムの実行
t_start = time.time()
for g in range(g+1, steps):
    # 親世代をランダムに抽出
    # 抽出数 = n_p
    x_parent_index = np.random.randint(0, CELL, n_p)
    x_parent = x[x_parent_index]

    # 交叉
    # 個体数は n_c
    #child, child_values = blx_alpha(x_parent, n_c)
    child, child_values = REX(x_parent, n_p, n_c)

    # エリートを選択
    # エリート数 = n_p
    # まずは child を評価値の小さい順にソート
    child = child[np.argsort(child_values)]
    # 次に 0 ~ np-1 番目の個体をエリートとする
    elite = child[:n_p]

    # x_parent を elite で置き換える
    x[x_parent_index] = elite

    # 最小となる個体の評価値を出力
    x_values = [rosenbrock(x[i]) for i in range(CELL)]
    x_min = np.min(x_values)
    print("Generation: {0}, Minimum: {1}".format(g, x_min))

    if (g+1) % 100 == 0:
        # print(x[0])
        t_end = time.time()
        print("Time: {1}".format(g, t_end - t_start))
        t_start = time.time()

    # 1000 世代ごとに途中経過を出力
    if (g+1) % 1000 == 0:
        filename = "rex_{0}.csv".format(g+1)
        output_csv(g, x, filename)

# 最終的な個体群の中で最も評価関数の値が小さい個体を選択
