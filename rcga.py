import numpy as np
from numpy.linalg import norm
import math
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
import matplotlib.pyplot as plt
from enum import Enum

# 交叉モデル
class Crossover(Enum):
    BLX_ALPHA = "blx_alpha" # ブレンド交叉
    REX = "rex"             # 多親交叉

# 世代交代モデル
class GenerationGap(Enum):
    MGG = "mgg"             # minimum generation gap
    JGG = "jgg"             # just generation gap

# dim = 50
DIM = 50
# cell = 1000
CELL = 1000
# 交叉率
Pc = 0.7
# 交叉個体数
n_c = 300
# 各ステップにおける親世代の置き換え数
n_p = 50
# 交叉モデル
crossover = Crossover.BLX_ALPHA
# 世代交代モデル
generation_gap = GenerationGap.JGG
# 途中経過ファイル名
filename_template = "results/{0}_{1}".format(crossover.value, generation_gap.value)

# BLX-α 交叉
def blx_alpha_onecycle(x1, x2, pc, alpha):
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
    crossover_x = np.random.randint(0, n_p, 2)
    for i in range(0, nc, 2):
        # ランダムに2つの個体を選択し、交叉率Pcの確率で交叉を行う
        child[i], child[i+1] = blx_alpha_onecycle(x_parents[crossover_x[0]], x_parents[crossover_x[1]], Pc, alpha)
        child_values[i] = rosenbrock(child[i])
        child_values[i+1] = rosenbrock(child[i+1])
    return child, child_values

# 単峰性正規分布交叉
def UNDX_distance(p1, p2, p3):
    ap = p3 - p1
    ab = p2 - p1
    ba = p1 - p2
    bp = p3 - p2
    if np.dot(ap, ab) < 0:
        distance = norm(ap)
        neighbor_point = p1
    elif np.dot(bp, ba) < 0:
        distance = norm(p3 - p2)
        neighbor_point = p2
    else:
        ai_norm = np.dot(ap, ab)/norm(ab)
        neighbor_point = p1 + (ab)/norm(ab)*ai_norm
        distance = norm(p3 - neighbor_point)
    return distance

def UNDX_onecycle(p1, p2, p3, alpha, beta):
    c1 = np.zeros(DIM, dtype=np.float64)
    c2 = np.zeros(DIM, dtype=np.float64)

    m = (p1 + p2) / 2

    e = np.zeros((DIM, DIM), dtype=np.float64)
    e[0] = (p2 - p1) / np.abs(p2 - p1)
    # e0に垂直かつ線形独立な単位ベクトルを生成
    for i in range(1, DIM):
        e[i] = np.random.normal(0, 1, DIM)
        e[i] -= np.dot(e[i], e[0]) * e[0]
        e[i] /= norm(e[i])
    z = np.zeros(DIM, dtype=np.float64)

    s1 = alpha * norm(p1 - p2)
    z[0] = np.random.normal(0, s1)
    # p1, p2 の直線と p3 の距離
    d2 = UNDX_distance(p1, p2, p3)
    s2 = (beta * d2) / (np.sqrt(DIM))
    for i in range(1, DIM):
        z[i] = np.random.normal(0, s2)
    
    zesum = z[0] * e[0]
    for k in range(1, DIM):
        zesum += z[k] * e[k]

    c1 = m + zesum
    c2 = m - zesum

    return c1, c2

def UNDX(x_parents, nc, alpha=0.5, beta=0.35):
    child = np.zeros((nc, DIM), dtype=np.float64)
    child_values = np.zeros(nc, dtype=np.float64)
    for i in range(0, nc, 2):
        # ランダムに3つの個体を選択し、交叉率Pcの確率で交叉を行う
        crossover_x = np.random.randint(0, n_p, 3)
        child[i], child[i+1] = UNDX_onecycle(x_parents[crossover_x[0]], x_parents[crossover_x[1]], x_parents[crossover_x[2]], alpha, beta)
        child_values[i] = rosenbrock(child[i])
        child_values[i+1] = rosenbrock(child[i+1])
    return child, child_values

# 多親交叉
def REX(x_parents, parents_n, children_n):
    x_children = np.zeros((children_n, DIM), dtype=np.float64)
    x_children_values = np.zeros(children_n, dtype=np.float64)

    # 親の重心を求める
    x_g = np.average(x_parents, axis=0)
    
    sigma = np.sqrt(1 / (parents_n))
    for i in range(children_n):
        # 平均0, 分散sigmaの正規分布に従う乱数を生成
        xi = np.random.normal(0, sigma, parents_n)
        #xi = np.random.uniform(-sigma, sigma, parents_n)
        
        # 各親間の距離 * xi
        s = np.zeros(DIM, dtype=np.float64)
        for j in range(parents_n):
            s += xi[j] * (x_parents[j] - x_g)

        # 子個体を生成
        x_children[i] = x_g + s
        x_children_values[i] = rosenbrock(x_children[i])
    
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
    
# 親世代の個体群からランダムに個体を選択
def select_parents(x_parents, n_p):
    x_parent_index = np.random.randint(0, CELL, n_p)
    x_parent = x_parents[x_parent_index]
    return x_parent, x_parent_index

# 子世代からエリートを選択
def select_elite(child, child_values, n_p):
    # まずは child を評価値の小さい順にソート
    child = child[np.argsort(child_values)]
    # 次に 0 ~ np-1 番目の個体をエリートとする
    return child[:n_p]

# 世代交代
# MGG (minimum generation gap)
def MGG(x, n_c):
    n_p = 2
    # 親世代からランダムに抽出
    # 抽出数 = n_p
    x_parent, x_parent_index = select_parents(x, n_p)

    # 交叉
    # 個体数は n_c
    if crossover == Crossover.BLX_ALPHA:
        child, child_values = blx_alpha(x_parent, n_c)
    elif crossover == Crossover.REX:
        child, child_values = REX(x_parent, n_p, n_c)

    # child を x_parent に追加
    parents_and_children = np.concatenate([x_parent, child])

    # エリートを選択
    elite = select_elite(parents_and_children, child_values, n_p)

# JGG (just generation gap)
def JGG(x, n_p, n_c):
    # 親世代をランダムに抽出
    # 抽出数 = n_p
    x_parent, x_parent_index = select_parents(x, n_p)

    # 交叉
    # 個体数は n_c
    if crossover == Crossover.BLX_ALPHA:
        child, child_values = blx_alpha(x_parent, n_c)
    elif crossover == Crossover.REX:
        child, child_values = REX(x_parent, n_p, n_c)

    # エリートを選択
    # 親世代をエリートに置き換える
    x[x_parent_index] = select_elite(child, child_values, n_p)

    return x

# 評価関数
def rosenbrock(x):
    sum = 0
    for i in range(1, DIM):
        sum += 100 * (x[0] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return sum
    #return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

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
    filename = "{0}_{1}.csv".format(filename_template, sys.argv[2])
    g, x = input_csv(filename)

# 遺伝的アルゴリズムの実行
t_start = time.time()
min_values = np.zeros(steps, dtype=np.float64)
for g in range(g+1, steps):
    x = JGG(x, n_p, n_c)

    # 最小となる個体の評価値を出力
    x_values = [rosenbrock(x[i]) for i in range(CELL)]
    x_min = np.min(x_values)
    min_values[g] = x_min
    print("Generation: {0}, Minimum: {1}".format(g, x_min))

    # 100 世代ごとに経過時間を出力
    if (g+1) % 100 == 0:
        t_end = time.time()
        print("Time: {1}".format(g, t_end - t_start))
        t_start = time.time()

    # 1000 世代ごとに途中経過を出力
    if (g+1) % 1000 == 0:
        filename = "{0}_{1}.csv".format(filename_template, g+1)
        output_csv(g, x, filename)

# 評価値の推移をグラフに出力
plt.plot(min_values)
plt.xlabel("Generation")
plt.ylabel("Minimum")
plt.show()
