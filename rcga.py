import numpy as np
import math

# BLX-α 交叉
def blx_alpha(x1, x2, pc, alpha=0.01):
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

# 評価関数
def rosenbrock(x):
    sum = 0
    for i in range(DIM-1):
        sum += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return sum
    #return sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

# dim = 50
DIM = 50
# cell = 300
CELL = 300
# 交叉率
Pc = 0.7

x = np.zeros((CELL, DIM), dtype=np.float64)
# x を [0, 1] の間でランダムに初期化
for i in range(CELL):
    for j in range(DIM):
        x[i][j] = np.random.rand()

# 遺伝的アルゴリズムの実行
for g in range(1000):
    x_values = np.zeros(CELL, dtype=np.float64)
    x_min = math.inf
    for i in range(CELL):
        # 評価関数の値を計算
        x_values[i] = rosenbrock(x[i])
        if x_min > x_values[i]:
            x_min = x_values[i]
    print("Generation: {0}, Minimum: {1}".format(g, x_min))

    # 親世代をランダムに抽出
    # 抽出数 = n_p
    n_p = 50
    x_parent_index = np.random.randint(0, CELL, n_p)
    x_parent = x[x_parent_index]

    # 交叉
    # 個体数は DIM * 10
    child = np.zeros((DIM * 10, DIM), dtype=np.float64)
    child_values = np.zeros(DIM * 10, dtype=np.float64)
    for i in range(0, DIM * 10, 2):
        # ランダムに2つの個体を選択し、交叉率Pcの確率で交叉を行う
        crossover_x = np.random.randint(0, n_p, 2)
        child[i], child[i+1] = blx_alpha(x_parent[crossover_x[0]], x_parent[crossover_x[1]], Pc)
        child_values[i] = rosenbrock(child[i])
        child_values[i+1] = rosenbrock(child[i+1])

    # エリートを選択
    # エリート数 = n_p
    # まずは child を評価値の小さい順にソート
    child = child[np.argsort(child_values)]
    # 次に 0 ~ np-1 番目の個体をエリートとする
    elite = child[:n_p]

    # x_parent を elite で置き換える
    x[x_parent_index] = elite

# 最終的な個体群の中で最も評価関数の値が小さい個体を選択
