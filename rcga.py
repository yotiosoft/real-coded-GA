import numpy as np
import math

# BLX-α 交叉
def blx_alpha(x1, x2, pc, alpha=0.5):
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

    # 次の世代の個体群を生成する
    # ここではトーナメント選択を採用
    # 次の世代の個体群の個数 = CELL
    # 取り出す数 = 2
    new_x = np.zeros((CELL, DIM), dtype=np.float64)
    for i in range(CELL):
        x_choiced = np.random.choice(CELL, 2, replace=False)
        if x_values[x_choiced[0]] < x_values[x_choiced[1]]:
            x1 = x[x_choiced[0]]
        else:
            x1 = x[x_choiced[1]]
        new_x[i] = x1

    # 交叉
    # ランダムに2つの個体を選択し、交叉率Pcの確率で交叉を行う
    crossover_x = np.random.choice(CELL, 2, replace=False)
    print(new_x[crossover_x[0]], new_x[crossover_x[1]])
    new_x[crossover_x[0]], new_x[crossover_x[1]] = blx_alpha(new_x[crossover_x[0]], new_x[crossover_x[1]], Pc)
    print(new_x[crossover_x[0]], new_x[crossover_x[1]])
    
    x = new_x

# 最終的な個体群の中で最も評価関数の値が小さい個体を選択
