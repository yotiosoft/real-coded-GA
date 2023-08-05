import numpy as np
from numpy.linalg import norm
import math
import csv
import sys
import time
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures

from enum import Enum

# 交叉モデル、世代交代モデルの列挙型
# パラメータでの指定に使用
# 交叉モデル
class Crossover(Enum):
    BLX_ALPHA = "blx_alpha" # ブレンド交叉
    REX = "rex"             # 多親交叉
# 世代交代モデル
class GenerationGap(Enum):
    MGG = "mgg"             # minimum generation gap
    JGG = "jgg"             # just generation gap

# 実数値 GA クラス
class RealCodedGA:
    def __init__(self, cell, p_c, n_c, n_p, alpha, crossover, generation_gap, max_steps=10000, thold=0.0001):
        # dim = 50
        self.DIM = 50
        # cell
        self.cell = cell
        # 交叉率
        self.p_c = p_c
        # 交叉個体数
        self.n_c = n_c
        # 各ステップにおける親世代の置き換え数
        self.n_p = n_p
        # BLX-α 交叉のα
        self.alpha = alpha
        # 交叉モデル
        self.crossover = crossover
        # 世代交代モデル
        self.generation_gap = generation_gap
        # 途中経過ファイル名
        self.filename_template = "results/{0}_{1}_{2}_{3}_{4}_{5}".format(self.crossover.value, self.generation_gap.value, self.cell, self.p_c, self.n_c, self.n_p)
        # 最大ステップ数
        self.max_steps = max_steps
        # 誤差の閾値
        self.thold = thold

    # BLX-α 交叉
    def blx_alpha_onecycle(self, x1, x2):
        c1 = np.zeros(self.DIM, dtype=np.float64)
        c2 = np.zeros(self.DIM, dtype=np.float64)
        # 交叉率p_cの確率で交叉を行う
        crossover_index = np.random.choice([True, False], size=self.DIM, p=[self.p_c, 1 - self.p_c])
        
        for j in range(self.DIM):
            if crossover_index[j]:
                # c1, c2 の各次元について、x1, x2 の値の小さい方から
                # (1 + 2 * alpha) 倍した値から alpha 倍した値を引く
                c1[j] = np.random.uniform(
                    min(x1[j], x2[j]) - self.alpha * abs(x1[j] - x2[j]), 
                    max(x1[j], x2[j]) + self.alpha * abs(x1[j] - x2[j])
                )
                c2[j] = np.random.uniform(
                    min(x1[j], x2[j]) - self.alpha * abs(x1[j] - x2[j]), 
                    max(x1[j], x2[j]) + self.alpha * abs(x1[j] - x2[j])
                )
            else:
                c1[j] = x1[j]
                c2[j] = x2[j]

        return c1, c2
    def blx_alpha(self, x_parents):
        child = np.zeros((self.n_c, self.DIM), dtype=np.float64)
        crossover_x = np.random.randint(0, self.n_p, 2)
        for i in range(0, self.n_c, 2):
            # ランダムに2つの個体を選択し、交叉率p_cの確率で交叉を行う
            child[i], child[i+1] = self.blx_alpha_onecycle(x_parents[crossover_x[0]], x_parents[crossover_x[1]])
        return child

    # 多親交叉
    def REX(self, x_parents, parents_n, children_n):
        x_children = np.zeros((children_n, self.DIM), dtype=np.float64)

        # 親の重心を求める
        x_g = np.average(x_parents, axis=0)
        
        sigma = np.sqrt(1 / (parents_n))
        for i in range(children_n):
            # 平均0, 分散sigmaの正規分布に従う乱数を生成
            xi = np.random.normal(0, sigma, parents_n)
            #xi = np.random.uniform(-sigma, sigma, parents_n)
            
            # 各親間の距離 * xi
            s = np.zeros(self.DIM, dtype=np.float64)
            for j in range(parents_n):
                s += xi[j] * (x_parents[j] - x_g)

            # 子個体を生成
            x_children[i] = x_g + s
        
        return x_children
        
    # 親世代の個体群からランダムに個体を選択
    def select_parents(self, x_parents, n_p):
        x_parent_index = np.random.randint(0, self.cell, n_p)
        x_parent = x_parents[x_parent_index]
        return x_parent, x_parent_index

    # 子世代からエリートを選択
    def select_elite(self, child, child_values, n_p):
        # まずは child を評価値の小さい順にソート
        child = child[np.argsort(child_values)]
        # 次に 0 ~ np-1 番目の個体をエリートとする
        return child[:n_p]

    # 世代交代
    # MGG (minimum generation gap)
    def MGG(self, x):
        # 親世代からランダムに抽出
        # 抽出数 = n_p
        x_parent, x_parent_index = self.select_parents(x, self.n_p)

        # 交叉
        # 個体数は n_c
        if self.crossover == Crossover.BLX_ALPHA:
            child = self.blx_alpha(x_parent)
        elif self.crossover == Crossover.REX:
            child = self.REX(x_parent, self.n_p, self.n_c)

        # child を x_parent に追加
        parents_and_children = np.concatenate([x_parent, child])

        # child と parents を含めた評価値を計算
        values = [self.rosenbrock(parents_and_children[i]) for i in range(self.n_p + self.n_c)]

        # エリートを選択
        # 親世代をエリートに置き換える
        elite_n = int(self.n_p/2)
        elite = self.select_elite(parents_and_children, values, elite_n)

        # ランク選択
        ranking_n = self.n_p - elite_n
        # 1位20%, 2位15%, 3位10%, 4位5%, 5位以下は全て{50/(n_p+n_c-4)}%
        ranking_p = [0.2, 0.15, 0.1, 0.05] + [0.5/(self.n_p+self.n_c-4)] * (self.n_p+self.n_c-4)
        ranking_index = np.random.choice(self.n_p + self.n_c, ranking_n, p=ranking_p)
        ranking = parents_and_children[ranking_index]

        # 親世代と入れ替え
        x[x_parent_index] = np.concatenate([elite, ranking])

        return x

    # JGG (just generation gap)
    def JGG(self, x):
        # 親世代をランダムに抽出
        # 抽出数 = n_p
        x_parent, x_parent_index = self.select_parents(x, self.n_p)

        # 交叉
        # 個体数は n_c
        if self.crossover == Crossover.BLX_ALPHA:
            child = self.blx_alpha(x_parent)
        elif self.crossover == Crossover.REX:
            child = self.REX(x_parent, self.n_p, self.n_c)

        # 子世代の評価値を計算
        child_values = [self.rosenbrock(child[i]) for i in range(self.n_c)]

        # エリートを選択
        # 親世代をエリートに置き換える
        x[x_parent_index] = self.select_elite(child, child_values, self.n_p)

        return x

    # 途中経過を csv に出力
    def output_csv(self, g, x, filename):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow([g+1])
            for i in range(self.cell):
                writer.writerow(x[i])

    # 途中経過を csv から読み込み
    def input_csv(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()
            x = np.zeros((self.cell, self.DIM), dtype=np.float64)
            g = int(lines[0])-1
            for i in range(self.cell):
                strx = lines[i+1].split(",")
                for j in range(self.DIM):
                    x[i][j] = float(strx[j])
            return g, x
        
    # 最適化結果の出力
    def output_result(self, min_values, result_x, time, filename):
        log_filename = "{0}_log.csv".format(filename)
        with open(log_filename, "w") as f:
            writer = csv.writer(f)
            for i in range(len(min_values)):
                writer.writerow([i+1, min_values[i]])
            writer.writerow(["time", time])

        result_filename = "{0}_result.csv".format(filename)
        with open(result_filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(result_x)

    # 評価関数
    def rosenbrock(self, x):
        sum = 0
        for i in range(1, self.DIM):
            sum += 100 * (x[0] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        return sum

    # 初期化＆実行
    def run(self):
        print("crossover: {0}, generation_gap: {1}, p_c: {2}, alpha: {3}".format(self.crossover.value, self.generation_gap.value, self.p_c, self.alpha))

        # x をランダムに初期化
        x = np.zeros((self.cell, self.DIM), dtype=np.float64)
        for i in range(self.cell):
            x[i] = np.random.uniform(-2.048, 2.048, self.DIM)

        # 遺伝的アルゴリズムの実行
        min_values = []
        result_x = np.full(self.DIM, -1, dtype=np.float64)
        start_time = time.time()
        g = -1
        for g in range(g+1, self.max_steps):
            if self.generation_gap == GenerationGap.MGG:
                x = self.MGG(x)
            elif self.generation_gap == GenerationGap.JGG:
                x = self.JGG(x)

            # 最小となる個体の評価値を計算
            x_values = [self.rosenbrock(x[i]) for i in range(self.cell)]
            x_min = np.min(x_values)
            result_x = x[np.argmin(x_values)]
            min_values.append(x_min)
            print("n_p: {0}, n_c: {1}, Generation: {2}, Minimum: {3}".format(self.n_p, self.n_c, g, x_min))

            # もし誤差が閾値以下になったら終了
            if x_min < self.thold:
                break
        finish_time = time.time()
        
        # 結果を出力
        self.output_result(min_values, result_x, finish_time - start_time, self.filename_template)

        # 評価値の推移をグラフに出力
        plt.clf()
        plt.plot(min_values)
        plt.xlabel("Generation")
        plt.ylabel("Minimum")
        plt.savefig("{0}_min.png".format(self.filename_template))
        #plt.show()

if __name__ == "__main__":
    # 実験1
    # 親個体数と子個体数固定、交叉手法・生存選択・交叉率・αを変えて実験
    # 交叉手法 : BLX-α, REX
    # 生存選択 : MGG, JGG
    # 交叉率 : 0.5, 0.7, 0.9
    # α : 0.25, 0.5, 0.75, 1.0
    # 親個体数 : 50
    # 子個体数 : 300
    #rcgas = []
    for crossover in [Crossover.BLX_ALPHA, Crossover.REX]:
        for generation_gap in [GenerationGap.MGG, GenerationGap.JGG]:
            if crossover == Crossover.BLX_ALPHA:
                for p_c in [0.5, 0.7, 0.9]:
                    # ファイルが既にないか確認
                    filename = "results/{0}_{1}_{2}_{3}_{4}_{5}".format(crossover.value, generation_gap.value, 1000, p_c, 600, 100)
                    if not os.path.exists(filename + "_log.csv"):
                        rcga = RealCodedGA(1000, p_c, 600, 100, 0.5, crossover, generation_gap)
                        rcga.run()
                        #rcgas.append(rcga)
            elif crossover == Crossover.REX:
                # ファイルが既にないか確認
                filename = "results/{0}_{1}_{2}_{3}_{4}_{5}".format(crossover.value, generation_gap.value, 1000, 0, 600, 100)
                if not os.path.exists(filename + "_log.csv"):
                    rcga = RealCodedGA(1000, 0, 600, 100, 0, crossover, generation_gap)
                    rcga.run()
                    #rcgas.append(rcga)

    
    # 実験2
    # 親個体数と子個体数を変化させて実験
    # 交叉手法：REX
    # 生存選択：JGG
    for n_p in [50, 100, 150, 300, 500]:
        for n_c in [100, 300, 600, 900, 1200, 1500, 2000]:
            if n_p > n_c:
                continue
            filename = "results/{0}_{1}_{2}_{3}_{4}_{5}".format(Crossover.REX.value, GenerationGap.JGG.value, 1000, 0, n_c, n_p)
            if not os.path.exists(filename + "_log.csv"):
                rcga = RealCodedGA(1000, 0, n_c, n_p, 0.5, Crossover.REX, GenerationGap.JGG)
                rcga.run()
