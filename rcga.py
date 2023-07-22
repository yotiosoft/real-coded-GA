import numpy as np

# dim = 50
DIM = 50
# cell = 300
CELL = 300

x = np.zeros((CELL, DIM), dtype=np.float64)
# x を [0, 1] の間でランダムに初期化
for i in range(CELL):
    for j in range(DIM):
        x[i][j] = np.random.rand()
print(x)    
