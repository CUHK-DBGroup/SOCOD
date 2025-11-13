from algorithm.cod_block import COD_Block
from algorithm.cod_algo import compute_spectral_norm_XYT_ABT
import numpy as np
import copy
from collections import deque
import time
from tqdm import trange
from algorithm.data_loader import load_data
import os

class EH_cod_sketch:
    def __init__(self, nrows_x, nrows_y, ncols, b, window_size):
        self.dx = nrows_x
        self.dy = nrows_y
        self.size = ncols
        self.limit_each_block = b
        self.window_size = window_size
        self.current_time = 0
        self.B0 = COD_Block(self.dx, self.dy, self.size * 2, self.size, 1)
        self.levels = 0
        self.L = [deque()]
        self.used_sketch_size = 0
        self.update_time = 0
        self.query_time = 0

    def check_expired_blocks(self):
        if self.levels == 0:
            return

        if self.L[self.levels][0].end_time + self.window_size <= self.current_time:
            self.used_sketch_size -= self.L[self.levels][0].used_cols
            self.L[self.levels].popleft()
            if len(self.L[self.levels]) == 0:
                self.levels -= 1
                self.L.pop()

    def update(self, x, y):
        start_time = time.perf_counter()
        self.current_time += 1
        self.used_sketch_size += 1
        self.check_expired_blocks()
        self.B0.update(x, y, False)
        self.B0.end_time = self.current_time
        if self.B0.size_norm >= self.size or self.B0.used_cols == self.B0.shrink_cols:
            if self.levels == 0:
                self.levels += 1
                self.L.append(deque())
            self.L[1].append(self.B0)
            self.B0 = COD_Block(
                self.dx, self.dy, self.size * 2, self.size, self.current_time + 1
            )

        size_limit_each_level = self.size
        for i in range(1, self.levels + 1):
            size_limit_each_level = size_limit_each_level * 2
            if len(self.L[i]) >= self.limit_each_block + 1:
                if i == self.levels:
                    self.levels += 1
                    self.L.append(deque())

                K1 = self.L[i].popleft()
                if K1.size_norm >= size_limit_each_level:
                    self.L[i + 1].append(K1)
                    continue
                K2 = self.L[i].popleft()
                self.used_sketch_size -= K1.used_cols
                self.used_sketch_size -= K2.used_cols

                K1.merge(K2)
                self.L[i + 1].append(K1)
                self.used_sketch_size += K1.used_cols

                K2 = None
        end_time = time.perf_counter()
        self.update_time += end_time - start_time

    def query(self):
        start_time = time.perf_counter()
        Result = None
        for i in range(1, self.levels + 1):
            for j in range(len(self.L[i])):
                if Result is None:
                    Result = copy.deepcopy(self.L[i][j])
                else:
                    Result.merge(self.L[i][j])
        
        end_time = time.perf_counter()
        self.query_time += end_time - start_time
        
        if Result is None:
            return np.zeros((self.dx, 1)), np.zeros((self.dy, 1))
        
        return Result.X[:, : Result.used_cols], Result.Y[:, : Result.used_cols]

    def sketch_size(self):
        tot_size = self.B0.sketch_size()
        for i in range(1, self.levels + 1):
            for j in range(len(self.L[i])):
                tot_size += self.L[i][j].sketch_size()
        return tot_size

def run_ehcod(dataset_name, query_times = 500):
    
    X, Y, nrows_x, nrows_y, tot_cols, window_size, Rx, Ry = load_data(dataset_name)

    epsilon = [4, 2.5, 2, 1.8, 1.5, 1.2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    if (X is None) or (Y is None):
        print("Dataset {} not found.".format(dataset_name))
        return
    
    print(
        "{} - nrows_x: {}, nrows_y: {}, tot_cols: {}, window_size: {}\n".format(
            dataset_name, nrows_x, nrows_y, tot_cols, window_size
        )
    )
    try:
        os.mkdir("result/{}".format(dataset_name))
    except FileExistsError:
        pass
    
    R = max(Rx, Ry)
    
    print("Rx: {}, Ry: {}, R: {}".format(Rx, Ry, R))
    
    print(epsilon)
    
    with open(
        "result/{}/ehcod_dx{}_dy{}_cols{}_ws{}.out".format(
            dataset_name, nrows_x, nrows_y, tot_cols, window_size
        ),
        "w",
    ) as output_file:
        output_file.write("EH_COD test begins.\n")
        output_file.write(
            "nrows_x: {}, nrows_y: {}, tot_cols: {}, window_size: {}\n".format(
                nrows_x, nrows_y, tot_cols, window_size
            )
        )
        output_file.flush()

        for eps in epsilon:
            l = int(np.ceil(8 / eps))
            EH_COD = EH_cod_sketch(nrows_x, nrows_y, l, l, window_size)

            all_errors = []
            max_error = 0
            avg_error = 0
            sketch_size = []

            for i in trange(
                tot_cols,
                desc="EH-COD Processing with epsilon = {}".format(eps),
                ncols=100,
            ):
                EH_COD.update(X[:, i], Y[:, i])
                sketch_size.append(EH_COD.sketch_size())
                    
                if i > 0 and i % (tot_cols // query_times) == 0:
                    A, B = EH_COD.query()
                    _X = X[:, max(i - window_size + 1, 0) : i + 1]
                    _Y = Y[:, max(i - window_size + 1, 0) : i + 1]
                    error = compute_spectral_norm_XYT_ABT(_X, _Y, A, B)
                    
                    max_error = max(max_error, error)
                    all_errors.append(error)
                    avg_error = np.mean(all_errors)
            
            sketch_size = np.array(sketch_size)
            output_file.write(
                "eps: {}, Max error: {}, Avg error: {}, Sketch Size: {}\n".format(
                    eps, max_error, avg_error, np.max(sketch_size)
                )
            )
            output_file.flush()
            

if __name__ == "__main__":
    pass