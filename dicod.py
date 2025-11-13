from algorithm.cod_block import COD_Block
from algorithm.cod_algo import compute_spectral_norm_XYT_ABT
import numpy as np
import copy
from collections import deque
import time
from tqdm import trange
from algorithm.data_loader import load_data
import os

class DI_cod_sketch:
    def __init__(self, nrows_x, nrows_y, Rx, Ry, L, window_size):
        self.dx = nrows_x
        self.dy = nrows_y
        self.base_len = 0
        self.Rx = Rx
        self.Ry = Ry
        self.levels = L
        self.window_size = window_size
        self.current_time = 0
        self.sizex = 0
        self.sizey = 0
        self.update_time = 0
        self.query_time = 0
        
        self.L_inactive = [deque() for _ in range(self.levels + 1)]
        self.L_buffer = [COD_Block(1, 1, 2, 1, 1)]
        
        self.l_size = [self.levels]
        
        for i in range(1, self.levels + 1):
            self.l_size.append(self.levels * 2 ** i)
            self.L_buffer.append(COD_Block(self.dx, self.dy, self.l_size[i] * 2, self.l_size[i], 1))
        
        self.used_sketch_size = 0

    def check_expired_blocks(self):
        
        for i in range(1, self.levels + 1):
            if len(self.L_inactive[i]) > 0 and self.L_inactive[i][0].start_time + self.window_size <= self.current_time:
                self.used_sketch_size -= self.L_inactive[i][0].used_cols
                self.L_inactive[i].popleft()


    def update(self, x, y):
        start_time = time.perf_counter()
        self.current_time += 1
        self.check_expired_blocks()
        
        for i in range(1, self.levels + 1):
            self.used_sketch_size -= self.L_buffer[i].used_cols
            self.L_buffer[i].update(x, y, 1)
            self.used_sketch_size += self.L_buffer[i].used_cols
            
            self.L_buffer[i].end_time = self.current_time
        
        
        self.sizex += np.sum(np.square(x))
        self.sizey += np.sum(np.square(y))
        
        if(self.sizex >= self.Rx * self.window_size / (2 ** self.levels) or self.sizey >= self.Ry * self.window_size / (2 ** self.levels)):
            self.sizex = 0
            self.sizey = 0
            self.base_len += 1
            v = self.base_len
            v = (v & -v).bit_length()
            v = min(v, self.levels + 1) 
            
            for i in range(1, v):
                self.L_inactive[i].append(self.L_buffer[i])
                self.L_buffer[i] = COD_Block(self.dx, self.dy, self.l_size[i] * 2, self.l_size[i], self.current_time + 1)
        
        end_time = time.perf_counter()
        self.update_time += end_time - start_time

    def query(self):
        start_time = time.perf_counter()
        a = self.current_time
        b = self.current_time - self.window_size + 1
        
        A = None
        B = None
        for i in range(self.levels, 0, -1):
            if(len(self.L_inactive[i]) != 0):
                lb = self.L_inactive[i][0]
                if(lb.start_time >= self.current_time - self.window_size + 1):
                    if(lb.end_time <= a):
                        if(A is None):
                            A = copy.copy(self.L_inactive[i][0].X)
                            B = copy.copy(self.L_inactive[i][0].Y)
                        else:
                            A = np.concatenate((A, self.L_inactive[i][0].X), axis = 1)
                            B = np.concatenate((B, self.L_inactive[i][0].Y), axis = 1)
                            
                        a = self.L_inactive[i][0].start_time

                    if(len(self.L_inactive[i]) == 1):
                        b = self.L_inactive[i][0].end_time 
                    else:
                        if(self.L_inactive[i][-1].start_time >= b):
                            if(A is None):
                                A = copy.copy(self.L_inactive[i][-1].X)
                                B = copy.copy(self.L_inactive[i][-1].Y)
                            else:
                                A = np.concatenate((A, self.L_inactive[i][-1].X), axis = 1)
                                B = np.concatenate((B, self.L_inactive[i][-1].Y), axis = 1)
                                
                            b = self.L_inactive[i][-1].end_time
                lb = None
              
        end_time = time.perf_counter()
        time_count = end_time - start_time
        self.query_time += time_count
          
        if A is None:
            return np.zeros((self.dx, 1)), np.zeros((self.dy, 1))
        
        return A, B

    def sketch_size(self):
        return self.used_sketch_size

def run_dicod(dataset_name, query_times = 500):

    X, Y, nrows_x, nrows_y, tot_cols, window_size, Rx, Ry = load_data(dataset_name)
    
    print(
        "{} - nrows_x: {}, nrows_y: {}, tot_cols: {}, window_size: {}\n".format(
            dataset_name, nrows_x, nrows_y, tot_cols, window_size
        )
    )

    R = max(Rx, Ry)
    print("Rx: {}, Ry: {}, R: {}".format(Rx, Ry, R))    
    
    try:
        os.mkdir("result/{}".format(dataset_name))
    except FileExistsError:
        pass
    
    possible_L = np.array([x for x in range(1, int(np.ceil(np.log2(R))) + 7)])
    print(possible_L)

    with open(
        "result/{}/dicod_dx{}_dy{}_cols{}_ws{}_.out".format(
            dataset_name, nrows_x, nrows_y, tot_cols, window_size
        ),
        "w",
    ) as output_file:
        output_file.write("DI_COD test begins.\n")
        output_file.write(
            "nrows_x: {}, nrows_y: {}, tot_cols: {}, window_size: {}\n".format(
                nrows_x, nrows_y, tot_cols, window_size
            )
        )
        output_file.flush()

        for L in possible_L:
            eps = R / (2 ** L)
            DI_COD = DI_cod_sketch(nrows_x, nrows_y, Rx, Ry, L, window_size)
            
            all_errors = []
            max_error = 0
            avg_error = 0
            sketch_size = []

            for i in trange(
                tot_cols,
                desc="DI-COD Processing with epsilon = {}".format(eps),
                ncols=100,
            ):
                DI_COD.update(X[:, i], Y[:, i])
                sketch_size.append(DI_COD.sketch_size())
                if i > 0 and i % (tot_cols // query_times) == 0:
                    A, B = DI_COD.query()
                    
                    if(A is None):
                        continue
                        
                    _X = X[:, max(i - window_size + 1, 0) : i + 1]
                    _Y = Y[:, max(i - window_size + 1, 0) : i + 1]
                    error = compute_spectral_norm_XYT_ABT(_X, _Y, A, B)

                    max_error = max(max_error, error)
                    all_errors.append(error)
                    avg_error = np.mean(all_errors)

            sketch_size = np.array(sketch_size)
            output_file.write(
                "eps: {}, Max error: {}, Avg error: {}, Sketch Size: {}\n".format(
                    eps, max_error, avg_error, np.max(sketch_size))
            )
            output_file.flush()


if __name__ == "__main__":
    pass