
from algorithm.cod_algo import compute_spectral_norm_XYT_ABT
from dataclasses import dataclass
import numpy as np
from collections import deque

import time
from tqdm import trange
import os
from algorithm.data_loader import load_data

@dataclass
class sampled_row:
    a: np.ndarray
    b: np.ndarray
    t: int
    rho: float

class Sample_sketch:
    def __init__(self, nrows_x, nrows_y, l, window_size):
        self.dx = nrows_x
        self.dy = nrows_y
        self.l = l
        self.window_size = window_size
        self.current_time = 0
        self.candidate = [deque() for _ in range(self.l)]
        self.norm_queue = deque()
        self.norm_A = 0
        self.norm_B = 0
        self.used_sketch_size = 0
        self.update_time = 0
        self.query_time = 0

    def check_expired_blocks(self):
        while(len(self.norm_queue) != 0 and self.norm_queue[0][2] <= self.current_time - self.window_size):
            self.norm_A -= self.norm_queue[0][0]
            self.norm_B -= self.norm_queue[0][1]
            self.norm_queue.popleft()
        
        for i in range(self.l):
            if len(self.candidate[i]) > 0 and self.candidate[i][0].t + self.window_size <= self.current_time:
                self.used_sketch_size -= 1
                self.candidate[i].popleft()

    def priority_score(self, x, y):
        random_num = np.random.uniform(0, 1)
        norm = x * y
        return random_num ** (1 / norm)
    
    def update(self, x, y):
        start_time = time.perf_counter()
        self.current_time += 1
        self.check_expired_blocks()
        
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        self.norm_queue.append((np.square(norm_x), np.square(norm_y), self.current_time))
        
        self.norm_A += np.square(norm_x)
        self.norm_B += np.square(norm_y)
        
        self.used_sketch_size = 0
        for i in range(self.l):
            rho = self.priority_score(norm_x, norm_y)
            while(len(self.candidate[i]) > 0 and self.candidate[i][-1].rho < rho):
                self.candidate[i].pop()
            self.candidate[i].append(sampled_row(x, y, self.current_time, rho))
            self.used_sketch_size += len(self.candidate[i])
        
        end_time = time.perf_counter()
        self.update_time += end_time - start_time

    def query(self):
        start_time = time.perf_counter()
        self.check_expired_blocks()
        resccaler_A = np.sqrt(self.l / self.norm_A)
        resccaler_B = np.sqrt(self.l / self.norm_B)
        A = np.zeros((self.dx, self.l))
        B = np.zeros((self.dy, self.l))
        
        for i in range(self.l):
            A[:, i] = self.candidate[i][0].a
            B[:, i] = self.candidate[i][0].b
            a_norm = np.linalg.norm(A[:, i])
            b_norm = np.linalg.norm(B[:, i])
            A[:, i] = A[:, i] / a_norm / resccaler_A
            B[:, i] = B[:, i] / b_norm / resccaler_B
        
        end_time = time.perf_counter()
        self.query_time += end_time - start_time
        
        return A, B

    def sketch_size(self):
        return self.used_sketch_size

def run_sampling(dataset_name, query_times = 500):
    
    X, Y, nrows_x, nrows_y, tot_cols, window_size, Rx, Ry = load_data(dataset_name)
    
    epsilon = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.05, 0.03]

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
    
    print(epsilon)
    
    R = max(Rx, Ry)
    
    print("Rx: {}, Ry: {}, R: {}".format(Rx, Ry, R))

    with open(
        "result/{}/sample_dx{}_dy{}_cols{}_ws{}.out".format(
            dataset_name, nrows_x, nrows_y, tot_cols, window_size
        ),
        "w",
    ) as output_file:
        output_file.write("Sampling test begins.\n")
        output_file.write(
            "nrows_x: {}, nrows_y: {}, tot_cols: {}, window_size: {}\n".format(
                nrows_x, nrows_y, tot_cols, window_size
            )
        )
        output_file.flush()

        for eps in epsilon:
            L = int(np.ceil(1 / (eps ** 2)))
            Sampling_ske = Sample_sketch(nrows_x, nrows_y, L, window_size)

            all_errors = []
            max_error = 0
            avg_error = 0
            sketch_size = []

            for i in trange(
                tot_cols,
                desc="Sampling Processing with epsilon = {}".format(eps),
                ncols=100,
            ):
                Sampling_ske.update(X[:, i], Y[:, i])
                sketch_size.append(Sampling_ske.sketch_size())
                
                if i > 0 and i % (tot_cols // query_times) == 0:
                    A, B = Sampling_ske.query()
                    
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