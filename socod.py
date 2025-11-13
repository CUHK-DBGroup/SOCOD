import numpy as np
from collections import deque
from algorithm.cod_algo import compute_spectral_norm_XYT_ABT

import time
from tqdm import trange
from algorithm.data_loader import load_data
import os

class SO_COD_Sketch_Fast:

    def __init__(
        self,
        nrows_x,
        nrows_y,
        ncols,
        threshold=-1,
        window_size=1,
        shrink_cols=0,
        limit_snapshot_num=-1,
    ):
        self.dx = nrows_x
        self.dy = nrows_y
        self.size = ncols
        self.used_cols = 0
        self.X = np.zeros((self.dx, self.size), order="F")
        self.Y = np.zeros((self.dy, self.size), order="F")
        self.Qx = np.zeros((self.dx, self.size), order="F")
        self.Qy = np.zeros((self.dy, self.size), order="F")
        self.Rx = np.zeros((self.size, self.size), order="F")
        self.Ry = np.zeros((self.size, self.size), order="F")
        self.snapshot_num = 0
        self.timestamp = deque()
        self.Sx = deque()
        self.Sy = deque()
        self.window_size = window_size
        self.current_time = 0

        self.threshold = threshold
        self.limit_snapshot_num = limit_snapshot_num
        self.latest_expire_time = - self.window_size - 10

        self.shrink_cols = shrink_cols

        if shrink_cols == 0:
            self.shrink_cols = np.min(ncols // 2, ncols - 1)

    def shrink(self):
        self.Qx, self.Rx = np.linalg.qr(self.X)
        self.Qy, self.Ry = np.linalg.qr(self.Y)
        self.K = np.dot(self.Rx, self.Ry.T)
        U, S, V = np.linalg.svd(self.K)
        S = np.maximum(S - S[self.shrink_cols], 0)
        S = np.sqrt(S)

        self.used_cols = self.shrink_cols
        self.X = np.dot(self.Qx, np.dot(U, np.diag(S)))
        self.Y = np.dot(self.Qy, np.dot(V.T, np.diag(S)))
        
        (
            self.Qx[:, : self.shrink_cols],
            self.Rx[: self.shrink_cols, : self.shrink_cols],
        ) = np.linalg.qr(self.X[:, : self.shrink_cols])
        (
            self.Qy[:, : self.shrink_cols],
            self.Ry[: self.shrink_cols, : self.shrink_cols],
        ) = np.linalg.qr(self.Y[:, : self.shrink_cols])

        self.Qx[:, self.shrink_cols :] = 0
        self.Qy[:, self.shrink_cols :] = 0
        self.Rx[:, self.shrink_cols :] = 0
        self.Ry[:, self.shrink_cols :] = 0
        
        
    def compress_snapshot(self):
        if self.threshold < 0:
            return

        self.K = np.dot(self.Rx, self.Ry.T)
        U, S, V = np.linalg.svd(self.K)
        V = V.T


        for i in range(self.size):
            if S[i] > self.threshold:
                u = U[:, i].reshape((-1, 1), order="F")
                v = V[:, i].reshape((-1, 1), order="F")

                Qu = self.Qx @ u
                uTRx = u.T @ self.Rx
                QuuTRx = Qu @ uTRx
                self.X = self.X - QuuTRx

                Qv = self.Qy @ v
                vTRy = v.T @ self.Ry
                QvvTRy = Qv @ vTRy
                self.Y = self.Y - QvvTRy

                uuTRx = u @ uTRx
                self.Rx = self.Rx - uuTRx

                vvTRy = v @ vTRy
                self.Ry = self.Ry - vvTRy

                u = Qu * np.sqrt(S[i])
                v = Qv * np.sqrt(S[i])

                self.Sx.append(u)
                self.Sy.append(v)
                self.timestamp.append(self.current_time)
                self.snapshot_num += 1

            else:
                break
        
    def check_expire_snapshot(self):
        while (
            self.snapshot_num > 0
            and self.timestamp[0] + self.window_size <= self.current_time
        ):
            self.latest_expire_time = self.timestamp[0]
            self.timestamp.popleft()
            self.Sx.popleft()
            self.Sy.popleft()
            self.snapshot_num -= 1

    def check_snapshots_limit(self):
        if self.limit_snapshot_num > 0:
            while self.snapshot_num > self.limit_snapshot_num:
                self.latest_expire_time = self.timestamp[0]
                self.timestamp.popleft()
                self.Sx.popleft()
                self.Sy.popleft()
                self.snapshot_num -= 1

    def update(self, x, y):
        self.current_time += 1

        self.check_expire_snapshot()

        self.X[:, self.used_cols] = x
        self.Y[:, self.used_cols] = y

        x_temp = np.copy(x)

        for i in range(self.used_cols):
            cor = np.dot(self.Qx[:, i], x_temp)
            x_temp = x_temp - cor * self.Qx[:, i]
            self.Rx[i, self.used_cols] = cor

        cor = np.linalg.norm(x_temp)
        self.Rx[self.used_cols, self.used_cols] = cor

        if cor > 1e-10:
            self.Qx[:, self.used_cols] = x_temp / cor

        y_temp = np.copy(y)

        for i in range(self.used_cols):
            cor = np.dot(self.Qy[:, i], y_temp)
            y_temp = y_temp - cor * self.Qy[:, i]
            self.Ry[i, self.used_cols] = cor

        cor = np.linalg.norm(y_temp)
        self.Ry[self.used_cols, self.used_cols] = cor

        if cor > 1e-10:
            self.Qy[:, self.used_cols] = y_temp / cor

        self.used_cols += 1
        
        if self.used_cols == self.size:
            self.shrink()

        self.compress_snapshot()
        self.check_snapshots_limit()

    def direct_update(self, x, y):
        self.current_time += 1
        self.Sx.append(x)
        self.Sy.append(y)
        self.timestamp.append(self.current_time)
        self.snapshot_num += 1
        self.check_snapshots_limit()

    def retrieve(self):
        return self.X, self.Y

    def retrieve_with_snapshot(self):
        X_ = np.zeros((self.dx, self.used_cols + self.snapshot_num))
        Y_ = np.zeros((self.dy, self.used_cols + self.snapshot_num))

        X_[:, : self.used_cols] = self.X[:, : self.used_cols]
        for i in range(self.snapshot_num):
            X_[:, self.used_cols + i] = self.Sx[i].reshape(-1)

        Y_[:, : self.used_cols] = self.Y[:, : self.used_cols]
        for i in range(self.snapshot_num):
            Y_[:, self.used_cols + i] = self.Sy[i].reshape(-1)
        
        return X_, Y_

    def sketch_size(self):
        return self.used_cols + self.snapshot_num

    def reset(self):
        self.__init__(
            self.dx,
            self.dy,
            self.size,
            self.threshold,
            self.window_size,
            self.shrink_cols,
        )


class SO_COD_Sketch_Layer:
    def __init__(
        self,
        nrows_x,
        nrows_y,
        ncols,
        threshold=-1.0,
        window_size=1,
        shrink_cols=0,
        limit_snapshot_num=-1,
    ):
        self.dx = nrows_x
        self.dy = nrows_y
        self.size = ncols
        self.window_size = window_size
        self.current_time = 0

        self.threshold = threshold
        self.limit_snapshot_num = limit_snapshot_num
        self.latest_expire_time = - self.window_size - 10

        self.shrink_cols = shrink_cols
        self.update_time = 0
        self.query_time = 0
        
        if shrink_cols == 0:
            self.shrink_cols = np.min(ncols // 2, ncols - 1)

        self.main_sketch = SO_COD_Sketch_Fast(
            nrows_x,
            nrows_y,
            ncols,
            threshold,
            window_size,
            shrink_cols,
            limit_snapshot_num,
        )
        self.auxiliary_sketch = SO_COD_Sketch_Fast(
            nrows_x,
            nrows_y,
            ncols,
            threshold,
            window_size,
            shrink_cols,
            limit_snapshot_num,
        )

    def restart_check(self):
        if self.current_time % self.window_size == 1:
            self.main_sketch = self.auxiliary_sketch
            self.auxiliary_sketch = SO_COD_Sketch_Fast(
                self.dx,
                self.dy,
                self.size,
                self.threshold,
                self.window_size,
                self.shrink_cols,
                self.limit_snapshot_num,
            )
            self.auxiliary_sketch.current_time = self.current_time - 1

    def update(self, x, y):
        start_time = time.perf_counter()
        self.current_time += 1
        self.restart_check()

        norm_x = np.sqrt(np.sum(np.square(x)))
        norm_y = np.sqrt(np.sum(np.square(y)))
        
        if norm_x * norm_y >= self.threshold:
            self.main_sketch.direct_update(x, y)
            self.auxiliary_sketch.direct_update(x, y)
        else:
            self.main_sketch.update(x, y)
            self.auxiliary_sketch.update(x, y)
        
        end_time = time.perf_counter()
        self.update_time += end_time - start_time
        
    def direct_update(self, x, y):
        self.current_time += 1
        self.restart_check()

        self.main_sketch.direct_update(x, y)
        self.auxiliary_sketch.direct_update(x, y)

    def retrieve(self):
        return self.main_sketch.retrieve()

    def retrieve_with_snapshot(self):
        return self.main_sketch.retrieve_with_snapshot()

    def sketch_size(self):
        return self.main_sketch.sketch_size() + self.auxiliary_sketch.sketch_size()

    def check_exceed_limit(self):
        if self.limit_snapshot_num > 0:
            if (
                self.main_sketch.latest_expire_time
                > self.current_time - self.window_size
            ):
                return False
        return True

    def query(self):
        start_time = time.perf_counter()
        A, B = self.main_sketch.retrieve_with_snapshot()
        end_time = time.perf_counter()
        self.query_time += end_time - start_time
        return A, B


class SO_COD_multilayers:
    def __init__(
        self,
        nrows_x,
        nrows_y,
        ncols,
        threshold=-1,
        window_size=1,
        shrink_cols=0,
        limit_snapshot_num=-1,
        num_layers=1,
    ):
        self.layers = []
        self.threshold = []
        self.num_layers = num_layers
        for i in range(num_layers + 1):
            self.threshold.append(threshold * (2 ** i))

        for i in range(self.num_layers + 1):
            self.layers.append(
                SO_COD_Sketch_Layer(
                    nrows_x,
                    nrows_y,
                    ncols,
                    self.threshold[i],
                    window_size,
                    shrink_cols,
                    limit_snapshot_num,
                )
            )
        
        self.update_time = 0
        self.query_time = 0

    def update(self, x, y):
        start_time = time.perf_counter()
        for i in range(self.num_layers + 1):
            self.layers[i].update(x, y)
        end_time = time.perf_counter()
        self.update_time += end_time - start_time

    def query(self):
        A, B = None
        start_time = time.perf_counter()
        for i in range(self.num_layers + 1):
            if self.layers[i].check_exceed_limit():
                A, B = self.layers[i].query()
                break
        
        end_time = time.perf_counter()
        self.query_time += end_time - start_time
        return A, B

    def sketch_size(self):
        size = 0
        for i in range(self.num_layers + 1):
            size += self.layers[i].sketch_size()
            
        return size

def run_socod(dataset_name, query_times = 500):
    
    X, Y, nrows_x, nrows_y, tot_cols, window_size, Rx, Ry = load_data(dataset_name)

    epsilon = [1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]
    
    if (X is None) or (Y is None):
        print("Dataset {} not found.".format(dataset_name))
        return
    
    print(
        "{} - nrows_x: {}, nrows_y: {}, tot_cols: {}, window_size: {}\n".format(
            dataset_name, nrows_x, nrows_y, tot_cols, window_size
        )
    )

    R = max(Rx, Ry)
    
    normalized = 0    
    try:
        os.mkdir("result/{}".format(dataset_name))
    except FileExistsError:
        pass

    if (np.abs(Rx - 1) < 1e-6) and (np.abs(Ry - 1) < 1e-6):
        normalized = 1
    
    print("Rx: {}, Ry: {}, R: {}".format(Rx, Ry, R))

    with open(
        "result/{}/socod_dx{}_dy{}_cols{}_ws{}.out".format(
            dataset_name, nrows_x, nrows_y, tot_cols, window_size
        ),
        "w",
    ) as output_file:
        output_file.write("SO_COD test begins.\n")
        output_file.write(
            "nrows_x: {}, nrows_y: {}, tot_cols: {}, window_size: {}\n".format(
                nrows_x, nrows_y, tot_cols, window_size
            )
        )
        output_file.flush()

        for eps in epsilon:
            
            L = int(np.ceil(1 / eps))

            SO_COD = SO_COD_Sketch_Layer(
                nrows_x, nrows_y, L * 2, eps * window_size, window_size, L, -1
            )

            if normalized == 0:
                Layers = int(np.ceil(np.log2(R)))
                limit_snapshot = int(np.ceil(4.01 / eps))
                print("layes: {}, limit_snapshot: {}".format(Layers, limit_snapshot))
                SO_COD = SO_COD_multilayers(
                    nrows_x, nrows_y, L * 2, eps * window_size, window_size, L, limit_snapshot, Layers
                )

            all_errors = []
            max_error = 0
            avg_error = 0
            sketch_size = []
            
            for i in trange(
                tot_cols,
                desc="SO-COD Processing with epsilon = {}".format(eps),
                ncols=100,
            ):
                SO_COD.update(X[:, i], Y[:, i])
                sketch_size.append(SO_COD.sketch_size())
                
                if i > 0 and i % (tot_cols // query_times) == 0:
                    A, B = SO_COD.query()

                    _X = X[:, max(i - window_size + 1, 0) : i + 1]
                    _Y = Y[:, max(i - window_size + 1, 0) : i + 1]
                    
                    error = compute_spectral_norm_XYT_ABT(_X, _Y, A, B)
                    
                    max_error = max(max_error, error)
                    all_errors.append(error)
                    avg_error = np.mean(all_errors)
            
            sketch_size = np.array(sketch_size)
            max_size = np.max(sketch_size)
            
            output_file.write(
                "eps: {}, Max error: {}, Avg error: {}, Sketch Size: {}\n".format(
                    eps, max_error, avg_error, max_size
                )
            )
            output_file.flush()


if __name__ == "__main__":
    pass