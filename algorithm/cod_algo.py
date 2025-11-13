import numpy as np

class COD_Sketch:
    
    def __init__(self, nrows_x, nrows_y, ncols, shrink_cols = 0):
        self.dx = nrows_x
        self.dy = nrows_y
        self.size = ncols
        self.used_cols = 0
        self.X = np.zeros((self.dx, self.size))
        self.Y = np.zeros((self.dy, self.size))
        self.shrink_cols = shrink_cols
    
        
        if(shrink_cols == 0):
            self.shrink_cols = np.min(ncols // 2,  ncols - 1)
        
        if(self.shrink_cols > min(nrows_x, nrows_y)):
            self.shrink_cols = min(nrows_x, nrows_y)
    
    def shrink(self):
        self.Qx, self.Rx = np.linalg.qr(self.X)
        self.Qy, self.Ry = np.linalg.qr(self.Y)
        self.K = np.dot(self.Rx, self.Ry.T)
        U, S, V = np.linalg.svd(self.K)
        if(self.shrink_cols < self.size and self.shrink_cols < min(self.dx, self.dy)):
            S = np.maximum(S - S[self.shrink_cols], 0)
        
        S = np.sqrt(S)
        
        self.used_cols = self.shrink_cols
        
        if(self.size <= min(self.dx, self.dy)):
            self.X = np.dot(self.Qx, np.dot(U, np.diag(S)))
            self.Y = np.dot(self.Qy, np.dot(V.T, np.diag(S)))
        else:
            S = np.diag(S)
            Sx = np.zeros((U.shape[1], self.size))
            Sx[:S.shape[0], :S.shape[1]] = S
            Sy = np.zeros((V.shape[1], self.size))
            Sy[:S.shape[0], :S.shape[1]] = S
            
            self.X = np.dot(self.Qx, np.dot(U, Sx))
            self.Y = np.dot(self.Qy, np.dot(V.T, Sy))
    
    def update(self, x, y):
        self.X[:, self.used_cols] = x
        self.Y[:, self.used_cols] = y
        self.used_cols += 1
        if self.used_cols == self.size:
            self.shrink()
    
    def sketch_size(self):
        return self.used_cols
    
    def retrieve(self):
        return self.X, self.Y

def compute_spectral_norm_XYT_ABT(X, Y, A, B):
    
    np.random.seed(99765)
    v = np.random.rand(Y.shape[0])
    v = v / np.linalg.norm(v)
    
    for _ in range(100):
        v_new = (X @ (Y.T @ v)) - (A @ (B.T @ v))
        v_new = (Y @ (X.T @ v_new)) - (B @ (A.T @ v_new))
        v_new = v_new / np.linalg.norm(v_new)
        
        if np.linalg.norm(v - v_new) < 1e-12:
            break
        
        v = v_new
            
    ans = np.linalg.norm((X @ (Y.T @ v)) - (A @ (B.T @ v)))
    
    if isinstance(X, np.ndarray):
        ans = ans / np.linalg.norm(X, 'fro') / np.linalg.norm(Y, 'fro')
        
    return ans


if __name__ == "__main__":
    pass
