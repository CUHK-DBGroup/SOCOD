from .cod_algo import COD_Sketch
import numpy as np

class COD_Block(COD_Sketch):
    def __init__(self, nrows_x, nrows_y, ncols, shrink_cols = 0, timestamp = 0):
        super().__init__(nrows_x, nrows_y, ncols, shrink_cols)
        self.sizex = 0
        self.sizey = 0
        self.size_norm = 0
        self.start_time = timestamp
        self.end_time = timestamp
    
    def update(self, x, y, shrink_tag = False):
        self.X[:, self.used_cols] = x
        self.sizex += np.sum(np.square(x))
        self.Y[:, self.used_cols] = y
        self.sizey += np.sum(np.square(y))
        self.size_norm = np.sqrt(self.sizex * self.sizey)
        self.used_cols += 1
        
        if shrink_tag and self.used_cols == self.size:
            self.shrink()
        
    
    def merge(self, other):
        self.X[:, self.used_cols:self.used_cols + other.used_cols] = other.X[:, :other.used_cols]
        self.Y[:, self.used_cols:self.used_cols + other.used_cols] = other.Y[:, :other.used_cols]
        self.used_cols += other.used_cols
        self.sizex += other.sizex
        self.sizey += other.sizey
        self.size_norm = np.sqrt(self.sizex * self.sizey)
        self.start_time = np.minimum(self.start_time, other.start_time)
        self.end_time = np.maximum(self.end_time, other.end_time)
        if(self.used_cols > self.shrink_cols):
            self.shrink()

if __name__ == "__main__":
    pass
    
       
    
    
    