import numpy as np

def get_normaliezd(X):
    cor = min(np.linalg.norm(X[:, col]) for col in range(X.shape[1]))
    return X / cor

def Noisy_Low_Rank_Matrix(n, m, k, noise_level):
    
    np.random.default_rng(679321 + n + m + k)
    X = np.random.normal(0, 1, (n, k))
    S = [1 - _ / k for _ in range(k)]
    S = np.diag(S)
    
    G = np.random.uniform(-1, 1, (m, k))
    Y, _ = np.linalg.qr(G, mode = 'reduced')
    
    Res = np.dot(X, np.dot(S, Y.T)) + np.random.normal(0, 1, (n, m)) / noise_level
    return Res 

def load_data(dataset_name):
    
    if dataset_name == "uniform":
        np.random.default_rng(998247)
        X = np.random.rand(2000, 10000)
        Y = np.random.rand(1000, 10000)
        
        X = get_normaliezd(X)
        Y = get_normaliezd(Y)
        
        Rx = max(np.linalg.norm(X[:, col]) for col in range(X.shape[1])) ** 2
        Ry = max(np.linalg.norm(Y[:, col]) for col in range(Y.shape[1])) ** 2
        
        return X, Y, X.shape[0], Y.shape[0], X.shape[1], 4000, Rx, Ry
    
    if dataset_name == "Random_Noisy":
        
        X = Noisy_Low_Rank_Matrix(2000, 10000, 400, 100)
        Y = Noisy_Low_Rank_Matrix(1000, 10000, 400, 50)
        
        X = get_normaliezd(X)
        Y = get_normaliezd(Y)
        
        Rx = max(np.linalg.norm(X[:, col]) for col in range(X.shape[1])) ** 2
        Ry = max(np.linalg.norm(Y[:, col]) for col in range(Y.shape[1])) ** 2
        
        return X, Y, X.shape[0], Y.shape[0], X.shape[1], 4000, Rx, Ry

    if dataset_name == 'hpmax':
        X_flat = np.fromfile('data/B_16384_32768_10per_Double.bin', dtype=np.float64)
        Y_flat = np.fromfile('data/A_32768_16384_10per_Double.bin', dtype=np.float64)
        
        X = X_flat.reshape((16384, 32768))
        Y = Y_flat.reshape((32768, 16384))
        
        Y = Y.T
        
        X = np.asfortranarray(X)
        Y = np.asfortranarray(Y)
        
        X = get_normaliezd(X)
        Y = get_normaliezd(Y)
        
        Rx = max(np.linalg.norm(X[:, col]) for col in range(X.shape[1])) ** 2
        Ry = max(np.linalg.norm(Y[:, col]) for col in range(Y.shape[1])) ** 2
        
        return X, Y, X.shape[0], Y.shape[0], X.shape[1], 10000, Rx, Ry
    
    if dataset_name == "multimodal":
        X = np.load('data/image_matrix.npy')
        Y = np.load('data/text_matrix.npy')
        X = X.T
        Y = Y.T
        
        X = np.asfortranarray(X)
        Y = np.asfortranarray(Y)
        
        X = get_normaliezd(X)
        Y = get_normaliezd(Y)
        
        Rx = max(np.linalg.norm(X[:, col]) for col in range(X.shape[1])) ** 2
        Ry = max(np.linalg.norm(Y[:, col]) for col in range(Y.shape[1])) ** 2
        
        return X, Y, X.shape[0], Y.shape[0], X.shape[1], 10000, Rx, Ry
        
if __name__ == "__main__":
    pass