import numpy as np

def Subspace_Power_Method(X, Y, l, q = 5):
    np.random.default_rng(998247)
    mx = X.shape[0]
    my = Y.shape[0]
    G = np.random.normal(0, 1, (my, l))
    F = np.dot(Y.T, G)
    F = np.dot(X, F)
    
    for _ in range(q):
        F = np.dot(X.T, F)
        F = np.dot(Y, F)
        F = np.dot(Y.T, F)
        F = np.dot(X, F)
    
    Z, _ = np.linalg.qr(F)
    
    T = Z.T @ X
    T = T @ Y.T
    k = min(T.shape)
    U, S, VT = np.linalg.svd(T)
    U = U[:, :k]
    S = S[:k]
    VT = VT[:k, :]
    
    S = np.diag(np.sqrt(S))
    
    A = np.dot(np.dot(Z, U), S)
    B = np.dot(VT.T, S)
    
    return A, B