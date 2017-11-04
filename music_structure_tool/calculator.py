import numpy as np
from sklearn.neighbors import NearestNeighbors

import speedup_library as spl


def majority_vote(square_array, window):
    N = square_array.shape[0]
    spl.majority_vote(square_array, window)
    return square_array[:N-2*window, :N-2*window]

# 音楽をサンプリングした元データを入力としている
def square_distance_matrix_all(array):
    dists_square = np.zeros((array.shape[0], array.shape[0]), dtype=np.float)
    spl.distance_matrix_all(array, dists_square)
    return dists_square

def mu(Rp_ij, delta):
    # calculate mu
    array1_d = np.sum(Rp_ij, axis=1)
    delta_d = np.sum(delta, axis=1)
    
    lower = np.sum((array1_d + delta_d)**2)
    upper = np.sum(np.multiply(delta_d, (array1_d+delta_d)))
    
    return upper / lower

def srep(dist_matrix_squared, sigma_2):
    spl.srep(dist_matrix_squared, sigma_2)
    return dist_matrix_squared

def matmul_diag_a(diag, a):
    result = np.zeros(a.shape, dtype=np.float)
    spl.matmul_diag_a(diag, a, result)
    return result
    
def matmul_a_diag(a, diag):
    result = np.zeros(a.shape, dtype=np.float)
    spl.matmul_a_diag(a, diag, result)
    return result
    
def eigenvector(a):
    return spl.eigenvector(a)