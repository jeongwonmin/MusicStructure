import librosa
import librosa.core
import numpy as np
import scipy.signal as sig
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import itertools

import draw_tool as dwts
from music_reader import MusicReader
import calculator as calc


class ClusterTool(object):
    def __init__(self, window):
        self._window = window
            
    def _nn_graph(self, cqt_array, k):
        nbrs = NearestNeighbors(n_neighbors=k,
                        metric='euclidean'
                ).fit(cqt_array)
        r_ij = nbrs.kneighbors_graph(cqt_array).toarray().astype(np.bool)
        r_ij_t = r_ij.transpose()
        r_ij_bool = np.logical_and(r_ij, r_ij_t)
        r_ij = r_ij_bool.astype(np.float)
        return nbrs, r_ij
    
    # generate R^{¥prime} from a mutual nearest neighbor graph
    def _calc_rprime(self, nng):
        # extend border of nng
        def rotate_extend(sq_mat):
            ww = self._window
            n = sq_mat.shape[0]
            ret = np.zeros((n+ww*2, n+ww*2),dtype=np.int)
            ret[:ww,:ww] = sq_mat[n-ww:,n-ww:]
            ret[:ww,ww:n+ww] = sq_mat[n-ww:,:]
            ret[:ww,n+ww:] = sq_mat[n-ww:,:ww]
            ret[ww:n+ww,:ww] = sq_mat[:,n-ww:]
            ret[ww:n+ww,ww:n+ww] = sq_mat[:,:]
            ret[ww:n+ww,n+ww:] = sq_mat[:,:ww]
            ret[ww+n:,:ww] = sq_mat[:ww,n-ww:]
            ret[ww+n:,ww:ww+n] = sq_mat[:ww,:]
            ret[ww+n:,ww+n:] = sq_mat[:ww,:ww]
            return ret
        
        rp = rotate_extend(nng)
        rp = calc.majority_vote(rp, self._window)
        return rp
    
    def _calc_delta(self, rprime):
        n = rprime.shape[0]
        delta_u = np.eye(n, n, k=1, dtype=np.int)
        delta_l = np.eye(n, n, k=-1, dtype=np.int)
        delta_ = delta_u + delta_l
        return delta_
    
    """
    def calc_mu(self, rprime, delta):
        return calc.mu(rprime, delta)
    """
    
    def _calc_sigma(self, nbr_obj, cqt_mfcc):
        distances, indices = nbr_obj.kneighbors(cqt_mfcc)
        dist_sums = np.sum(distances**2, axis=0)
        dist_sum = dist_sums[-1]
        sigma = dist_sum / cqt_mfcc.shape[0]
        return sigma
    
    def _affine(self, record):
        C_t = record[1][0]
        mfcc_t = record[1][1]
        k = record[1][2]
        r = self._nn_graph(C_t, k)
        nbrs_mfcc = NearestNeighbors(n_neighbors=k,
                        metric='euclidean'
                        ).fit(mfcc_t)
        
        rp = self._calc_rprime(r[1])
        dlt = self._calc_delta(rp)
        sigma_rp = self._calc_sigma(r[0], C_t)
        sigma_dlt = self._calc_sigma(nbrs_mfcc, mfcc_t)
        
        srep = calc.srep(calc.square_distance_matrix_all(C_t), sigma_rp)
        sloc = calc.srep(calc.square_distance_matrix_all(mfcc_t), sigma_dlt)
    
        mu = calc.mu(rp, dlt)
        A = mu*np.multiply(rp, srep) + (1-mu) * np.multiply(dlt, sloc)
        return A
    
    def _laplacian(self, record):
        a = self._affine(record)
        D_sqrt_inv = np.sqrt(np.reciprocal(np.sum(a, axis=1)))
        id_mat = np.eye(a.shape[0], dtype=np.float)
        dsqt_a_dsqt = calc.matmul_a_diag(
            calc.matmul_diag_a(D_sqrt_inv, a), D_sqrt_inv)
        L = id_mat - dsqt_a_dsqt
        return L
    
    def music_clusters(self, record):
        L = self._laplacian(record)
        eigvec = calc.eigenvector(L)
        def normalize(vecs):
            norm = np.tile(np.linalg.norm(vecs, axis=1), (vecs.shape[1], 1)).transpose()
            n_vecs = vecs / (norm + np.tile(
                np.repeat(1.0e-16, vecs.shape[0]), (vecs.shape[1], 1)).transpose()) 
            return n_vecs
        
        clusters = []
        for i in range(1, 11):
            eigvec_sub = eigvec[:,:i]
            n_eigvec = normalize(eigvec)
            pred = KMeans(n_clusters=i).fit_predict(n_eigvec)
            clusters.append(pred)
        
        return np.array(clusters)
    
    # record : includes music file name, music matrices
    def __call__(self, record):
        return self.music_clusters(record)
