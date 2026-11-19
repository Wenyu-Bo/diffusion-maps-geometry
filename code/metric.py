import numpy as np
from sklearn.neighbors import NearestNeighbors
from math import gamma, pi
from tangent_space import TangentSpaceEstimator
from src.geometry import Points, Embedding
from src.geometry.embedding import laplacian_embedding
from src.linalg import RMetric

class LearnMetric(TangentSpaceEstimator):
    def __init__(self, data, intrinsic_dim, lam=1.0):
        TangentSpaceEstimator.__init__(self, data, intrinsic_dim)
        self.lam = lam


    def laplacian(self, epsilon = 1/3, radius = None):
        X = Points(self.data)
        eps = 2 * epsilon 
        dist_mat = X.pairwise_distance(dist_type="euclidean")
        if radius is not None:
            dist_mat = dist_mat.threshold(radius=radius, in_place=True)
        aff_mat = dist_mat.affinity(aff_type="gaussian", eps=eps, in_place=False)
        self.L = aff_mat.laplacian(lap_type="geometric", diag_add=1.0, aff_minus_id=True)
        
        return self.L.as_nparray()
    
    
    def __normalize(self, V, epsilon):
        X = self.data
        
        nbrs = NearestNeighbors(radius=epsilon).fit(X)
        neighborhoods = nbrs.radius_neighbors(X, return_distance=False)
        N = np.array([len(neigh) for neigh in neighborhoods], dtype=float)
        N[N == 0] = 1.0
        
        omega_d_1 = 2 * pi**((self.d)/2) / gamma((self.d)/2)
        const = omega_d_1 * epsilon**self.d / self.d
        for k in range(V.shape[1]):
            norm_sq = const * np.sum(V[:, k]**2 / N)
            V[:, k] /= np.sqrt(norm_sq)

        return V
    
    
    def _laplacian_eigenmap_without_t(
            self,
            L = None,       
            epsilon: float = 0.5, 
            radius: float = np.inf,
            m: int = 3, 
        ):
        '''
        r = radius
        h = bandwidth
        m = embedding dimension
        '''
        X = self.data
    
        if L is None:
            self.laplacian(epsilon=epsilon, radius=radius)
        
        emb_sys = laplacian_embedding(self.L, ncomp=m, eigen_solver="dense", drop_first=True)
        eigvecs = emb_sys.eigenvectors.as_nparray()
                    
        self.vecs = self.__normalize(eigvecs, epsilon=epsilon)
        self.vals = emb_sys.eigenvalues.as_nparray()
        

    def eigenmap_t(self, t, m):
        const = (2 * t) ** ((self.d + 2) / 4) * np.sqrt(2) * (4 * np.pi) ** (self.d / 4)
        eig_part = self.vecs[:, :m] * np.exp(-self.vals[:m] * t)
        return  const * eig_part

    
    def laplacian_eigenmap(
            self,
            L = None,       
            epsilon: float = 0.5,
            radius: float = None, 
            t: float = 1.0,
            m: int =3, 
        ):
        '''
        h = bandwidth
        t = time
        m = embedding dimension
        '''

        self._laplacian_eigenmap_without_t(L=L, epsilon=epsilon, radius=radius, m=m)
        self.X_embedding = self.eigenmap_t(t=t, m=m)
        
        return  self.X_embedding
    
    
    def learnmetric(self, F, bsize = 1000):
        rmetric = RMetric.local(
                    x_pts=Embedding(F),
                    lap=self.L,
                    dual=True,
                    bsize=bsize
                    )
        return rmetric
    
    
    def distortion(self, L = None, epsilon = None, radius = None, indices = None):
        data = self.data

        if L is None:
            L = self.laplacian(epsilon=epsilon, radius=radius)
        T = self.tangent_space()
        
        if indices is None:
            diff = data[None, :, :] - data[:, None, :]
            Y = np.matmul(diff, T)
            H = np.matmul(Y.transpose(0,2,1), Y * L[:,:,None])
        else:
            diff = data[None, :, :] - data[indices, None, :]
            Y = np.matmul(diff, T[indices])
            H = np.matmul(Y.transpose(0,2,1), Y * L[indices,:,None])
            
        return H
    
    
    
    # def pseudoinverse(self, H_tilde):
    #     n, s, _ = H_tilde.shape
        
    #     vals, vecs = np.linalg.eigh(H_tilde)

    #     top_d_indices = np.argsort(vals, axis=1)[:, -self.d:]
    #     vals_d = np.take_along_axis(vals, top_d_indices, axis=1)  
    #     vecs_d = np.take_along_axis(
    #         vecs, 
    #         top_d_indices[:, None, :].repeat(s, axis=1),
    #         axis=2
    #     )
    #     inv_vals = 1.0 / vals_d
        
    #     H = np.einsum('nik,nk,njk->nij', vecs_d, inv_vals, vecs_d)
        
    #     return H
    
    # def learnmetric(self, F = None):
    #     n = self.n
    #     L = self.L
    #     if F is None:
    #         F = self.data
    #     # F = self.data
    #     m = F.shape[1]

    #     H_tilde = np.zeros((n, m, m))
    #     LF = L @ F
    #     F_prod = F[:, :, None] * F[:, None, :]
    #     L_F_prod = L @ F_prod.reshape(n, -1)
    #     L_F_prod = L_F_prod.reshape(n, m, m)

    #     cross = (
    #         F[:, :, None] * LF[:, None, :] +
    #         F[:, None, :] * LF[:, :, None]
    #     )

    #     H_tilde = 0.5 * (L_F_prod - cross)

    #     H = self.pseudoinverse(H_tilde)
    #     return F, H

    
    # def laplacian_batch(self, X, epsilon, radius = np.inf):
    #     B, n, _ = X.shape

    #     diff = X[:, :, None, :] - X[:, None, :, :]
    #     sq_dists = np.sum(diff ** 2, axis=-1)

    #     W = np.zeros_like(sq_dists)
    #     mask = sq_dists <= radius ** 2
    #     W[mask] = np.exp(-sq_dists[mask] / (4 * epsilon ** 2))

    #     d = W.sum(axis=2)
    #     D_lambda_inv = d ** (-self.lam)

    #     W_tilde = (
    #         D_lambda_inv[:, :, None] *
    #         W *
    #         D_lambda_inv[:, None, :]
    #     )

    #     d_tilde = W_tilde.sum(axis=2)
    #     W_norm = W_tilde / d_tilde[:, :, None]

    #     I = np.eye(n)[None, :, :]
    #     L = (1 / epsilon ** 2) * (W_norm - I)

    #     return L
    
    # def distortion(self, L = None, epsilon = None, radius = None, L_r = None):

    #     data, n, d = self.data, self.n, self.d

    #     if L is None:
    #         L = self.laplacian(epsilon=epsilon, radius=L_r)
        
    #     T = self.tangent_space()
        
    #     # if k is not None:
    #     #     nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
    #     #     neighbors = nbrs.kneighbors(data, return_distance=False)
    #     #     diffs = data[neighbors] - data[:, None, :]
    #     #     Y = np.matmul(diffs, T)
    #     #     rows = np.arange(n)[:, None]
            
    #     #     # L = self.laplacian_batch(diffs, eps)
            
    #     #     L_i = L[rows, neighbors]
    #     #     H = np.matmul(Y.transpose(0,2,1), Y * L_i[:,:,None])
            
    #     #     # H = np.matmul(Y.transpose(0,2,1), Y * L.transpose(0,2,1)[:,:,0][:,:,None])
            
    #     nbrs = NearestNeighbors(radius=radius).fit(data)
    #     _, neighbors = nbrs.radius_neighbors(data, return_distance=True, sort_results=True)
    #     H = np.zeros((n,d,d))

    #     for i in range(n):
    #         diff = data[neighbors[i]] - data[i]
    #         Y = np.matmul(diff, T[i])
    #         L_i = L[i, neighbors[i]]
    #         H[i] = Y.T @ (Y * L_i[:, None])
    
    #     return H
    
    

         