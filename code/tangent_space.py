import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures
from typing import Literal


class TangentSpaceEstimator:
    def __init__(self, data, intrinsic_dim):
        self._raw_data = np.asarray(data)
        self._n, self._D = data.shape
        self._data = self._raw_data

        print(f"Input {self.n} points, ambient dimension: {self.D}, intrinsic dimension: {intrinsic_dim}")

        self._d = intrinsic_dim

        self._coef = None
        self._tangent_space = None
        self._normal_space = None

        self.data_embedding = None

        self._k = max(10, 10 * math.floor(math.log10(self.n)))
        self._r = None

        self._eps = 1
        self._degree = 2

        self._neighbor_scale = None
        self._PCA = False

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def n(self):
        return self._n

    @property
    def data(self):
        return self._data

    @property
    def D(self):
        return self._D

    @property
    def d(self):
        return self._d

    @property
    def k(self):
        return self._k

    @property
    def r(self):
        return self._r

    @property
    def eps(self):
        return self._eps

    @property
    def degree(self):
        return self._degree

    # ------------------------------------------------------------------ #
    #  Private helpers: neighbor radius / count                           #
    # ------------------------------------------------------------------ #

    def _compute_r(self):
        nbrs = NearestNeighbors(n_neighbors=self._k + 1).fit(self._data)
        dist, _ = nbrs.kneighbors(self._data, return_distance=True)
        radius = np.max(dist[:, 1:], axis=1)
        return radius

    def _compute_k(self):
        nbrs = NearestNeighbors(radius=self._r).fit(self._data)
        _, neighbors = nbrs.radius_neighbors(self._data, return_distance=True, sort_results=True)
        neighbor_counts = np.array([len(neigh[1:]) for neigh in neighbors])
        return neighbor_counts

    # ------------------------------------------------------------------ #
    #  Parameter management                                               #
    # ------------------------------------------------------------------ #

    def setPara(self, **kwargs):
        """
        Update algorithm parameters and clear cached results.

        Parameters
        ----------
        k      : int    number of nearest neighbors (sets r to None)
        r      : float  neighborhood radius (sets k to None)
        eps    : float  bandwidth for Gaussian kernel weighting
        degree : int    polynomial degree for local regression (>= 1)
        """
        valid_keys = ['k', 'r', 'eps', 'degree', 'iter_times']
        for key in kwargs:
            if key not in valid_keys:
                raise ValueError(f"Invalid Parameters: {key}，Valid Parameters: {valid_keys}")

        if kwargs:
            print("Updated Parameters:")
            for key, value in kwargs.items():
                print(f"  {key} = {value}")

        if 'k' in kwargs:
            self._k = kwargs['k']
            self._r = None
            radius = self._compute_r()
            self._neighbor_scale = np.min(radius)
            print(f"  Neighbors Radius = [{np.min(radius)}, {np.max(radius)}]")

        if 'r' in kwargs:
            self._r = kwargs['r']
            self._neighbor_scale = self._r
            self._k = None
            neighbor_counts = self._compute_k()
            print(f"  Neighbors Count = [{np.min(neighbor_counts)}, {np.max(neighbor_counts)}]")
            print(f"  Average Neighbors Count = {np.mean(neighbor_counts)}")

        if 'eps' in kwargs:
            self._eps = kwargs['eps']

        if 'degree' in kwargs:
            self._degree = kwargs['degree']

        if kwargs:
            self._normal_space = None
            self._tangent_space = None
            self._coef = None

    # ------------------------------------------------------------------ #
    #  Data utilities                                                      #
    # ------------------------------------------------------------------ #

    def add_noise(self, R):
        """
        Add uniform ball noise to the data.

        Parameters
        ----------
        R : float   radius of the noise ball
        """
        n, D = self._n, self._D

        # random directions
        direction = np.random.normal(size=(n, D))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)

        # random radius
        u = np.random.rand(n, 1)
        radius = R * u ** (1.0 / D)

        noise = radius * direction
        self._data = self._raw_data + noise
        
        self._normal_space = None
        self._tangent_space = None
        self._coef = None

    # ------------------------------------------------------------------ #
    #  Private: weighted PCA                                              #
    # ------------------------------------------------------------------ #

    ### Local Weighted PCA
    def _weighted_PCA_tangent(self):
        """
        Estimate tangent space at each point via locally weighted PCA.

        Returns
        -------
        T     : ndarray, shape (n, D, d)   initial tangent basis at each point
        diffs : ndarray or list            displacement vectors to neighbors
        """
        
        n, D, d, data = self.n, self.D, self.d, self.data
        k, r, eps = self._k, self._r, self._eps

        if k is not None:
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
            neighbors = nbrs.kneighbors(data, return_distance=False)[:, 1:]

            diffs = data[neighbors] - data[:, None, :]
            dist2 = np.sum(diffs ** 2, axis=2)
            Ki = np.exp(-dist2 / 4 * eps ** 2)

            xi_bar = np.sum(Ki[:, :, None] * data[neighbors], axis=1) / np.sum(Ki, axis=1, keepdims=True)

            sqrtK = np.sqrt(Ki)[:, :, None]
            Z = sqrtK * (data[neighbors] - xi_bar[:, None, :])

            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
            T = Vt[:, :d, :]
            return T.transpose(0, 2, 1), diffs
        else:
            nbrs = NearestNeighbors(radius=r).fit(data)
            _, neighbors = nbrs.radius_neighbors(data, return_distance=True, sort_results=True)
            T = np.zeros((n, D, d))
            diffs = []
            for i in range(n):
                diff = data[neighbors[i][1:]] - data[i]
                diffs.append(diff)

                dist2 = np.sum(diff ** 2, axis=1)
                Ki = np.exp(-dist2 / 4 * eps ** 2)
                xi_bar = np.sum(Ki[:, None] * data[neighbors[i][1:]], axis=0) / np.sum(Ki)
                sqrtK = np.sqrt(Ki)[:, None]
                Z = sqrtK * (data[neighbors[i][1:]] - xi_bar)
                U, S, Vt = np.linalg.svd(Z, full_matrices=False)
                Ti = Vt[:d, :].T
                T[i] = Ti
            return T, diffs

    # ------------------------------------------------------------------ #
    #  Private: polynomial regression                                     #
    # ------------------------------------------------------------------ #

    def _compute_coef(self, T, diffs, degree):
        """
        Fit local polynomial regression in the tangent coordinate system.

        The i-th column of coef gives coefficients [a0, a1, ...] such that:
            f_i = a0 + a1*u + a2*v + a3*u^2 + a4*u*v + ...

        Parameters
        ----------
        T      : ndarray, shape (n, D, d)   tangent basis at each point
        diffs  : ndarray or list            displacement vectors to neighbors
        degree : int                        polynomial degree

        Returns
        -------
        coef : ndarray, shape (n, P, D)   regression coefficients at each point
        """
        n, D, d = self.n, self.D, self.d
        k, r = self.k, self.r

        poly = PolynomialFeatures(degree=degree, include_bias=True)
        if k is not None:
            local_coordinate = np.matmul(diffs, T)

            ### Make the Local Coordinate to Polynomial Form
            coordinate_poly = local_coordinate.reshape(-1, d)
            coordinate_poly = poly.fit_transform(coordinate_poly)
            P = coordinate_poly.shape[1]
            coordinate_poly = coordinate_poly.reshape(n, -1, P)

            ### Regression to Get the Coefficients for All Features
            ### coef will minimize |Design matrix * coef - Points (diffs)|
            ### Thus coef is (D^tD)^-1 D Y
            coordinate_poly_t = np.transpose(coordinate_poly, (0, 2, 1))
            coef = np.linalg.solve(coordinate_poly_t @ coordinate_poly, coordinate_poly_t @ diffs)
        else:
            feature_count = math.comb(d + degree, d)
            coef = np.zeros((n, feature_count, D))
            for i in range(n):
                local_coordinate = np.matmul(diffs[i], T[i])
                coordinate_poly = poly.fit_transform(local_coordinate)
                coef[i] = np.linalg.solve(coordinate_poly.T @ coordinate_poly, coordinate_poly.T @ diffs[i])
        return coef

    def _regression_coef(self):
        """
        Compute the final regression coefficients with iterative basis refinement.

        Iteratively re-estimates the local tangent basis from the previous
        regression coefficients before computing the final degree-polynomial fit.
        Result is cached in self._coef.

        Returns
        -------
        coef : ndarray, shape (n, P, D)   regression coefficients at each point
        """
        if self._coef is None:
            d = self.d
            T, diffs = self._weighted_PCA_tangent()

            ### Here should be degree + 1, but T,R of the last iteration will not be used. So we use 'degree'.
            for i in range(2, self._degree):
                coef = self._compute_coef(T, diffs, i)
                T = coef[:, 1:(d + 1), :] / np.linalg.norm(coef[:, 1:(d + 1), :], axis=2, keepdims=True)
                ### Make sure the basis are orthogonal
                T, R = np.linalg.qr(T.transpose(0, 2, 1))

            ### If no iteration above, the following is regression directly with degree.
            ### Here compute the final coef with degree we want
            self._coef = self._compute_coef(T, diffs, self._degree)

        return self._coef

    ### (u,v) -> (f(u,v), g(u,v), h(u,v))
    ### This method is best until now.
    def _local_coordiante_tangent(self):
        """
        Compute tangent space via local coordinate regression (best method).

        If self._PCA is True or degree == 1, falls back to weighted PCA.
        Otherwise runs iterative polynomial regression to refine the tangent basis.

        Returns
        -------
        T : ndarray, shape (n, D, d)   tangent basis at each point
        """
        n, D, d = self.n, self.D, self.d

        ### If we want PCA, just return PCA result. If not, we do not need call _weighted_PCA_tangent again.
        if self._PCA or self._degree == 1:
            T, diffs = self._weighted_PCA_tangent()
            return T

        ### Local PCA ==> Local Basis ==> Local Coordinate
        ### The 0-th row is [(df1/du), (df2/du), ...] at 0, which is the tangent basis.

        ### We can use more accurate tangent space to re-estimate local coordinate and repeat computing tangent basis, may improve accuracy.
        coef = self._regression_coef()
        T = coef[:, 1:(d + 1), :] / np.linalg.norm(coef[:, 1:(d + 1), :], axis=2, keepdims=True)
        ### Make sure the basis are orthogonal
        T, R = np.linalg.qr(T.transpose(0, 2, 1))
        return T

    # ------------------------------------------------------------------ #
    #  Public: tangent / norm space                                       #
    # ------------------------------------------------------------------ #

    def transfer_space(self, space):
        """
        Compute the orthogonal complement of a given subspace bundle.

        Given a tangent space bundle, returns the normal space, and vice versa.

        Parameters
        ----------
        space : ndarray, shape (n, D, dim)   subspace basis at each point

        Returns
        -------
        complement : ndarray, shape (n, D, D-dim)   orthogonal complement basis
        """
        U, S, Vt = np.linalg.svd(space, full_matrices=True)
        _, D, d = space.shape
        ### The last column is the orthogonal basis for "space"
        ### Dim(tangent space) + Dim(norm space) = D (ambient dimension)
        return U[:, :, -(D - d):]

    def tangent_space(self, **kwargs):
        """
        Compute and return the tangent space basis at each point.

        Result is cached; recomputed only if parameters change.

        Parameters
        ----------
        **kwargs : forwarded to setPara (e.g. k=20, degree=2)

        Returns
        -------
        T : ndarray, shape (n, D, d)   tangent basis at each point
        """
        if kwargs:
            self.setPara(**kwargs)
        if self._tangent_space is None:
            self._tangent_space = self._local_coordiante_tangent()
        return self._tangent_space

    def normal_space(self, **kwargs):
        """
        Compute and return the normal space basis at each point.

        Result is cached; recomputed only if parameters change.

        Parameters
        ----------
        **kwargs : forwarded to setPara (e.g. k=20, degree=2)

        Returns
        -------
        N : ndarray, shape (n, D, D-d)   normal basis at each point
        """
        if kwargs:
            self.setPara(**kwargs)
        if self._normal_space is None:
            self._normal_space = self.transfer_space(self.tangent_space())
        return self._normal_space

    def error_tangent_space(self, space, i=None):
        """
        Compare an estimated subspace against the computed tangent/normal space.

        Uses the largest principal angle between subspaces as the error metric.

        Parameters
        ----------
        space : ndarray, shape (n, D, dim) or (D, dim)
                Estimated subspace. Automatically matched to tangent (dim=d)
                or normal (dim=D-d) space for comparison.
        i     : int or array-like, optional
                Point index/indices to evaluate. If None, evaluates all points.

        Returns
        -------
        error : ndarray, shape (n,) or (len(i),)
                sin(largest principal angle), i.e. ||pi_U - pi_V||_op
        """
        ### decide tangent space or norm space according to dimension
        compare = self.tangent_space() if space.shape[-1] == self.d else self.normal_space()

        if i is not None:
            if isinstance(i, int):
                indices = np.array([i])
            else:
                indices = np.array(i)
            cos_theta = np.min(np.abs(np.linalg.eigvals(compare[indices].transpose(0, 2, 1) @ space)), axis=1)
            print(f"Compute the error at points: {i}")
        else:
            ### This will return the value of cos(the largest principal angle)
            cos_theta = np.min(np.abs(np.linalg.eigvals(compare.transpose(0, 2, 1) @ space)), axis=1)

        ### This is sin(the largest principal angle), a.k.a |pi_U - pi_V|_{op}
        return np.sqrt(np.clip(1 - cos_theta ** 2, 0, 1))