import numpy as np
from tangent_space import TangentSpaceEstimator


class ReachEstimator(TangentSpaceEstimator):
    """
    Extends TangentSpaceEstimator with reach estimation methods.

    Local reach  : 1 / max principal curvature at each point
    Global reach : min over all point pairs of |x-y|^2 / (2 * d(x, T_y M))
    Reach        : min(local reach, global reach)
    """

    # ------------------------------------------------------------------ #
    #  Public: reach                                                      #
    # ------------------------------------------------------------------ #

    def local_reach(self, **kwargs):
        """
        Compute the local reach at every point.

        Returns
        -------
        local_reach : ndarray, shape (n,)
            1 / max|principal curvature| at each point.
        """
        if kwargs:
            self.setPara(**kwargs)

        n, D, d = self.n, self.D, self.d

        if self._degree < 2:
            raise ValueError(
                f"Polynomial degree {self._degree} has no quadratic term; "
                "cannot compute local reach."
            )

        coef = self._regression_coef()

        tangent_vector = coef[:, 1:(d + 1), :]

        # Metric tensor G and its inverse
        G = tangent_vector @ tangent_vector.transpose(0, 2, 1)
        G_inv = np.linalg.inv(G)

        # Second-order coefficients  d²fk / dui duj
        coef_ij = coef[:, (d + 1):(1 + int((d + 3) * d / 2)), :]

        # Shape operator projected onto normal space
        L = coef_ij @ self.normal_space()

        i_upper, j_upper = np.triu_indices(d)
        L_mat = np.zeros((n, d, d, D - d))
        L_mat[:, i_upper, j_upper, :] = L
        L_mat = L_mat + L_mat.transpose(0, 2, 1, 3)

        S = np.einsum('nij,njka->naik', G_inv, L_mat)

        principal_curvatures = np.linalg.eigvals(S)

        return 1 / np.max(np.abs(principal_curvatures), axis=(1, 2))

    def reach_matrix(self, **kwargs):
        """
        Compute the pairwise reach matrix  |x-y|² / (2 d(x, T_y M)).

        Returns
        -------
        rea        : ndarray, shape (n, n)   (Inf for invalid pairs)
        local_r    : ndarray, shape (n,)     local reach at each point
        """
        if kwargs:
            self.setPara(**kwargs)

        n, D = self.n, self.D
        P = self.normal_space()
        data = self.data

        diff = data[None, :, :] - data[:, None, :]
        normal_part = diff @ P

        # Squared distance to tangent plane
        TangentDistance = np.linalg.norm(normal_part, axis=2)

        sq = np.sum(data * data, axis=1, keepdims=True)
        Dis2 = sq + sq.T - 2 * (data @ data.T)
        Dis2 = np.maximum(Dis2, 0)

        local_r = self.local_reach()
        tau_l = np.min(local_r)

        with np.errstate(divide='ignore', invalid='ignore'):
            rea = Dis2 / (2 * TangentDistance)

        local_radius = (
            self._neighbor_scale
            if self._neighbor_scale is not None
            else np.min(self._compute_r_fallback())
        )

        invalid_mask = (
            (~np.isfinite(rea))
            | (Dis2 >= (2 * tau_l) ** 2)
            | (Dis2 <= local_radius ** 2)
        )
        rea[invalid_mask] = np.inf

        self._reach_matrix_cache = rea
        return rea, local_r

    def compute_reach(self, **kwargs):
        """
        Compute the reach of the manifold.

        Returns
        -------
        reach    : float   estimated reach τ(M)
        reach_pt : index of the point that achieves the minimum
        """
        if kwargs:
            self.setPara(**kwargs)

        global_r, local_r = self.reach_matrix()

        if np.min(local_r) < np.min(global_r):
            reach = np.min(local_r)
            reach_pt = np.argmin(local_r)
        else:
            reach = np.min(global_r)
            reach_pt = np.unravel_index(np.argmin(global_r), global_r.shape)

        return reach, reach_pt
