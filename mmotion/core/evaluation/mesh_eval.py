# ------------------------------------------------------------------------------
# Adapted from https://github.com/akanazawa/hmr
# Original licence: Copyright (c) 2018 akanazawa, under the MIT License.
# ------------------------------------------------------------------------------
import torch


def compute_similarity_transform(source_points,
                                 target_points,
                                 return_tform=False):
    """Computes a similarity transform (sR, t) that takes a set of 3D points
    source_points (N x 3) closest to a set of 3D points target_points, where R
    is an 3x3 rotation matrix, t 3x1 translation, s scale.

    And return the
    transformed 3D points source_points_hat (N x 3). i.e. solves the orthogonal
    Procrutes problem.
    Notes:
        Points number: N
    Args:
        source_points (np.ndarray([N, 3])): Source point set.
        target_points (np.ndarray([N, 3])): Target point set.
        return_tform (bool) : Whether return transform
    Returns:
        source_points_hat (np.ndarray([N, 3])): Transformed source point set.
        transform (dict): Returns if return_tform is True.
            Returns rotation: r, 'scale': s, 'translation':t.
    """
    assert target_points.shape[0] == source_points.shape[0]
    assert target_points.shape[1] == 3 and source_points.shape[1] == 3

    source_points = source_points.float().T
    target_points = target_points.float().T

    # 1. Remove mean.
    mu1 = source_points.mean(dim=1, keepdims=True)
    mu2 = target_points.mean(dim=1, keepdims=True)
    X1 = source_points - mu1
    X2 = target_points - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.matmul(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, _, Vh = torch.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0]).to(U)
    Z[-1, -1] *= torch.sign(torch.linalg.det(U.matmul(V.T)))
    # Construct R.
    R = V.matmul(Z.matmul(U.T))

    # 5. Recover scale.
    scale = torch.trace(R.matmul(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.matmul(mu1))

    # 7. Transform the source points:
    source_points_hat = scale * R.matmul(source_points) + t

    source_points_hat = source_points_hat.T

    if return_tform:
        return source_points_hat, {
            'rotation': R,
            'scale': scale,
            'translation': t
        }

    return source_points_hat
