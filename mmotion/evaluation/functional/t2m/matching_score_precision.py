import numpy as np
import torch
# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
import torch.nn.functional as F


def euclidean_distance_matrix(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * torch.mm(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = torch.sum(torch.square(matrix1), dim=1, keepdims=True)  # shape (num_test, 1)
    d3 = torch.sum(torch.square(matrix2), dim=1)  # shape (num_train, )
    dists = torch.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def cal_mmdist_rprecision(A_embeddings,
                          B_embeddings,
                          r_precision_batch: int = 256,
                          top_k=3,
                          reduction=False):
    """
    :param A_embeddings: text or motion embeddings
    :param B_embeddings: motion or text embeddings
    :param r_precision_batch: r precision batch size
    :param top_k: top_3 as default
    :param reduction: whether cal mean
    :return:
    """
    num_samples = A_embeddings.shape[0]
    mm_dist = 0
    top_k_mat = torch.zeros((top_k,))
    # matching score and r-precision calculation
    for i in range(num_samples // r_precision_batch):
        group_A = F.normalize(
            A_embeddings[i * r_precision_batch: (i + 1) * r_precision_batch])
        group_B = F.normalize(
            B_embeddings[i * r_precision_batch: (i + 1) * r_precision_batch])

        dist_mat = euclidean_distance_matrix(
            group_A, group_B
        ).float()

        mm_dist += dist_mat.trace()
        argsmax = torch.argsort(dist_mat, dim=1)
        top_k_mat += calculate_top_k(argsmax, top_k=top_k).sum(axis=0)
    if reduction:
        valid_num_samples = num_samples // r_precision_batch * r_precision_batch
        mm_dist /= valid_num_samples
        top_k_mat /= valid_num_samples
    return mm_dist, top_k_mat


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = (
        torch.unsqueeze(torch.arange(size), 1).to(mat.device).repeat_interleave(size, 1)
    )
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        #         print(correct_vec, bool_mat[:, i])
        correct_vec = correct_vec | bool_mat[:, i]
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = torch.cat(top_k_list, dim=1)
    return top_k_mat
