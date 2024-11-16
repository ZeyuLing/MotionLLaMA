from scipy.ndimage import gaussian_filter


def motion_temporal_filter(motion, sigma=1):
    """
    :param motion: t c
    :param sigma:
    :return:
    """
    # print(motion.shape)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)