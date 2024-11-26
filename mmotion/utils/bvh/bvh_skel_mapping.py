# When transferring numpy joints(in smpl skeleton) to bvhs, some mapping relations should be defined
smpl_reorder = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]
smpl_reorder_inv = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 18, 13, 15, 19, 16, 20, 17, 21]
smpl_end_points = [4, 8, 13, 17, 21]
smpl_parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
# smplh
smplh_reorder = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                 32, 33, 34, 35, 36, 14, 17, 19, 21, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
smplh_reorder_inv = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 33, 13, 15, 34, 16, 35, 17, 36, 18, 19, 20, 21, 22,
                     23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
smplh_end_points = [4, 8, 13, 20, 23, 26, 29, 32, 39, 42, 45, 48, 51]
smplh_parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17, 18, 19, 17, 21, 22
    , 17, 24, 25, 17, 27, 28, 17, 30, 31, 11, 33, 34, 35, 36, 37, 38, 36, 40, 41, 36, 43, 44, 36, 46
    , 47, 36, 49, 50]
reorders_mapping = dict(
    smpl=smpl_reorder,
    smplh=smplh_reorder
)

reorders_inv_mapping = dict(
    smpl=smpl_reorder_inv,
    smplh=smplh_reorder_inv
)

end_points_mapping = dict(
    smpl=smpl_end_points,
    smplh=smplh_end_points
)

parents_mapping = dict(
    smpl=smpl_parents,
    smplh=smplh_parents
)


def get_reorder(data_source: str):
    if data_source not in reorders_mapping:
        raise NotImplementedError(f"Unsupported data source: {data_source}, supported: {reorders_mapping.keys()}")
    return reorders_mapping[data_source]


def get_reorder_inv(data_source: str):
    if data_source not in reorders_inv_mapping:
        raise NotImplementedError(f"Unsupported data source {data_source}, supported: {reorders_inv_mapping.keys()}")
    return reorders_inv_mapping.get(data_source)


def get_end_points(data_source: str):
    if data_source not in end_points_mapping:
        raise NotImplementedError(f"Unsupported data source {data_source}, implemented: {end_points_mapping.keys()}")
    return end_points_mapping[data_source]


def get_parents(data_source: str):
    if data_source not in parents_mapping:
        raise NotImplementedError(f"Unsupported data source {data_source}, supported: {parents_mapping.keys()}")
    return parents_mapping.get(data_source)
