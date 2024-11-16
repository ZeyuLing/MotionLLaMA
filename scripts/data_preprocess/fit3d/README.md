Download Fit3D dataset from https://fit3d.imar.ro/fit3d.

Note that the test subset contains no smplx, so we only use the train subset.

1. Run following Code to get annotation file of Fit3D:

`python scripts/data_preprocess/fit3d/get_anno.py`

2. Run `python scripts/data_preprocess/smplx2hm3d.py data/motionhub/fit3d/train.json` to get the humanml3d motion vectors.

3. Run `python scripts/data_preprocess/smplx2interhuman.py data/motionhub/fit3d/train.json` to get the interhuman motion vectors.
