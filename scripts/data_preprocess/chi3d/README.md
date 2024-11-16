Download Chi3D dataset from https://ci3d.imar.ro/download.

Note that the test subset contains no smplx, so we only use the train subset.

1. Separate persons in each smplx json file with following code:
`python scripts/data_preprocess/chi3d/sep_person.py`

2. Run `scripts/data_preprocess/chi3d/standardize_smplx.py` to get the standard smpl dict.

3. Run following Code to get annotation file of Chi3D:

`python scripts/data_preprocess/chi3d/get_anno.py`

4. [Optional] Run `python scripts/data_preprocess/smplx2hm3d.py data/motionhub/chi3d/train.json` to get the humanml3d motion vectors.
5. Run `python scripts/data_preprocess/smplx2interhuman.py data/motionhub/chi3d/train.json` to get the interhuman motion vectors.