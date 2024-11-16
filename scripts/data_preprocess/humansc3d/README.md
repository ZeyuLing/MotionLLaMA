Download HumanSC3D dataset from http://sc3d.imar.ro/humansc3d.

Note that the test subset contains no smplx, so we only use the train subset.

1. Run `python scripts/data_preprocess/humansc3d/standardize_smplx.py`

2. Run following Code to get annotation file of HumanSC3D:

`python scripts/data_preprocess/humansc3d/get_anno.py`

3. Run following Code to get annotation file of HumanSC3D:

`python scripts/data_preprocess//get_anno.py`


4. [Optional] Run `python scripts/data_preprocess/smplx2hm3d.py data/motionhub/humansc3d/train.json` to get the humanml3d motion vectors.
5. Run `python scripts/data_preprocess/smplx2interhuman.py data/motionhub/humansc3d/train.json` to get the interhuman motion vectors.