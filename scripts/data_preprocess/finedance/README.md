Download FineDance dataset from https://github.com/li-ronghui/FineDance.
1. Run `python scripts/data_preprocess/finedance/standardize_smplx.py`

2. Run following Code to get annotation file of FineDance:

`python scripts/data_preprocess/finedance/get_anno.py`

3. Run `python scripts/data_preprocess/smplx2hm3d.py data/motionhub/finedance/train.json` to get the humanml3d motion vectors.

4. Run `python scripts/data_preprocess/smplx2interhuman.py data/motionhub/finedance/train.json` to get the interhuman motion vectors.

