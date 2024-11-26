Following instructions in https://github.com/IDEA-Research/Motion-X/tree/main 
to download and process Motion-X dataset.

Place the Motion-X dataset at ./data/motionhub/motionx
1. Get the standard smplx dict files(npz) with `python scripts/data_preprocess/motionx/standardize_smplx.py`

2. Run the following code to generate the annotation file:

`python scripts/data_preprocess/motionx/get_anno.py`

3. [Optional] Run following code to get the humanml3d motion vectors:
```
python scripts/data_preprocess/smplx2hm3d.py data/motionhub/motionx/train.json
python scripts/data_preprocess/smplx2hm3d.py data/motionhub/motionx/val.json
python scripts/data_preprocess/smplx2hm3d.py data/motionhub/motionx/test.json
```

4. Some caption of humman subset have errors, run following code to modify.
```bash
python scripts/data_preprocess/motionx/modify_humman_caption.py
```

5. Run following code to get the interhuman motion vectors:
```
python scripts/data_preprocess/smplx2interhuman.py data/motionhub/motionx/train.json
python scripts/data_preprocess/smplx2interhuman.py data/motionhub/motionx/val.json
python scripts/data_preprocess/smplx2interhuman.py data/motionhub/motionx/test.json
```