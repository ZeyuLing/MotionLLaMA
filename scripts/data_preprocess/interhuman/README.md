Download InterHuman Dataset from https://drive.google.com/drive/folders/1oyozJ4E7Sqgsr7Q747Na35tWo5CjNYk3

Place the downloaded InterHuman dataset at ./data/motionhub/interhuman
Please Manually delete 3945, 4106 since the motions are empty.
Manually delete 5236, 5237, 5238 since the motions are in bad quality. 

TODO: Get the annotated file.

1. Get the separate caption for each person with `python scripts/data_preprocess/interhuman/sep_caption.py`

2. Separate persons from the original pkl file with following code:
    `python scripts/data_preprocess/interhuman/sep_person.py`

3. Run `python scripts/data_preprocess/interhuman/standardize_smplx.py`

4. Get the annotation file with:
    `python scripts/data_preprocess/interhuman/gen_anno.py`
5. [Optional]Get the humanml3d motion vectors
```
python scripts/data_preprocess/smplx2hm3d.py data/motionhub/interhuman/train.json
python scripts/data_preprocess/smplx2hm3d.py data/motionhub/interhuman/val.json
python scripts/data_preprocess/smplx2hm3d.py data/motionhub/interhuman/test.json
```
6. [Optional]Get the interhuman motion vectors
```
python scripts/data_preprocess/smplx2interhuman.py data/motionhub/interhuman/train.json
python scripts/data_preprocess/smplx2interhuman.py data/motionhub/interhuman/val.json
python scripts/data_preprocess/smplx2interhuman.py data/motionhub/interhuman/test.json
```