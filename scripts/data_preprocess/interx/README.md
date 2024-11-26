Download Inter-X Dataset from https://github.com/liangxuy/Inter-X

1. seperate caption for each person with code `python scripts/data_preprocess/interx/sep_caption.py`
2. Run `python scripts/data_preprocess/interx/standardize_smplx.py`
3. Get annotation files with `python scripts/data_preprocess/interx/get_anno.py`
4. [Optional] Get the humanml3d motion vectors with 
```
python scripts/data_preprocess/smplx2hm3d.py data/motionhub/interx/train.json
python scripts/data_preprocess/smplx2hm3d.py data/motionhub/interx/val.json
python scripts/data_preprocess/smplx2hm3d.py data/motionhub/interx/test.json
```

5. Get the interhuman motion vectors with 
```
python scripts/data_preprocess/smplx2interhuman.py data/motionhub/interx/train.json
python scripts/data_preprocess/smplx2interhuman.py data/motionhub/interx/val.json
python scripts/data_preprocess/smplx2interhuman.py data/motionhub/interx/test.json
```