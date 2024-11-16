Download Beat_v2 dataset from https://drive.google.com/drive/folders/1ukbifhHc85qWTzspEgvAxCXwn9mK4ifr.

Place the downloaded BEATv2 dataset at ./data/motionhub/beat_v2.0.0

1. Run `python scripts/data_preprocess/beat/standardize_smplx.py` to standardize the smplx file format in BEAT.

2. Run following code to get the annotation file of BEAT_v2 dataset.

`python scripts/data_preprocess/beat/get_anno.py`


3. [Optional] Run
```
python scripts/data_preprocess/smplx2hm3d.py data/motionhub/beat_v2.0.0/train.json
python scripts/data_preprocess/smplx2hm3d.py data/motionhub/beat_v2.0.0/test.json

``` 
to get the humanml3d motion vectors.

4.Run 
```
python scripts/data_preprocess/smplx2interhuman.py data/motionhub/beat_v2.0.0/train.json
python scripts/data_preprocess/smplx2interhuman.py data/motionhub/beat_v2.0.0/test.json
``` 
to get the interhuman motion vectors.