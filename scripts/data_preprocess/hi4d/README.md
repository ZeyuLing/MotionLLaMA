Download Hi4D dataset from https://hi4d.ait.ethz.ch/download.php?dt=def50200ace684fc7e5830d6e2abb9e155b5ea5fbc2df29d52846130b9b4e790df5cdcbd87a5d34c1850a1232936d79ca73cc912ffaa44e5111a36e17111cad63da7cba3d4f80f1ccbdeacdc4a071bbd70910713cc1354b9a8283412c2ccc7619fa214c4684f151c6bea88738caa

1. Get the standard smplx npz files with `python scripts/data_preprocess/hi4d/standardize_smplx.py`
2. Get the annotation file with `python scripts/data_preprocess/hi4d/get_anno.py`
3. [Optional]Get the humanml3d motion vectors with `scripts/data_preprocess/smplx2hm3d.py data/motionhub/hi4d/train.json`
4. Get the interhuman motion vectors with `scripts/data_preprocess/smplx2interhuman.py data/motionhub/hi4d/train.json`
