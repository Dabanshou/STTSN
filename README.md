# STTSN
A MLP-based model for 3D human pose prediction.


### Requirements
------
- PyTorch >= 1.5
- Numpy
- CUDA >= 10.1
- Easydict
- pickle
- einops
- scipy
- six

### Data Preparation
------
Download all the data and put them in the `./data` directory.

[H3.6M](https://drive.google.com/file/d/15OAOUrva1S-C_BV8UgPORcwmWG2ul4Rk/view?usp=share_link)

[Original stanford link](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip) has crashed, this link is a backup.

Directory structure:
```shell script
data
|-- h36m
|   |-- S1
|   |-- S5
|   |-- S6
|   |-- ...
|   |-- S11
```

[AMASS](https://amass.is.tue.mpg.de/)

Directory structure:
```shell script
data
|-- amass
|   |-- ACCAD
|   |-- BioMotionLab_NTroje
|   |-- CMU
|   |-- ...
|   |-- Transitions_mocap
```

[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)

Directory structure: 
```shell script
data
|-- 3dpw
|   |-- sequenceFiles
|   |   |-- test
|   |   |-- train
|   |   |-- validation
```

### Training
------
#### H3.6M
```bash
python main_h36m.py --seed 777 --layer-norm-axis spatial --exp-name baseline.txt --with-normalization --num 36 --num2 12
```


## Evaluation
------
#### H3.6M
```bash
cd exps/baseline_h36m/
python test.py --model-pth your/model/path
```
