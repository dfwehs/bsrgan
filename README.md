# BSRGAN

Dataset is available [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/)

### Dependencies

Python 3.6, PyTroch 1.3

### DATA Folder

The structure of data folder:

```
data--
	--APY
	--AWA2
	--CUB
	--SUN
```

### Run our code

To run the experiments from the paper, navigate to the model folder and execute the following:

```
python ./bsrgan.py --c ./config/sun_zsl.yaml
```

