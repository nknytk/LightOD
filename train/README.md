Training implementation for object detectors.

### Requirements

* Linux
* Python3
* Nvidia GPU
   - with CUDA and CuDNN

### Setup

```
$ python3 -m venv .venv
$ . .venv/bin/activate
$ pip install --upgrade pip wheel
$ CC="cc -mavx2" pip install -r requirements.txt
```

### Training

```
$ python train.py ../conf/${filename}
```
