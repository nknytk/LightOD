Inference implementation for object detectors.

### Requirements

* Linux
* Python 3.8 or later
* OpenCV
* ONNX RUntime
* PyQt5
* Web camera

### Run

```
$ python detect.py simple_6_pascal_voc3.onnx
```

### Setup Example on Raspberry Pi

Install Python3.8

```
$ sudo apt install -y build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev libffi-dev
$ wget https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tgz
$ tar zxf Python-3.8.2.tgz
$ cd Python-3.8.2
$ ./configure --enable-optimizations --enable-shared --prefix=/opt/python3.8 LDFLAGS=-Wl,-rpath,/opt/python3.8/lib
$ make -j 4
# sudo make altinstall
```

Install packages

```
/opt/python3.8/bin/pip3.8 install numpy PyQt5
```

Install OpenCV

```
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
unznip opencv.zip
unzip opencv_contrib.zip

sudo dphys-swapfile swapoff
sudo dphys-swapfile uninstall
sudo sed -i.bak 's/^#\?\(CONF_SWAPSIZE=\).*/\12048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

cd opencv-4.2.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/opt/opencv4.2 -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-4.2.0/modules -D ENABLE_NEON=ON -D ENABLE_VFPV3=ON -D BUILD_JAVA=OFF  -D BUILD_DOCS=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D OPENCV_ENABLE_NONFREE=ON -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF PYTHON_EXECUTABLE=/opt/python3.8/bin/python3.8 -D PYTHON3_EXECUTABLE=/opt/python3.8/bin/python3.8 -D PYTHON_DEFAULT_EXECUTABLE=/opt/python3.8/bin/python3.8 -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=ON ..
make
sudo make install
```

Install onnx runtime

If your machine has x86_64 CPU, `/opt/python3.8/bin/pip3.8 install onnxruntime` will just make it fine.  
If your machien is Raspberry Pi, get latest build package from [here](https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux/tree/master/wheels).
