# pyBING++

This is a fork with a python wrapper implemented for BING++ ([BING++: A Fast High Quality Object Proposal Generator at 100fps](http://arxiv.org/abs/1511.04511)). For the original implementation please refer to the [BING++ repository](https://github.com/tolga-b/BINGpp).

Tested on Ubuntu 14.04 with CUDA 7.5.


## Build
To build:
```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DOPENCV_PATH=/path/to/opencv/ (e.g. /home/user/opencv3.0.0/)
make
```


## Example

```
import bingpp
bing = bingpp.BINGpp("/home/minh/src/BINGpp/datasets/VOC2007/Results")
import cv2
img = cv2.imread("/home/minh/src/web/facedu/img_381.jpg", cv2.IMREAD_COLOR)
print bing.getObjBndBoxes(img)
```
