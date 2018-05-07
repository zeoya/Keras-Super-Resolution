# Keras-Super-Resolution
Keras implementation of SrCNN as proposed by Dong et al "Image Super-Resolution Using Deep Convolutional Networks".

## Brief
They key parameter is SCALE in conf/myConfig.py. This is the parameter by which CNN will be trained to super-resolve the input images. The input patch(underresolved) size for training was kept as 33X33, the target patch(natural) size was kept 21X21.
Acc Dong et al using zero-padding would introduce border artifacts. Since zero-padding is avoided therefore target size is less than input size. The srcnn.py in myUtils/ defines the architecture of cnn with 'build' method. The entire architecture mostly consist of CONV => RELU with no zero-padding. The final CONV layers has either 3 or 1 filter/s depending upon RGB or gray-scale image. Before calling the train.py to train the CNN, run build_dataset.py. It will generate inputs.hdf5(underesolved patches) and outputs.hdf5 in the outputs/ folder. Running train.py will load the architecture, the inputs, the targets and will save the trained model and loss-plot in the outputs/. And finally test your model with resize.py. The resize.py requires three arguments:
- **-i**: The path to our input, low resolution image.
- **-b**: The path to save base image. The input image will be upscaled to base image by SCALE factor. This will be compared with CNN's output
- **-o**: The path to save super-resolved image.

## Commands
``` shell
$ python build_dataset.py #this will create inputs.hdf5(underresolved) and outputs.hdf5 in outputs/ folder
$ python train.py #this will train the CNN and save loss-plot and model in outputs/
$ python resize.py -i lena.jpg -b base.png -o output.png #load lena image, upsacle by SCALE factor,then superresolved it.
```
## Environment
Python3, keras2.1 and CV-3 were being used for development on nv-GTX 1080.

## Results
- for the input image lena.jpg, left is base image and right is super-resolved output

![loss-plot]
(./outputs/plot.png)






