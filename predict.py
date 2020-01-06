import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from VOCLike import VOCLike
from matplotlib import pyplot as plt
from gluoncv.data.transforms.presets.segmentation import test_transform

from gluoncv.model_zoo.segbase import get_segmentation_model

import gluoncv as gcv
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg

import mxnet as mx

# using cpu
ctx = mx.cpu(0)

filename = 'test.jpg'
img = image.imread(filename)

# plt.imshow(img.asnumpy())
# plt.show()

img = test_transform(img, ctx)
print(VOCLike.classes)

# model = gluoncv.model_zoo.get_model('fcn_resnet50_custom', pretrained=False)
model = get_segmentation_model(model='fcn', dataset='pascal_aug',
                                           backbone='resnet50', norm_layer=mx.gluon.nn.BatchNorm,
                                           norm_kwargs={}, aux=False,
                                           crop_size=480)
model.load_parameters('checkpoint.params')

output = model.predict(img)
# print(output)
print(type(output[0]))
data = mx.nd.argmax(output[0], 1)
predict = mx.nd.squeeze(*data).asnumpy()

mask = get_color_pallete(predict, 'pascal_voc')
mask.save('output.png')

mmask = mpimg.imread('output.png')
plt.imshow(mmask)
plt.show()