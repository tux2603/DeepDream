#!/usr/bin/env python
# Code based on the example at https://github.com/google/deepdream/blob/master/dream.ipynb
# Note that you probably don't want to run this program directly. It will be started in a
#   custom singularity container by the main.py program

import argparse
import caffe
from functools import partial
from glob import glob
from google.protobuf import text_format
from numba import jit, prange
import numpy as np
import PIL.Image
import scipy.ndimage as nd


# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def postprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

# A set of target objectives. Still learning how they work, but it has to do
#   with the maximization step
def objectiveL2(dst):
    dst.diff[:] = dst.data 

# TODO This would probably be nice to be jit'ed
def objective_guide(dst):
    x = dst.data[0].copy()
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y)
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)]

def makeStep(net, stepSize=1.5, end='inception_4c/output', 
                jitter=32, clip=True, objective=objectiveL2):
    # input image is stored in Net's 'data' blob
    src = net.blobs['data']
    dst = net.blobs[end]

    # apply jitter shift
    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) 

    # specify the optimization objective
    net.forward(end=end)
    objective(dst)  
    net.backward(start=end)
    g = src.diff[0]

    # apply normalized ascent step to the input image
    src.data[:] += stepSize / np.abs(g).mean() * g

    # unshift image
    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def getProgress(numIters, currentIter, numOctaves, currentOctave, numFrames, currentFrame):
    totalPasses = numIters * numOctaves * numFrames
    progress = currentIter + currentOctave * numIters + currentFrame * numOctaves * numIters
    return 100 * progress / totalPasses



def deepdream(net, baseImage, stepsPerOctave=10, numOctaves=4, octaveScale=1.4, 
              end='inception_4c/output', clip=True, numFrames=1, currentFrame=0, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, baseImage)]
    for i in range(numOctaves-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octaveScale, 1.0 / octaveScale), order=1))
    
    # allocate image for network-produced details
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1])

    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        # resize the network's input image size
        src.reshape(1,3,h,w) 
        src.data[0] = octave_base+detail
        for i in range(stepsPerOctave):
            makeStep(net, end=end, clip=clip, **step_params)
            
            # visualization
            vis = postprocess(net, src.data[0])

            # adjust image contrast if clipping is disabled
            if not clip: 
                vis = vis * (255.0 / np.percentile(vis, 99.98))
            percentComplete = getProgress(stepsPerOctave, i, numOctaves, octave, numFrames, currentFrame)
            print(f'\r  {percentComplete:.1f}%', end='')
            
        # extract details produced on the current octave
        detail = src.data[0] - octave_base

    # returning the resulting image
    return postprocess(net, src.data[0])

@jit(nopython=True, parallel=True, nogil=True)
def blend(img1, img2, blendFactor):
    blendedImage = np.zeros_like(img1, dtype=np.float32)

    for i in prange(len(img1)):
        for j in prange(len(img1[i])):
            for k in prange(3):
                blendedImage[i][j][k] = img1[i][j][k] * (1 - blendFactor) + img2[i][j][k] * blendFactor

    return blendedImage

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--num-iterations', help='The number of times to run the dream algorithm', default=5, type=int)
    parser.add_argument('--octave-count', help='The number of octaves to run per iteration', default=4, type=int)
    parser.add_argument('--octave-scale', help='The amount to scale each subsequent octave by', default=1.3, type=float)
    parser.add_argument('--no-gpu', help='Disables any attempt to run the container on GPU and forces the neural net to use CPU instead', action='store_true', default=False)
    parser.add_argument('--steps-per-octave', help='The number of algorithm steps to run per octave', default=10, type=int)
    parser.add_argument('--maximize', help='The layer in the neural net to maximize', default='inception_4c/output')
    parser.add_argument('--jitter', help='The amount of jitter to add to each step during image processing', default=32, type=int)
    parser.add_argument('--use-guide', help='Whether or not to use a guide image for context', default=False, action='store_true')
    parser.add_argument('--blend', help='The amount to blend the previous frame into the current one', action='store_true', default=False)

    args = parser.parse_args()

    if not args.no_gpu:
        print('Using GPU')
        caffe.set_mode_gpu()

        # TODO: Add a command line argument to specify CUDA device
        caffe.set_device(0) 
    
    else:
        print('Using CPU')
        caffe.set_mode_cpu()

    modelPath = '/opt/models/bvlc_googlenet/' # substitute your path here
    netFileName   = modelPath + 'deploy.prototxt'
    parametersFile = modelPath + 'bvlc_googlenet.caffemodel'

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(netFileName).read(), model)
    model.force_backward = True

    # What is the purpose of writing to a file only to read back in immediately?
    open('/tmp/tmp.prototxt', 'w').write(str(model))
    net = caffe.Classifier('/tmp/tmp.prototxt', parametersFile,
        mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
        channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
    
    # Load in a list of images to be processed
    imgNames = []
    imgNames.extend(['/'.join(i.split('/')[4:]) for i in glob('/opt/images/source/*.jpg')])
    imgNames.extend(['/'.join(i.split('/')[4:]) for i in glob('/opt/images/source/*.jpeg')])

    imgNames.sort()

    if args.use_guide:
        guide = np.float32(PIL.Image.open('/opt/images/guideImage.jpg'))
        h, w = guide.shape[:2]
        src, dst = net.blobs['data'], net.blobs[args.maximize]
        src.reshape(1,3,h,w)
        src.data[0] = preprocess(net, guide)
        net.forward(end=args.maximize)
        guide_features = dst.data[0].copy()

        dreamFunc = partial(deepdream, net, stepsPerOctave=args.steps_per_octave,
            numOctaves=args.octave_count, octaveScale=args.octave_scale, 
            numFrames=args.num_iterations, end=args.maximize, 
            objective=objective_guide)

    else:
        dreamFunc = partial(deepdream, net, stepsPerOctave=args.steps_per_octave,
            numOctaves=args.octave_count, octaveScale=args.octave_scale, 
            numFrames=args.num_iterations, end=args.maximize, 
            objective=objectiveL2)

    lastImage = None

    for imgName in imgNames:
        print(f'\n\033[1mProcessing {imgName}\033[0m')
        img = np.float32(PIL.Image.open(f'/opt/images/source/{imgName}'))

        if lastImage is not None and args.blend:
            print('  Blending previous frame')
            img = blend(img, lastImage, np.random.random() * 0.4 + 0.05)

        for i in range(args.num_iterations):
            img = dreamFunc(img, currentFrame=i)
            
        result = PIL.Image.fromarray(img.astype('uint8'))
        print('\n  rendering final image...')
        result.save(f'/opt/images/destination/{imgName}')
        print('  done')
        lastImage = img
