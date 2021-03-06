#!/usr/bin/env python

import argparse
from os import mkdir
from os.path import exists
from shutil import which
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Arguments for reading and writing the images
    parser.add_argument('-s', '--source', help='Directory to read source images from', required=True)
    parser.add_argument('-d', '--destination', help='Directory to write processed images to', required=True)

    # General arguments to control the generation of the dreams
    parser.add_argument('-g', '--guide', help='Guide image for stylistic suggestions', default=None)
    parser.add_argument('-n', '--num-iterations', help='The number of times to run the dream algorithm', default=1, type=int)

    parser.add_argument('-m', '--model', help='The caffe model to be used for the generation of dreams', default='bvlc_googlenet/bvlc_googlenet.caffemodel')
    parser.add_argument('-p', '--prototext', help='The prototext for the specified caffe model', default='bvlc_googlenet/deploy.prototxt')

    # Special purpose arguments to control the generation of the dreams
    parser.add_argument('--blend', '-b', help='The amount to blend the previous frame into the current one. Helps stabilize videos', action='store_true', default=False)
    parser.add_argument('--jitter', '-j', help='The amount of jitter to add to each step during image processing', default=32, type=int)
    parser.add_argument('--maximize', '-l', help='The layer in the neural net to maximize', default='inception_4c/output')
    parser.add_argument('--no-gpu', help='Disables any attempt to run the container on GPU and forces the neural net to use CPU instead', action='store_true', default=False)
    parser.add_argument('--octave-count', help='The number of octaves to run per iteration', default=4, type=int)
    parser.add_argument('--octave-scale', help='The amount to scale each subsequent octave by', default=1.3, type=float)
    parser.add_argument('--steps-per-octave', help='The number of algorithm steps to run per octave', default=10, type=int)
    
    # Control of what GPUs are to be used
    parser.add_argument('--use-gpu', help='Specify the id of a CUDA device to be used by the network. This flag can be given multiple times to enable multi-gpu', action='append', type=int)


    args = parser.parse_args()

    # If no GPU was given, use the default one
    if args.use_gpu is None:
        args.use_gpu = ['0']


    # The model will be run in singularity, so check to make sure it's present
    if which('singularity') is None:
        print('This project requires that singularity be installed. See the installation instructions ')
        exit(1)

    # If the container image isn't present, request it to be built
    if not exists('caffe.sif'):
        print('Virtual environment image is not present. Please build it by running `sudo make\'')
        exit(1)

    # Check to make sure that all the specified files and directories are present
    if not exists(args.source):
        print(f'Source image directory {args.source} could not be found')
        exit(1)

    if not exists(args.destination):
        try:
            mkdir(args.destination)
        except PermissionError:
            print(f'Destination directory {args.destination} does not exist and it could not be created')
            exit(1)

    if args.guide is not None and not exists(args.guide):
        print(f'Guide image {args.guide} could not be found')
        exit(1)

    if args.guide is not None and args.guide[-4:] != '.jpg' and args.guide[-5:] != '.jpeg':
        print(f'Guide image must be a jpeg')
        exit(1)

    commandArgs = ['singularity', 'exec', '--writable-tmpfs']

    # Add nvidia GPU support and set the working directory
    commandArgs.append('--pwd=/opt/src')

    if not args.no_gpu:
        commandArgs.append('--nv')

    # Bind all of the necessary directories
    commandArgs.extend(['-B', f'{args.model}:/opt/model/model.caffemodel'])
    commandArgs.extend(['-B', f'{args.prototext}:/opt/model/deploy.prototxt'])
    commandArgs.extend(['-B', 'src:/opt/src'])
    commandArgs.extend(['-B', f'{args.source}:/opt/images/source'])
    commandArgs.extend(['-B', f'{args.destination}:/opt/images/destination'])
    
    if args.guide is not None:
        commandArgs.extend(['-B', f'{args.guide}:/opt/images/guideImage.jpg'])
    
    # Add the container and command to be run
    commandArgs.extend(['caffe.sif', 'conda', 'run', '--no-capture-output', '-n', 'cpu' if args.no_gpu else 'gpu', './dream.py'])

    # Pass all the necessary arguments down the line
    commandArgs.extend([
        '--num-iterations', args.num_iterations,
        '--octave-count', args.octave_count,
        '--octave-scale', args.octave_scale,
        '--steps-per-octave', args.steps_per_octave,
        '--maximize', args.maximize,
        '--jitter', args.jitter
        ])

    if args.no_gpu:
        commandArgs.append('--no-gpu')

    else:
        for gpuID in args.use_gpu:
            commandArgs.extend(['--use-gpu', gpuID])
        
    if args.blend:
        commandArgs.append('--blend')  

    if args.guide is not None:
        commandArgs.append('--use-guide')

    
    commandArgs = [str(i) for i in commandArgs]

    print(' '.join(commandArgs))
    subprocess.run(commandArgs)

