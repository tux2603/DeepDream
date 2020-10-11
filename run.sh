#!/bin/bash

singularity exec --nv -B src:/opt/src -B caffe/models:/opt/models -B images:/opt/images --pwd=/opt/src caffe.sif /opt/src/main.py "$@"