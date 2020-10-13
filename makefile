all: caffe.sif

caffe.sif: environment.def
	singularity build caffe.sif environment.def
