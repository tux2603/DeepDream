all: downloadModels caffe.sif

.PHONY: downloadModels

downloadModels: 
	[ -e caffe/models/bvlc_googlenet ]  || mkdir -p caffe/models/bvlc_googlenet
	[ -e caffe/models/bvlc_googlenet/deploy.prototxt ] || wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt -O caffe/models/bvlc_googlenet/deploy.prototxt
	[ -e caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel ] || wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel -O caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel
	
caffe.sif: environment.def
	sudo singularity build caffe.sif environment.def
	