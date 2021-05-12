all: downloadModels caffe.sif

.PHONY: downloadModels

downloadModels: 
	[ -e bvlc_googlenet ] || mkdir -p bvlc_googlenet
	[ -e bvlc_googlenet/deploy.prototxt ] || wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt -O bvlc_googlenet/deploy.prototxt
	[ -e bvlc_googlenet/bvlc_googlenet.caffemodel ] || wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel -O bvlc_googlenet/bvlc_googlenet.caffemodel
	
caffe.sif: environment.def
	sudo singularity build caffe.sif environment.def
	