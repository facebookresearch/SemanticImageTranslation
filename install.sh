# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
if [ -d "taming-transformers"]
then
echo "VQGAN repository already downloaded"
else
git clone --depth 1 --branch master https://github.com/CompVis/taming-transformers
fi

CACHE_PATH=~/.cache/torch/vqgan/models/vqgan_imagenet_f16_1024
mkdir -p $CACHE_PATH

if [ -d "$CACHE_PATH/last.ckpt" ] 
then
    echo "Model already downloaded" 
else
    curl https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt --output $CACHE_PATH/last.ckpt
    curl https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml --output $CACHE_PATH/model.yaml
    mkdir generated
fi
