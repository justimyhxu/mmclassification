set -x

for i in {50..100}
do 
 PORT=1111 ./tools/dist_test.sh configs/grid_search50/r50_imagenet_resnet_official_encoder_lincls_lr1-62423040-aug05-res5-gp2-img224-ema-mstest.py  work_dirs/r50_imagenet_resnet_official_encoder_lincls_lr1-62423040-aug05-res5-gp2-img224-ema/epoch_$i.pth-ema.pth  8
done
