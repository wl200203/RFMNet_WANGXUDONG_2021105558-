CUDA_VISIBLE_DEVICES=2 python train_xudong.py \
    --datapath "/home/gpuadmin/hds/SAM-Adapter-PyTorch/load/MSD_source" \
    --dataset "MSD" \
    --mode "train" \
    --batch_size 12 \
    --lr 0.01 \
    --momen 0.9 \
    --decay 5e-4 \
    --epoch 150 \
    --backbone_path "./resnext/resnext_101_32x4d.pth"\
    --net_name "Net_xudong"\
    --sh_filename "xudong_han_attetnion"


