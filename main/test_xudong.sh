CUDA_VISIBLE_DEVICES=1 python test_qigan.py \
    --dataset ${1:-"MSD"} \
    --datapath ${2:-"/home/gpuadmin/hds/SAM-Adapter-PyTorch/load/MSD_source"} \
    --snapshot ${3:-"/home/gpuadmin/hds/mirror_rorate/HetNet/output_2024-11-26_01-09-20_xudong_han/models/best_mae.pth"} \
    --save_path ${4:-"./Test_model_primitive"}\
    --net_name "Net_xudong"\

