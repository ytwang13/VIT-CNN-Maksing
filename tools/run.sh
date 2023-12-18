echo "4 gpu 33 cpu 40 mem"
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29998 \
    tools/train.py \
    /scratch/yw6594/hpml/mmpretrain/configs_mask/mae_1k.py \
    --launcher pytorch



