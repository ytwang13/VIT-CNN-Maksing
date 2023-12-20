
    {
    python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29991 \
    tools/train.py \
    /scratch/yw6594/hpml/mmpretrain/configs_mask/mae_100pre_mix.py \
    --cfg-options work_dir='/scratch/yw6594/out/mask100/mix' \
    --launcher pytorch
    }
    # ||
    # {
    # python tools/train.py /scratch/yw6594/hpml/mmpretrain/configs_mask/mae_1kme.py
    # /scratch/yw6594/hpml/mmpretrain/configs_mask/mae_1kprofile.py \
    # /scratch/yw6594/hpml/mmpretrain/configs_mask/maevit_1k.py \
    # }



# python tools/train.py \
#     /scratch/yw6594/hpml/mmpretrain/configs_mask/mae_1k.py \

