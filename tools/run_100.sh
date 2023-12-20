
    {
    python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29997 \
    tools/train.py \
    /scratch/yw6594/hpml/mmpretrain/configs_mask/mae_100pre.py \
    --cfg-options optim_wrapper.optimizer.lr=1e-3 work_dir='/scratch/yw6594/out/mask100/sclr1e-3' \
    --launcher pytorch
    }
    # ||
    # {
    # python tools/train.py /scratch/yw6594/hpml/mmpretrain/configs_mask/mae_1kme.py
    # /scratch/yw6594/hpml/mmpretrain/configs_mask/mae_1kprofile.py \
    # /scratch/yw6594/hpml/mmpretrain/configs_mask/maevit_1k.py \
    # optim_wrapper.optimizer.lr=5e-3
    # }



# python tools/train.py \
#     /scratch/yw6594/hpml/mmpretrain/configs_mask/mae_1k.py \

