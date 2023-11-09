# python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path my-config/blip2/scienceqa/ft_w_lora.yaml
python -m torch.distributed.run \
    --nproc_per_node=1 evaluate.py \
    --cfg-path my-config/blip2/okvqa/eval.yaml \
