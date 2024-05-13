pretrained_params='wit400m laion400m laion2b surgery'
for pretrained_param in $pretrained_params
do
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port 12352 \
downstream_phase/run_phase_finetuning_dualpath_timm.py \
--batch_size 16 \
--epochs 30 \
--save_ckpt_freq 5 \
--model vit_large_224_dual_path \
--pretrained_data $pretrained_param \
--patch_size 16 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--opt adamw \
--lr 3e-4 \
--layer_decay 0.1 \
--weight_decay 5e-2 \
--warmup_epochs 3 \
--data_path /data/caizy/data/cholec80 \
--eval_data_path /data/caizy/data/cholec80 \
--nb_classes 7 \
--data_strategy online \
--output_mode key_frame \
--num_frames 31 \
--sampling_rate 4 \
--data_set Cholec80 \
--train_data_fps 1fps \
--test_data_fps 1fps \
--output_dir /data/caizy/PETL4SurgVideo/Cholec80/PETL/DUALPATH/ \
--log_dir /data/caizy/PETL4SurgVideo/Cholec80/PETL/DUALPATH/ \
--num_workers 10 \
--dist_eval \
--no_auto_resume 
done