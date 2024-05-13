pretrained_params='imagenet21k wit400m laion400m laion2b'
for pretrained_param in $pretrained_params
do
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 \
--master_port 12322 \
downstream_phase/run_phase_finetuning_stadapter_timm.py \
--batch_size 32 \
--epochs 30 \
--save_ckpt_freq 5 \
--model vit_base_224_stadapter24x384 \
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
--num_frames 8 \
--sampling_rate 4 \
--data_set Cholec80 \
--train_data_fps 1fps \
--test_data_fps 1fps \
--output_dir /data/caizy/SurgPETL/Cholec80/PETL/STAdapter/ \
--log_dir /data/caizy/SurgPETL/Cholec80/PETL/STAdapter/ \
--num_workers 10 \
--dist_eval \
--no_auto_resume
done