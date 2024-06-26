CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port 12320 \
run_phase_finetuning_timm.py \
--batch_size 8 \
--epochs 30 \
--save_ckpt_freq 5 \
--model vit_large_224_timm \
--pretrained_data wit400m \
--patch_size 14 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--opt adamw \
--lr 5e-4 \
--layer_decay 0.75 \
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
--output_dir /data/caizy/SurgPETL/Cholec80/TimeSformer/ \
--log_dir /data/caizy/SurgPETL/Cholec80/TimeSformer/ \
--num_workers 10 \
--dist_eval \
--no_auto_resume