frame_rates='1 2 3 4 5 6 7 8'
for frame_rate in $frame_rates
do
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
--nproc_per_node=2 \
--master_port 12324 \
downstream_phase/run_phase_finetuning_aim_timm.py \
--batch_size 32 \
--epochs 30 \
--save_ckpt_freq 5 \
--model vit_base_224_aim_timm \
--pretrained_data wit400m \
--patch_size 16 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--opt adamw \
--lr 3e-4 \
--layer_decay 0.1 \
--weight_decay 5e-2 \
--warmup_epochs 3 \
--data_path /project/mmendoscope/surgical_video/Cholec80 \
--eval_data_path /project/mmendoscope/surgical_video/Cholec80 \
--nb_classes 7 \
--data_strategy online \
--output_mode key_frame \
--num_frames 8 \
--sampling_rate $frame_rate \
--data_set Cholec80 \
--train_data_fps 1fps \
--test_data_fps 1fps \
--output_dir /home/syangcw/SurgPETL/Cholec80/AIM/ \
--log_dir /data/caizy/SurgPETL/Cholec80/AIM/ \
--num_workers 16 \
--dist_eval \
--no_auto_resume
done