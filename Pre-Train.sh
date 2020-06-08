#!/bin/bash
echo $"Starting Net..."
CUDA_VISIBLE_DEVICES=2,3 nohup python -u ./Source/main.py \
                       --gpu 2 --phase train \
                       --dataset FlyingThing \
                       --modelName CCANet \
                       --modelDir ./PAModel/ \
                       --auto_save_num 1 \
                       --imgNum 35454 \
                       --valImgNum 0 \
                       --maxEpochs 20 \
                       --learningRate 0.001 \
                       --outputDir ./Result/ \
                       --trainListPath ./Dataset/trainlist_scene_flow.txt \
                       --trainLabelListPath ./Dataset/label_scene_flow.txt \
                       --valListPath ./Dataset/val_trainlist_kitti_2015.txt \
                       --valLabelListPath ./Dataset/val_labellist_kitti_2015.txt \
                       --batchSize 2 \
                       --pretrain false > PreTrainRun.log 2>&1 &
echo $"You can get the running log via the command line that tail -f PreTrainRun.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"