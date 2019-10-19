#!/bin/bash
echo $"Starting Net..."
CUDA_VISIBLE_DEVICES=0,1 nohup python -u  -W ignore ./Source/main.py \
                      --gpu 2 --phase train \
                      --dataset KITTI \
                      --modelDir ./PAModel/ \
                      --auto_save_num 10 \
                      --imgNum 160 \
                      --valImgNum 0 \
                      --maxEpochs 1000 \
                      --learningRate 0.001 \
                      --outputDir ./Result/ \
                      --trainListPath ./Dataset/trainlist_kitti_2015.txt \
                      --trainLabelListPath ./Dataset/labellist_kitti_2015.txt \
                      --corpedImgWidth 512 \
                      --corpedImgHeight 256 \
                      --batchSize 4 \
                      --pretrain false > TrainKitti.log 2>&1 &
echo $"You can get the running log via the command line that tail -f TrainKitti.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"
