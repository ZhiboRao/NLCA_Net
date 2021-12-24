#!/bin/bash
echo $"Starting Net..."
CUDA_VISIBLE_DEVICES=0 python ./Source/main.py \
                       --gpu 1 --phase test \
                       --modelDir ./PAModel/ \
                       --imgNum 200 \
                       --outputDir ./Result1/ \
                       --resultImgDir ./ResultImg/ \
                       --testListPath ./Dataset/testlist_kitti_2015.txt \
                       --batchSize 1 \
                       --pretrain false
CUDA_VISIBLE_DEVICES=0 python Source/Tools/Evaluation.py
echo $"You can get the running log via the command line that tail -f ./Result/*.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"