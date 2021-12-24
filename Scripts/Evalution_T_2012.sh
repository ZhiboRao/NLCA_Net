#!/bin/bash
echo $"Starting Net..."
CUDA_VISIBLE_DEVICES=4 python ./Source/main.py \
                       --gpu 1 --phase test \
                       --modelDir ./PAModel/ \
                       --imgNum 195 \
                       --outputDir ./Result1/ \
                       --resultImgDir ./ResultImg/ \
                       --testListPath ./Dataset/testlist_kitti_2012.txt \
                       --batchSize 1 \
                       --pretrain false
CUDA_VISIBLE_DEVICES=4 python Source/Tools/Evaluation2012.py
echo $"You can get the running log via the command line that tail -f ./Result/*.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"