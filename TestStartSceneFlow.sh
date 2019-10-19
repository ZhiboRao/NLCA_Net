#!/bin/bash
echo $"Starting Net..."
CUDA_VISIBLE_DEVICES=0 nohup python ./Source/main.py \
                       --gpu 1 --phase test \
                       --modelDir ./PAModel/ \
                       --padedImgWidth 960 \
                       --padedImgHeight 576 \
                       --imgNum 4370 \
                       --outputDir ./TestResult/ \
                       --resultImgDir ./ResultImg/ \
                       --testListPath ./Dataset/testlist_scene_flow.txt \
                       --batchSize 1 \
                       --pretrain false > TestRun.log 2>&1 &
echo $"You can get the running log via the command line that tail -f TestRun.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"