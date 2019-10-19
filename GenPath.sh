#!/bin/bash
echo $"Starting Net..."
nohup python ./Source/Tools/TrainList_Kitti_2012.py > ./Result/TrainListGen2012.log 2>&1 &
nohup python ./Source/Tools/TrainList_Kitti_2015.py > ./Result/TrainListGen2015.log 2>&1 &
nohup python ./Source/Tools/TrainList_SceneFlow.py  > ./Result/TrainListGenScence.log 2>&1 &
nohup python ./Source/Tools/TestList_Kitti_2015.py  > ./Result/TestListGen2015.log 2>&1 &
echo $"You can get the running log via the command line that tail -f ./Result/*.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"