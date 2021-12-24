#!/bin/bash
echo $"Starting Net..."
CUDA_VISIBLE_DEVICES=0 python Source/Tools/EvaluationSceneFlow.py
echo $"You can get the running log via the command line that tail -f ./Result/*.log"
echo $"The result will be saved in the result folder"
echo $"If you have any questions, you could contact me. My email: raoxi36@foxmail.com"