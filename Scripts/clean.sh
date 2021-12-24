#!/bin/bash
echo $"Start to clean the project"
rm -r Result
rm -r ResultImg
rm -r log
rm *.log
find -iname "*.pyc" -exec rm -f {} \;
echo $"Finish"
