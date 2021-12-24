>This is the project of the StereoMatching Project. This project based on my framework (if you want to use it to build the Network, you can find it in my website: [fadeshine](http://www.fadeshine.com/). If you have any questions, you can send an e-mail to me. My e-mail: raoxi36@foxmail.com)

### Software Environment
1. OS Environment  
    os == linux 16.04  
    cudaToolKit == 9.0  
    cudnn == 7.3.0  
2. Python Environment  
    python == 2.7.15  
    tensorflow == 1.9.0  
    numpy == 1.14.5  
    opencv == 3.4.0  
    PIL == 5.1.0  

### Model
We have upload our model in baidu disk:
https://pan.baidu.com/s/11FNUv8M5L4aO_Are9UjRUA
password: qrho

### Hardware Environment
- GPU: 1080TI * 4 or other memory at least 11G.(Batch size: 2)  
if you not have four gpus, you could change the para of model. The Minimum hardware requirement:  
- GPU: memory at least 5G. (Batch size: 1)

### Train the model by running:
1. Get the Training list or Testing list
```
$ ./Scripts/GenPath.sh
```
Please check the path. The source code in Source/Tools.

2. Run the pre-training.sh
```
$ ./Scripts/Pre-Train.sh
```

3. Run the trainstart.sh
```
$ ./Scripts/TrainKitti2012.sh # for kitti2012
$ ./Scripts/TrainKitti2015.sh # for kitti2015
```

4. Run the teststart.sh
```
$ ./Scripts/TestKitt2012.sh # for 2012
$ ./Scripts/TestKitt2015.sh # for 2015
```

if you want to change the para of the model, you can change the *.sh file. Such as:
```
$ vi ./Scripts/TestStart.sh
```

### File Struct
```
.                          
├── Source # source code                 
│   ├── Basic       
│   ├── Evaluation       
│   └── ...                
├── Dataset # Get it by ./GenPath.sh, you need build folder                   
│   ├── label_scene_flow.txt   
│   ├── trainlist_scene_flow.txt   
│   └── ...                
├── Result # The data of Project. Auto Bulid                   
│   ├── output.log   
│   ├── train_acc.csv   
│   └── ...       
├── ResultImg # The image of Result. Auto Bulid                   
│   ├── 000001_10.png   
│   ├── 000002_10.png   
│   └── ...       
├── PAModel # The saved model. Auto Bulid                   
│   ├── checkpoint   
│   └── ...   
├── log # The graph of model. Auto Bulid                   
│   ├── events.out.tfevents.1541751559.ubuntu      
│   └── ...       
├── Scripts
│   ├── GetPath.sh
│   └── ...       
├── LICENSE
├── requirements.txt
└── README.md               
```

### Update log
#### 2019-10-23 (v1)
1. Finsih refactoring job;
2. Add some files and change the Source/JackBasicStructLib

#### 2019-10-19 (New fork)
1. New project from nlca-net and jacklib projects;
2. Tested the project and make it work;
3. Add some files
4. The target of this project is to build the quantization network for stereo matching tasks.

___

#### 2019-06-17
1. CHanged the file path;
2. Finish review the code of jacklib

#### 2019-01-05
1. Fixed some bugs in random crop process;
2. Update the ReadMe

#### 2018-12-15
1. Add the requirements.txt and LICENSE;
2. Update the 3D module
3. In the feature, We will update refine network.

#### 2018-12-08
1. Change the ReadMe.md;
2. Update the loghangdler.py;
3. Add the building network process in the log file;
4. Fixed some bugs in log file.

#### 2018-12-07
1. Fixed the long time in builduing network during the testing;
2. Add the LICENSE
3. Add the requirenments.txt

#### 2018-11-11
1. Modify the README file.

#### 2018-11-11
1. Write the README file;
2. Fixed some Bugs;
3. Change tensorflow to 1.9.0.

#### 2018-11-08
1. Add Test.py file;
2. Add Switch.py file;
3. Fixed some bugs.

#### 2018-11-05
1. Add the GenPath.sh file;
2. Add Path tool to get the training or Testing list on scence flow or KITTI
3. Add attention moudle;
4. Add GN module;

#### 2018-11-01
1. Finish the StereMatchingNext;
2. Add some file. e.g. Pre-Train.sh

#### 2018-10-30
1. Change the input file;
2. Build the Net Work

#### 2018-10-15
1. Add Multi-GPU, Test the program by Sensitivity Project;

#### 2018-08-25
1. Build the new project;
2. Add some basic network struct;
3. Add the __init__.py
4. Change the file folder.
