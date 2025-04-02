# DARPA Triage Challenge Cont.
##  Tyler Ault

## Overview
The overall synopsis of this project is a participation of CSUCI Undergrad and Graduate students in the [DARPA Triage Challenge](https://triagechallenge.darpa.mil/). Per the competition page:

>The objective of the Systems Competition is to detect and identify physiological signatures of injury derived from data captured by stand-off sensors onboard autonomous systems. This will enable early prioritization of casualties in primary triage, allowing medical care professionals to quickly focus on the most urgent casualties.

>Teams will develop algorithms that detect those signatures in real-time from stand-off sensor data to provide decision support appropriate for austere and complex pre-hospital settings. Teams will use their stand-off sensors, robotic mobility platforms (e.g., UAVs, UGVs), and algorithms to autonomously process sensor data and provide real-time casualty identification and injury assessment. Of particular interest are signatures of acutely life-threatening conditions that medics are trained and equipped to treat during primary triage, such as hemorrhage and airway injuries.

My portion of the project consisted of a continuation of Daniel Peace's work done during his undergrad capstone. His work can be found [here](https://github.com/Daniel-Peace/senior_project).

The majority of effort during this project was put towards supplementing model training with additional labeled data and fine tuning of the models for increased accuracy and performance. The model initially implemented in the project was the [YoloV8](https://docs.ultralytics.com/models/yolov8/) model, developed by Ultralytics. I elected to train and evaluate the performance of the [YoloV11](https://docs.ultralytics.com/models/yolo11/) model, and ultimately determined that the v11 model outperformed the v8 model in identifying most classes. 

## Hardware and Software
Training was conducted on an RTX 4080 Laptop GPU, 12GB.

Jupyter Notebook was utilized as the primary IDE for ease of running indivdual cells asynchronously. Any external files, such as the data.yaml and best_hyperparameters.yaml files, were edited in VS Code, if necessary.

[labelimg](https://github.com/HumanSignal/labelImg) was utilized for labeling all images. It is no longer being actively developed, so it may not work properly with newer versions of Python. I encountered an issue with datatypes while trying to run it, which resulted in the program crashing anytime there was an attempt to draw a bounding box; the solution requires editing certain Python libraries to ensure the the correct datatypes are being passed; if this issue is encountered, the solution can be found [here](https://github.com/HumanSignal/labelImg/issues/872#issuecomment-1309017766).

Due to DARPA requirements on data privacy, [VeraCrypt](https://www.veracrypt.fr/code/VeraCrypt/) was utilized for all data encryption.
