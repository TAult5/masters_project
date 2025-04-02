# DARPA Triage Challenge Cont.
##  Tyler Ault

## Overview
The overall synopsis of this project is a participation of CSUCI Undergrad and Graduate students in the [DARPA Triage Challenge](https://triagechallenge.darpa.mil/). Per the competition page:

>The objective of the Systems Competition is to detect and identify physiological signatures of injury derived from data captured by stand-off sensors onboard autonomous systems. This will enable early prioritization of casualties in primary triage, allowing medical care professionals to quickly focus on the most urgent casualties.

>Teams will develop algorithms that detect those signatures in real-time from stand-off sensor data to provide decision support appropriate for austere and complex pre-hospital settings. Teams will use their stand-off sensors, robotic mobility platforms (e.g., UAVs, UGVs), and algorithms to autonomously process sensor data and provide real-time casualty identification and injury assessment. Of particular interest are signatures of acutely life-threatening conditions that medics are trained and equipped to treat during primary triage, such as hemorrhage and airway injuries.

My portion of the project consisted of a continuation of Daniel Peace's work done during his undergrad capstone. His work can be found [here](https://github.com/Daniel-Peace/senior_project).

The majority of effort during this project was put towards supplementing model training with additional labeled data and fine tuning of the models for increased accuracy and performance. The model initially implemented in the project was the [YoloV8](https://docs.ultralytics.com/models/yolov8/) model, developed by Ultralytics. I elected to train and evaluate the performance of the [YoloV11](https://docs.ultralytics.com/models/yolo11/) model, and ultimately determined that the v11 model outperformed the v8 model in identifying most classes. 
