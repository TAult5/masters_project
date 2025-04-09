# DARPA Triage Challenge Cont.
##  Tyler Ault

## Overview
The overall synopsis of this project is a participation of CSUCI Undergrad and Graduate students in the [DARPA Triage Challenge](https://triagechallenge.darpa.mil/). Per the competition page:

>The objective of the Systems Competition is to detect and identify physiological signatures of injury derived from data captured by stand-off sensors onboard autonomous systems. This will enable early prioritization of casualties in primary triage, allowing medical care professionals to quickly focus on the most urgent casualties.

>Teams will develop algorithms that detect those signatures in real-time from stand-off sensor data to provide decision support appropriate for austere and complex pre-hospital settings. Teams will use their stand-off sensors, robotic mobility platforms (e.g., UAVs, UGVs), and algorithms to autonomously process sensor data and provide real-time casualty identification and injury assessment. Of particular interest are signatures of acutely life-threatening conditions that medics are trained and equipped to treat during primary triage, such as hemorrhage and airway injuries.

My portion of the project consisted of a continuation of Daniel Peace's work done during his undergrad capstone. His work can be found [here](https://github.com/Daniel-Peace/senior_project).

The majority of effort during this project was put towards supplementing model training with additional labeled data and fine tuning of the models for increased accuracy and performance. The model initially implemented in the project was the [YoloV8](https://docs.ultralytics.com/models/yolov8/) model, developed by Ultralytics. I elected to train and evaluate the performance of the [YoloV11](https://docs.ultralytics.com/models/yolo11/) model, and ultimately determined that the v11 model outperformed the v8 model in identifying most classes. 

## Hardware and Software
Training was conducted on an RTX 4080 Laptop GPU, 12GB. CUDA acceleration would work on training the models, but I found that no matter which troubleshooting I attempted, I could not get CUDA to work properly when using the `tune` function. Even explicitely listing `device = 'cuda'` in the paramters to be passed in the function, it would default back to CPU while running the function. This was discovered by running `print(next(model.parameters()).device)` after running a tuner.

Jupyter Notebook was utilized as the primary IDE for ease of running indivdual cells asynchronously. Any external files, such as the data.yaml and best_hyperparameters.yaml files, were edited in VS Code, if necessary.

[labelimg](https://github.com/HumanSignal/labelImg) was utilized for labeling all images. It is no longer being actively developed, so it may not work properly with newer versions of Python. I encountered an issue with datatypes while trying to run it, which resulted in the program crashing anytime there was an attempt to draw a bounding box; the solution requires editing certain Python libraries to ensure the the correct datatypes are being passed; if this issue is encountered, the solution can be found [here](https://github.com/HumanSignal/labelImg/issues/872#issuecomment-1309017766).

Due to DARPA requirements on data privacy, [VeraCrypt](https://www.veracrypt.fr/code/VeraCrypt/) was utilized for all data encryption.

## File Structure
Below is how the file structure was organized for this project. The code will need to be changed as needed to accommadate a different file structure. main_images and main_labels were used to store all images and labels prior to being split into training, validation, and test sets, and left in place as a backup, should issues arise with the dataset files. The v8_training and v11_training folders were used to store the weights, which could be referenced by function in the overall project. Only **ONE** best.pt file should be kept in their respective weight folders. The data.yaml file will be explained in detail below, with explanations of the different project files.

There is no need to manually allocate images and labels to their respective folders, once placed in the main_images and main_labels folders. A function, which will be explained below will handle that.

S:/ Drive
- data_tyler
  - data.yaml
  - best_hyperparameters_v8
  - best_hyperparameters_v11
  - dataset
    - train
      - images
      - labels
    - val
      - images
      - labels
    - test
      - images
      - labels
- main_images
- main_labels
- v8_training
  - weights
    - best.pt
    - last.pt
- v11_training
  - weights
    - best.pt
    - last.pt

## Project Files

As mentioned earlier when discussed that the IDE of choice was Jupyter Notbook, with the exception of the data.yaml file, all other project files (excluding images and labels) are saved as the .ipynb filetype.

### data.yaml
The data.yaml file is used to referance the appropriate filepaths for training, validation, and testing, as well as storing the number of classes, `nc: #`, as necessary, and the class names, stored as an array of `names: ['class1, class2, ...]`.

The `train:`, `val:`, and `test:` pathing may need to be changed to support a different file structure. Referencing the file structure section as an example, the path for my training set would be `train: S:/data_tyler/dataset/train`. You do not need to reference the image or label folders within the train, val, or test folders.

### mscs_Compare_directories_funct.ipynb
This is the _good housekeeping_ file. Initially the project had images stored in multiple locations, so after consolidating all the images and labels into the main_images and main_labels folders, running this file will provide an output of any image files that don't have corresponding labels, and any labels that don't have corresponding images associated with them.

### mscs_rename_files.ipynb
This file is intended to rename all images and their corresponding labels to one uniform syntax. This may not be necessary, but was developed after the labeling process when I was using multiple directories to store the images and labels. After consolidating all images to main_images and all labels to main_labels, and running the compare_directories function, this function was ran to create the uniform syntax of image_000001, image_000002, ..., and likewise for all labels associated with each image. A warning will be triggered if the number of images and labels don't match, with the output of "Warning: The number of image and label files do not match!" being produced if that is the case. In that case, the user can run the compare_directories function again to determine what image or label file may be causing the issue.

### mscs_train_val_test_split_func
This code is used to properly split the data, once organized, into appropriate directories. Again, pathing referenced in this file may need to be changed to accommadate different file structures. 

`image_dir` should reference main_images, `label_dir` should reference main_labels, and `dest_dirs` should reference the corresponding train, val, and test directories. The images are shuffled using the `random` function, and then the ratios are assigned. For the purpose of this project, I used a 70% training / 20% validation split, with the remainder being put into the test folder. 

Running the `for split` loop will copy the image to the new directory, then determine the corresponding lable and copy that to the new directory, as well. Once this function has been ran, it's good practice to go back and manually verify that the function correctly deposited images and labels in proper amounts. Edits can be made to the compare_directories function to also verify this.

### mscs_hyp_tuner
This file is used to determine the best hyperparameters to train the models with. Upon completion of organizing all files and establishing the proper train, val, test split, this file may be used. Cell one loads both the v8 and v11 models as `model` and `model2` respectively. `file_path_data` is used to reference the location of the data.yaml file. I left the cell with `import torch` in there in an effort to force the tuner to use CUDA for the tuning, but could not get it to properly do so. Some sort of callback in the loop of the tuner was forcing it to defaul to the CPU for tuning.

Model tuning is pretty straightforward, but very computationally time consuming. Changing `iterations` will change how many "rounds" the tuner goes through. For example, I set `iterations = 100` when running the tuner for v8. The total time to run 100 iterations on the current dataset took 33,813.27 seconds, or about 9.4 hours in total. The output, saved as a best_hyperparameters.yaml file will give the specifics on each of the best parameters found for that model, including learning rates, momentum, decay, and some data augmentation, and will include the best fitness metrics found during the tuning process, and at what iteration the metrics were found at. 

Additionally, tuning time is effected by the version of the model, as well. Yolov11 was only tuned for 50 iterations, and had a total tuning time of 53,386.27 seconds, or roughly 14.8 hours. 

It is important to note where the default output of your IDE goes to. My default was set to `C:/Users/tyler/runs/detect/tuner#`, which is where the best_hyperparameters.yaml file will be located.

### mscs_train_models
Onto training. Just like with the tuner file, each model needs to be called and assigned; in this case, `model`(v8) and `model2`(v11). You will also need to reference the data.yaml file, if running this in a seperate kernel from the tuner file. 

Cell 3 of this file will manually load the hyperparameters to their respective model. Yolo's `train` functions do not allow you to directly reference hyperparameters as some models do, so this cell is necessary to properly load them. This uses the `model.overrides.update` function to manually override the default hyperparameters.

Cell 4 is used to verify that the hyperparameters were properly loaded to the model. I commented in what the expected values of a couple hyperparameters should be, but when using different hyperparameters than what I have, you can just open the yaml file and verify manually.

Cell 5 was used to train the Yolov8 model. `data` will reference your `file_path_data`, which is your data.yaml file. Epochs will change how long the model will run for, and I used a `patience` parameter of 20, which is Yolo's version of early stopping. This will stop training if there are no improvements to the model after 20 total iterations. This can be set to 0 or 300 to ignore patience, or deleted entirely and not passed as a parameter. 

The same procedures were used for training the Yolov11, with the exception being that a different `best_hyperparameters.yaml` file will need to be referenced, in this case being `best_hyperparameters_v11.yaml`.

Likewise with the tuning file, the `train`function will output weights and results to wherever your IDE defaults to. Outputs include best and last weights, confusion matrices, and different graphs on the results. Because Yolo saves both your best and last weights, you don't necessarily need to pass a patience parameter, if time or computation isn't a concern.




