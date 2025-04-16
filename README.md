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

### Additional Packages
I elected to use Albumentations for Yolov11, as Yolov11 integrates automatically with the library, and requires no additional code to implement. Albumentations is a library for easy image augmentation, in addition to what is already implemented by default with the Yolo models. Installed through command line via `pip install albumentation ultralytics`. Documentation for the package may be found [here](https://www.albumentations.ai/docs/).

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
## Images & Labeling
As mentioned above, labelimg was used as the labeling tool for labeling all images. The first step is to load a directory of images to be labeled. Whenever I opened a new directory, I would start with a junk image to label with all my classes, and then later remove it. This was to circumvent a weird quirk with the way labelimg loads classes as you label. You will also need to open a save directory, typically `labels`, where you would like the labels to be stored. labels are stored as a .txt file, with the corresponding class saved as a number, and the coordinates for the bounding box drawn during labeling. Because there are 7 classes, the first number will correspond with each class, 0-6. Labelimg will add additional rows to the text file for each label in the image. 

The classes are as follows: 
- trauma_head
- trauma_torso
- trauma_lower_ext
- amputation_lower_ext
- trauma_upper_ext
- amputation_upper_ext
- severe_hemorrhage

For the purpose of labeling, I tried to maintain a uniform labeling policy. For injuries pertaining to the head, only the head was encapsulated by the bounding box. For the purpose of all other injuries, the entirety of the body was encapsulated. This was to give spacial context to the labels during training, and due to time constraints, I was not able to test other methods of labeling for efficiency. My concern was that the model would not be able to differentiate between upper and lower injuries, especially amputations, without additional context on the location of the injury in relation to the victim.

Once labeling of a directory was complete, the junk image, typically named `junk_image` and its corresponding label were discarded and all other images and labels would be moved to main_images and main_labels. Although rare, there is a chance to encounter naming conflicts when compiling the images and labels in the main folders, so that needs to be resolved during movement. One of the files mentioned below will assist with a standardized naming syntax once all images and labels are in the main folders, so no need to worry about renaming all of the images and labels prior to moving.

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

### mscs_val_test_models
This file is pretty straightforward; this is used to manually conduct validation and testing on the models performance. 

Cell 1 loads the models with their trained weights referenced.

Cell 3 and 4 conduct validation on v8 and v11, respectively.

Cell 6 is used to manually test unlabled images with the model. The unlabeled images need to be loaded as an array. When the `for` loop is ran, it will iterate through the loaded array and attempt to label each image. **THIS WILL TAKE A COUPLE OF MINUTES OR LONGER TO RUN, AS IT LOADS EACH IMAGE AFTERWARDS WITH THE LABEL IT THINKS THE IMAGE SHOULD HAVE.** It is not recommended to run more than a dozen images through this loop at a time, as it will launch each labeled image in your default image viewer with the correct label.

## Results

At the time of writing this documentation, no real-world (integrated) testing was conducted with these updated models. All validation and testing was conducted through functions, as seen in the `mscs_val_test_models.ipynb` file. 

The overall accuracy of the v8 model using current hyperparameters and weights was ~80% averaged between all classes, with a range of 63% for the lowest class, up to 100% for the highest class. 

The overall accuracy of the v11 model using current hyperparameters and weights was ~79.9% averaged between all classes, with a range of 62% for the lowest class, up to 100% for the highest class.

Limited "blind" testing was conducted on models to determine how well they would perform and label completely "unseen" data (images that had no similar likeness to images in the validation sets). At this time, I do not have extensive data on the overall performance of the blind tests, as they were conducted on the limited images I had available to conduct the test, and were conducted in very small batches. As seen below in the first labeled image, the model accurately predicted lower extremity trauma, albeit with a lower confidence score of just 54%. The second labeled image is an example of the model predicting the wrong label with high confidence (torse trauma at 85%), although it did correctly predict severe hemorrhaging with very low confidence (25%). 

Below are the best validation results and F1-Confidence curve for the v8 model. 
![Confusion Maxtrix for Yolov8.](https://github.com/TAult5/masters_project/blob/main/Images/confusion_matrix_v8.png)
![Normalized confusion matrix for Yolov8.](https://github.com/TAult5/masters_project/blob/main/Images/confusion_matrix_normalized_v8.png)
![F1 curve for Yolov8.](https://github.com/TAult5/masters_project/blob/main/Images/F1_curve_v8.png)


Below are the best validation results and F1-Confidence curve for the v11 model.
![Confusion Maxtrix for Yolov11.](https://github.com/TAult5/masters_project/blob/main/Images/confusion_matrix_v11.png)
![Normalized confusion matrix for Yolov11.](https://github.com/TAult5/masters_project/blob/main/Images/confusion_matrix_normalized_v11.png)
![F1 curve for Yolov8.](https://github.com/TAult5/masters_project/blob/main/Images/F1_curve_v11.png)



Blind test conducted on unseen data, prediction labels from model:
![Blind test on v11 model.](https://github.com/TAult5/masters_project/blob/main/Images/Labeled%20blind%20test%201.png)


Example of poor performance on blind test:
![Blind test on v11 model with incorrect predictions.](https://github.com/TAult5/masters_project/blob/main/Images/Labeled%20blind%20test%202%20bad.png)

### Conclusion
While the models arguably perform decently well with validation, they are still struggling with data that is drastically different than the training and validation images. The core cause of this issue can most likely be attributed to the fact that most of the training data is derived from a limited number of runs, meaning the training images have a very similar likeness to the validation images. Additionally, this means many of the class examples are very limited to specific orientations and setups. Although data augmentation was applied internally through hyperparameter tuning and Albumentations, I don't think it's enough to overcome the limited examples of casualties in the dataset. Also, while most of the dataset is very balanced, there are some examples of imbalance, such as upper extremity amputation only have 2 images in the validation set (due to there being limited examples of that class available at all).

### Recommendations
First and foremost, the project needs substantially more data to train with. The model training would greatly benefit from a diverse set of examples for each class, which it currently lacks. The current training set contains 802 images, but there are many images that can be considered duplicates, due to their similarities with other images. I'd estimate a valuable training set for this problem should contain ~2000 images, balanced across classes, to provide meaningful training to the model.

Labeling: I experimented with different labeling techniques before settling on how the images in this project were labeled. As mentioned above, I focused labeling of head trauma to the head, and all other labels were meant to encompass the entire casualty to provide spatial context for the model to differentiate between upper and lower injuries. While I think this approach was good, I'd recommend experimenting with other labeling techniques to check for improvements in the model performance.
