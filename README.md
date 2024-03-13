# LTM_ECE_CAPSTONE - A Real Time Hand Tracking Project

## START GUIDE
* To begin with this project training data will be needed. The following will examplify the process with the Landmark implementation but the overall process will be identical with the Traditional. 
  Begin this process by utilizing the LandmarkDatasetCreationTool python scripts included under both the traditional and landmark directories. This tool opens your USB connected camera and turns all captured frames in a given time frame into image files. Once this is completed create a **Hand_Models** directory in the base directory of whichever method you are 
  currently training (Traditional/Landmark). The layout of these directories will be as follows:
  - Landmark
    - Hand_Models
      - A
      - B
      - C
      - D 
  In this new directory create another directory with a name that reflects the respective class that is depicted with these images. Move all recently created images with the LandmarkDatasetCreationTool into this new directory, repeat this dataset creation for every class you would like to include in your training. If you are training the tradiitional method you will 
  need to include a 'nothing' class that includes images of the static background when no hand is in view of the camera. 

* Once your training data is completed open a terminal and run the command **python Landmark_HandTracking.py**
  When this command runs you will be promtped two questions that will ask if you would like confidence scores to be calculated or if you would like to evalute the model's accuracy on a single model. Respond **no** to both of these for now and proceed with the program's execution. The code will see that no models currently exists and will begin training. Once training 
  is completed the model will begin evaluating.

* At this point it will be up to you to fine tune the training data you have created to get optimal results. If you are unable to get consistently accurate models, consider some of these points that can lead to lesser performance:
  - Sufficent lighting is needed in your experimental environment for good results. Specifc to the Traditional method, your training data will need to properly trained to account for the slight differences in lighting that your environment may have throught the day.
  - Prune 'bad' data from your datasets. For the Landmark method, the landmarks sometimes jitter in low light environments making some of the images not properly reflect their class. Another issue that can arise for both methods is camera blur. If the user moves fast as the DatasetCreationTool is capturing images the resulting picture could be a poor representation of the intended hand sign. If any 'bad' images are removed, replace them with proper images. 
  - Make sure that each class has a roughly even amount of images. Training is completed with a random selection of 80% of your data (the remaining 20% used for accuracy testing), thus if certain classes have more iamges it can lead to heavily biased model predicitions.

## EVALUATION
* When running the handtracking python scripts the user is prompted two questions that relate to the calculation of confidence scores and testing the accuracy of the model on a single output.
  - The first question will enable confidence calculations when you respond 'y'. These confidence scores help showcase how well the model is trained however the calculation of these scores is intensive so it is recommended to only include these calculations when needed.
  - If you respond 'y' to the question regarding testing the model's output on a sinle value you will be promted once more asking which class you would like to test against. When the program begins it will capture 100 frames of your camera's input and compare the model's prediciton against the class you instructed it with. Upon completion the program will provide and average prediction score as well as an average confidence calculation if available.
