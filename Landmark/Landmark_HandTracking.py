import os
import cv2 
import sys
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import numpy as np
import mediapipe as mp
import json
import torchvision.transforms as transforms
from FFNN import Net
from ASL import asl_dict
from torch.utils.data import DataLoader
from torch.backends import cudnn
from LandmarkDataset import LandmarkDataset 
from createLandmarkCSV import csvBuilder

# File pathings
modelPATH = ""
csvPATH = ""
trainingPATH = 'Hand_Models/customASL'

# Neural Network setttings
cudnn.benchmark = True
num_classes = 26
in_channel = 3
learning_rate = 1e-3
batch_size = 32
num_epochs = 1 

# Runtime flags
prev = 0
frameCnt = 0
calculated_frames = 0
confidence_calc = False
letter_accuracy_test = False
test_input = ""
runtime_outputs = []


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
    model.train()

def train_model(device, model):

    model.to(device)

    # settings for training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # model training
    for epoch in range(num_epochs):
        losses=[]
        for batch_idx, (data, targets) in enumerate(train_loader):
            #get data to cuda
            print(targets)
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)
            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent
            optimizer.step()

        print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}')
    return model

#Set the Webcam 
def webcam_320x240(cap):
    cap.set(3,320)
    cap.set(4,240)

def calculate_runtime_accuracy(results):
    total = len(results)
    score = total
    avg_confidence = 0
    ret = ""
    if (not confidence_calc): 
        for val in results:
            if asl_dict[val].lower() != test_input:
                score -= 1
        percent = str(round(((score/total) * 100), 2))
        ret = str(score) + "/" + str(total) + " frames correct: " + percent + "%"
    else:
        for val in results:
            if asl_dict[val[0]].lower() != test_input:
                score -= 1
            avg_confidence += float(val[1])
        avg_confidence = avg_confidence/total
        percent = str(round(((score/total) * 100), 2))
        ret = str(score) + "/" + str(total) + " frames correct: " + percent + "%"
        ret = ret + "\nAverage Confidence: " + str(avg_confidence) + "\n"
    return ret

if __name__ =="__main__":

    if (input("Would you like to output confidence accuracy scores? (Y/N): ").lower() == "y"):
        confidence_calc = True

    if (input("Would you like to test the models accuracy on a single output? (Y/N): ").lower() == "y"):
        letter_accuracy_test = True
        while(len(test_input) != 1 or not test_input.isalpha()): 
            test_input = input("Enter the letter you would like to test the accuracy of: ").lower()

    modelPATH = 'Landmark_CNN.pt'
    csvPATH = 'Landmark_CNN.csv'
    model = torchvision.models.googlenet(pretrained=True)


    # Define CUDA Device
    device = torch.device('cuda')
    print('CUDA Capable device found and set to ' + torch.cuda.get_device_name(torch.cuda.current_device()))

    if (not os.path.isfile(modelPATH)):
        # Generate CSV file
        data_file = csvBuilder(trainingPATH, csvPATH)

        dataset = LandmarkDataset(csv_file = csvPATH, root_dir = 'Hand_Models/', transform = transforms.ToTensor())
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

        train_model(device, model)   
        print('Testing accuracy on newly trained model')
        check_accuracy(test_loader, model)
        
        #save new model
        torch.save(model, modelPATH)
        model.eval()
    else:
        # load existing model
        model = torch.load(modelPATH)
        model.to(device='cuda')
        model.eval()

    cap = cv2.VideoCapture(0)
    webcam_320x240(cap)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    while (cap.isOpened()):

        try:
            # read frame from camera
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # parse hands from frame
            results = hands.process(imgRGB)
            
            frameCnt += 1
            if results.multi_hand_landmarks and frameCnt%3:
                pose = []

                for hand in results.multi_hand_landmarks:
                    calculated_frames += 1
                    saved_frame = np.zeros((320, 240, 3), np.uint8)
                    mpDraw.draw_landmarks(saved_frame, hand, mpHands.HAND_CONNECTIONS)

                    # Optional draw landmarks to screen, leave off for optimal performance
                    mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
                    
                    # convert saved frame to tensor
                    tensorImg = transforms.ToTensor()(saved_frame)
                    tensorImg = tensorImg.to(device='cuda')

                    # get prediction
                    result = model(tensorImg.unsqueeze(0))
                    output_tensor = torch.argmax(result, 1)
                    char_idx = str(output_tensor).split(',')[0].strip('tensor(')

                    if (confidence_calc):
                        #calculate prediction confidence
                        probs = str(max((torch.nn.functional.softmax(result, dim=1))[0]))
                        confidence_score = str(probs.split(',')[0].strip("tensor("))
                        print("Character: " + char_idx + " - Confidence Score: " + confidence_score)
                    else:
                        print("Character:  "+ char_idx)

                    if (letter_accuracy_test):
                        if calculated_frames <= 100:
                            char_idx = char_idx.replace("[", "").replace("]", "")
                            if (confidence_calc):
                                runtime_outputs.append([char_idx, confidence_score])
                            else:
                                runtime_outputs.append(char_idx)
                        else:                       
                            acc = calculate_runtime_accuracy(runtime_outputs)
                            print("\nAccuracy Test Completed")
                            print(acc)
                            sys.exit()

            current = time.time()
            fps = 1 / (current - prev)
            prev = current
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)
            cv2.imshow("Image", img)
            cv2.waitKey(1) 

        except KeyboardInterrupt:

            print("Finished running")          
            break