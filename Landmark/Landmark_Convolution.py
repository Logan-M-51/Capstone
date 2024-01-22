import os
import cv2 
import sys
import time
#import pandas
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import numpy as np
import mediapipe as mp
import json
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.backends import cudnn
from LandmarkDataset import LandmarkDataset 
from modelClassLandmark import TheModelClass
from createLandmarkCSV import csvBuilder

# File pathings
modelPATH = 'model-FFNN.pt'
csvPATH = 'test.csv'
trainingPATH = 'Hand_Models/custom'

# Neural Network setttings
cudnn.benchmark = True
num_classes = 10
in_channel = 3
learning_rate = 1e-3
batch_size = 32
num_epochs = 1 


# runtime flags
calc_pred = False


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

def train_model(device):

    model = torchvision.models.googlenet(pretrained=True)
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
def Webcam_200p(cap):
    cap.set(3,200)
    cap.set(4,200)

def get_user_input():
    ret = input("Calculate Prediciton Confidence? (Y/N)")
    if (ret.lower() == 'y'):
        calc_pred = True

if __name__ =="__main__":

    # Define CUDA Device
    device = torch.device('cuda')
    print('CUDA Capable device found and set to ' + torch.cuda.get_device_name(torch.cuda.current_device()))

    get_user_input()

    if (not os.path.isfile(modelPATH)):
        # Generate CSV file
        data_file = csvBuilder(trainingPATH)

        dataset = LandmarkDataset(csv_file = csvPATH, root_dir = 'Hand_Models/', transform = transforms.ToTensor())
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

        model = train_model(device)
        model.to('cuda')

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
    Webcam_200p(cap)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    threads = list()
    prev = 0
    frameCnt = 0

    while (cap.isOpened()):
        try:
            # read frame from camera
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # parse hands from frame
            results = hands.process(imgRGB)
            #print(results.multi_hand_landmarks)
            
            frameCnt += 1
            if results.multi_hand_landmarks and frameCnt%3:
                pose = []
                for hand in results.multi_hand_landmarks:

                    saved_frame = np.zeros((200, 200, 3), np.uint8)
                    mpDraw.draw_landmarks(saved_frame, hand, mpHands.HAND_CONNECTIONS)

                    # Optional draw landmarks to screen, leave off for optimal performance
                    #mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
                    
                    # convert saved frame to tensor
                    tensorImg = transforms.ToTensor()(saved_frame)
                    tensorImg = tensorImg.to(device='cuda')

                    # get prediction
                    result = model(tensorImg.unsqueeze(0))
                    output_tensor = torch.argmax(result, 1)
                    char_idx = str(output_tensor).split(',')[0].strip('tensor(')

                    # calculate prediction confidence
                    if (calc_pred):
                        probs = str(max((torch.nn.functional.softmax(result, dim=1))[0]))
                        confidence_score = str(probs.split(',')[0].strip("tensor("))

                        print("Character: " + char_idx + " - Confidence Score: " + confidence_score)
                    else:
                        print("Chracter: " + char_idx)

            current = time.time()
            fps = 1 / (current - prev)
            prev = current
            #print('fps:'+str(fps))
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)
            cv2.imshow("Image", img)
            cv2.waitKey(1)     
        except KeyboardInterrupt:
            break