import os
import cv2 
import sys
import time
import json
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mediapipe as mp
import torchvision.transforms as transforms
from FFNN import Net
from torch.utils.data import DataLoader
from torch.backends import cudnn
from LandmarkDataset import LandmarkDataset 
from modelClassLandmark import TheModelClass
from createLandmarkCSV import csvBuilder


# File pathings
modelPATH = 'Traditional_FFNN.pt'
csvPATH = 'Traditional_FFNN.csv'
trainingPATH = 'Hand_Models/customASL'

# Neural Network setttings
cudnn.benchmark = True
num_classes = 25
in_channel = 3
learning_rate = 1e-3
batch_size = 32
num_epochs = 1 


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

def train_model(device, train_loader):

    #instantiate Convolutional NN
    model = Net()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    return model

#Set the Webcam 
def Webcam_200p(cap):
    cap.set(3,200)
    cap.set(4,200)

if __name__ =="__main__":

    # Define CUDA Device
    device = torch.device('cuda')
    print('CUDA Capable device found and set to ' + torch.cuda.get_device_name(torch.cuda.current_device()))

    if (not os.path.isfile(modelPATH)):
        # Generate CSV file
        data_file = csvBuilder(trainingPATH)

        dataset = LandmarkDataset(csv_file = csvPATH, root_dir = 'Hand_Models/', transform = transforms.ToTensor())
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        print('Training Convolutional Neural Network Model')
        model = train_model(device, train_loader)

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
                    mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
                    
                    # convert saved frame to tensor
                    tensorImg = transforms.ToTensor()(saved_frame)
                    tensorImg = tensorImg.to(device='cuda')

                    # get prediction
                    result = model(tensorImg.unsqueeze(0))
                    output_tensor = torch.argmax(result, 1)
                    char_idx = str(output_tensor).split(',')[0].strip('tensor(')

                    # calculate prediction confidence
                    # probs = str(max((torch.nn.functional.softmax(result, dim=1))[0]))
                    # confidence_score = str(probs.split(',')[0].strip("tensor("))
                    # print("Character: " + char_idx + " - Confidence Score: " + confidence_score)

                    print("Character: " + char_idx)

            current = time.time()
            fps = 1 / (current - prev)
            prev = current
            #print('fps:'+str(fps))
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)
            cv2.imshow("Image", img)
            cv2.waitKey(1)     
        except KeyboardInterrupt:
            print("Finished running")
            break



