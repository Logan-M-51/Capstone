import os
import sys
import cv2
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from createTraditionalCSV import csvBuilder
from customTraditionalDataset import TraditionalDataset

# File pathings
modelPATH = 'model-FFNN.pt'
csvPATH = 'test.csv'
trainingPATH = 'Hand Models/customASL'

# Neural Network setttings
cudnn.benchmark = True
num_classes = 10
in_channel = 3
learning_rate = 1e-3
batch_size = 32
num_epochs = 1 

#fps calculation
prev = 0
frameCnt = 0

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


if __name__ == "__main__":

	# Define CUDA Device
	device = torch.device('cuda')
	print('CUDA Capable device found and set to ' + torch.cuda.get_device_name(torch.cuda.current_device()))

	# if torch model not defined
	if (not os.path.isfile(modelPATH)):
		data_file = csvBuilder(trainingPATH)
		# set up dataLoaders
		dataset = TraditionalDataset(csv_file = csvPATH, root_dir = 'Hand Models/', transform = transforms.ToTensor())
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

	try:
		# Open up CV and then poll frames from camera
		# input to process against model
		capture = cv2.VideoCapture(0)
		Webcam_200p(capture)

	except Exception as e:
		print(str(e) + ': No Camera Source Found, exiting')
		sys.exit(1)

	while (capture.isOpened()):
		try:	
			# display capture on screen	
			success, frame = capture.read()

			frameCnt += 1
			if frameCnt%3:
				# Convert 
				cvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				tensorImg = transforms.ToTensor()(cvImg)
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
			cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)
			cv2.imshow("Image", frame)
			cv2.waitKey(1)
		except KeyboardInterrupt:
			print("Finished running")
			break