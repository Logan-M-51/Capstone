import csv 
import os

class csvBuilder():

	def get_cols(self):
		cols = []
		for i,j,y in os.walk(self.file_path):
			cols.append(i)
		return cols


	def buildCsv(self):
		data = []
		tensor_val = 0
		for path in self.cols:
			letters = os.listdir(path)
			for letter in letters:
				if (len(path.split('/')[-1]) > 1):
					if(len(letter.split('.')) > 1):
						model_type = path.split('/')[-1]
						data.append([model_type + '/' + letter, tensor_val])
			tensor_val += 1

		# Write CSV file
		with open("test.csv", "w", newline='') as fp:
		    writer = csv.writer(fp, delimiter=",")
		    writer.writerows(data)
		print("csv training file created")


	def __init__(self, file_path):
		self.file_path = file_path
		self.csv = 'myCSV.csv'
		self.cols = self.get_cols()
		self.buildCsv()
