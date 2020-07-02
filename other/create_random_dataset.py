import csv
from datetime import datetime
import random

with open('samples/1616116101_1-energy_aplus_inc.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	f = open('samples/random_data.csv','a')
	for Timestamp, Value in readCSV:
		print(Timestamp)
		if Timestamp == "ds":
			f = open('samples/random_data.csv','w')
			f.write('"ds","y"\n') #Give your csv text here.
			## Python will convert \n to os.linesep

		else:
			newValue = random.randint(0, 330)

			f.write('"' + Timestamp + '",' + str(float(newValue)) + "\n") #Give your csv text here.
			## Python will convert \n to os.linesep
	f.close()