import csv
from datetime import datetime
import random
import math
from scipy import signal
import pandas as pd

with open('samples/688AB5004D91_02-energy_aplus_inc.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	f = open('samples/periodic_data.csv','a')
	i = 0
	for Timestamp, Value in readCSV:
		print(Timestamp)
		if Timestamp == "ds":
			f = open('samples/periodic_data.csv','w')
			f.write('"ds","y"\n') #Give your csv text here.
			## Python will convert \n to os.linesep

		else:
			f1 = 100.0 * math.sin(-3.0+ 2.0*(math.pi/48.0)*i)
			f2 = 100.0 * math.sin(-4.0 + 2.0*(math.pi/96.0)*i)
			f3 = 100.0 * math.sin(-11/6 + 2.0*(math.pi/96.0)*i)
			newValue = 150.0 + (f1 + f2 + f3) / 3.0 

			f.write('"' + Timestamp + '",' + str(float(newValue)) + "\n") #Give your csv text here.
			## Python will convert \n to os.linesep
			i += 1
	f.close()

"""
df = pd.read_csv('samples/periodic_data.csv', engine='python', parse_dates=True)
dataset = df['y'] + signal.gaussian(0,)
print(dataset)"""