import csv
from datetime import datetime

with open('samples/1616116101_1-energy_aplus_inc.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	f = open('samples/1616116101_1-energy_aplus_inc_new.csv','a')
	for Timestamp, Value in readCSV:
		print(Timestamp)
		if Timestamp == "ds":
			f = open('samples/1616116101_1-energy_aplus_inc_new.csv','w')
			f.write('"ds","y"\n') #Give your csv text here.
			## Python will convert \n to os.linesep

		else:
			new_row_0 = datetime.fromtimestamp(int(Timestamp)/1000)

			f.write('"' + str(new_row_0) + '",' + Value + "\n") #Give your csv text here.
			## Python will convert \n to os.linesep
	f.close()