import csv
from datetime import datetime
import time

with open('samples/example_wp_log_peyton_manning.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    f = open('samples/example_wp_log_peyton_manning_new.csv','a')
    for Timestamp, Value in readCSV:
        #print(Timestamp)
        if Timestamp == "ds":
            f = open('samples/example_wp_log_peyton_manning_new.csv','w')
            f.write('"ds","y"\n') #Give your csv text here.
            ## Python will convert \n to os.linesep
            
        else:
            new_row_0 = time.mktime(datetime.strptime(Timestamp, '%Y-%m-%d').timetuple())
            print(new_row_0)
            f.write('"' + str(new_row_0) + '",' + Value + "\n") #Give your csv text here.
            ## Python will convert \n to os.linesep
    f.close()