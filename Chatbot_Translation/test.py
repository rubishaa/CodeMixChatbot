from chatbot import *
from fuzzywuzzy import fuzz
import csv

with open("/content/drive/MyDrive/Chatbot_Translation/test/UserTest.csv") as file_name:
    file_read = csv.reader(file_name)

    testQuery = list(file_read)

f = open('/content/drive/MyDrive/Chatbot_Translation/test/UserTestResults.csv', 'w')

writer = csv.writer(f)
header = ["Query","Expected Answer", "Actual Answer","Query Match Score", "Processed Time", "Answer Match Score", "Results"]
writer.writerow(header)


for query in testQuery:
  answer,score,elapsed = get_response(query[0])
  ratio = fuzz.ratio(query[1],answer)
  ansScore = ratio/100
  results = "Fail"
  if (ansScore>0.7):
    results = "Pass" 
  row = [query[0],query[1],answer,score,elapsed,ansScore,results]
  writer.writerow(row)


f.close()