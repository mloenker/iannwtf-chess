import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def create_graph(epochs, data):
  """
    Creates a graph that visualizes all five evaluation results over training time
      - epochs = list of all models # of epochs, after which they were saved
      - data = data containing the results of each evaluation
    Restriction: can only visualize 5 evaluation criteria
  """
  #transpose data to compare models for each evaluation (instead of evaluations for each model)
  data = list(zip(*data))
  #plot the values
  plt.plot(epochs, data[0], label="from opening position of known human game")
  plt.plot(epochs, data[1], label="from position after move 5 of human game")
  plt.plot(epochs, data[2], label="from position after move 10 of human game")
  plt.plot(epochs, data[3], label="from position after move 5 of artificial game")
  plt.plot(epochs, data[4], label="from position after move 10 of artificial game")
  #prepare plot
  plt.title('Performance of model trained on 10 000 games')
  plt.xlabel('Epochs')
  plt.ylabel('Average amount of legal moves generated*')
  plt.legend(bbox_to_anchor=(1.05, 1))
  plt.annotate('* Mean value was drawn from 5 generated games',
              xy = (1.82, 0.5),
              xycoords='axes fraction',
              ha='right',
              va="center",
              fontsize=10)
  #show plot
  plt.show()
  
def create_table(data1, data2):
  """
    Visualizes data in a table
      - data1 = amount of valid moves of each generated game
      - data2 = illegal move (in SAN) of each generated game
    Requirement: data1 and data2 are of similar size
  """
  #transpose data to compare models for each evaluation (instead of evaluations for each model)
  data1 = list(zip(*data1))
  data2 = list(zip(*data2))
  #create column headings of table
  table_data = [['Evaluation', 'Model', 'Amount valid moves', 'Illegal move']]
  #add data to table
  for e in range(len(data1)):
    for m in range(len(data1[0])):
      table_data.append([e+1,m+1,data1[e][m],data2[e][m]])
    #add empty row between data of different evaluations
    table_data.append(['','',''])
  #create table
  print(tabulate(table_data, headers='firstrow'))
