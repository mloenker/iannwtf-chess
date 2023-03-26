import numpy as np
import matplotlib.pyplot as plt

def create_eval_visual(n_epochs, eval1, eval2, eval3, eval4, eval5):
  """
    Creates a graph that visualizes all five evaluation results over training time
      - n_epochs = amount of epochs used in training the model
      - evalX = data containing the results of evaluation X
  """
  #define x-values 
  epochs = np.arange(n_epochs+1)
  #plot the values
  plt.plot(epochs, eval1, label="from opening position of known human game")
  plt.plot(epochs, eval2, label="from position after move 5 of human game")
  plt.plot(epochs, eval3, label="from position after move 10 of human game")
  plt.plot(epochs, eval4, label="from position after move 5 of artificial game")
  plt.plot(epochs, eval5, label="from position after move 10 of artificial game")
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
