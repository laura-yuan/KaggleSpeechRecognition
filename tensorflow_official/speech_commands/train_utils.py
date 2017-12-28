import pickle
import numpy as np
# learning step index by 0
class jy_summary:
    def __init__(self, max_step):
        self.accurarcy = np.zeros(max_step)
        self.entropy = np.zeros(max_step)
        self.confusion_matrix = [None for ii in range(max_step)]
        self.learning_rate = np.zeros(max_step)
        self.step = np.zeros(max_step)
    def update(self, which_step, accuracy_value, entropy_value, confusion_matrix_value,learning_rate_value):
        self.step = which_step
        self.accurarcy[which_step] = accuracy_value
        self.entropy[which_step] = entropy_value
        self.confusion_matrix[which_step] = confusion_matrix_value
        self.learning_rate[which_step] = learning_rate_value
    def save(self, directory):
        with open(directory, 'wb') as f:
            pickle.dump(self, f)
