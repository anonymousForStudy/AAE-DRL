# libraries 
import torch
import torch.nn as nn

import torch.nn.functional as F

class Classifier(nn.Module):
    """Classifier class"""
    def __init__(self):
        """
        initialize Classifier model

        simple classifier for one channel images (e.g. mnist dataset)
        """
        super(Classifier, self).__init__()
        self.conv1 = nn.Linear(30, 10)
        self.conv2 = nn.Linear(10, 30)
        self.conv2_drop = nn.Dropout()
        self.fc1 = nn.Linear(30, 50)
        self.fc2 = nn.Linear(50, 4)

    def forward(self, x):
        """
        Function for completing a forward pass of the classifier
        
        Parameters
        ----------
        x : torch.Tensor 
            an image tensor with dimensions (batch_size, 1, im_height, im_width)

        Returns
        -------
        torch.Tensor
            class output
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)




