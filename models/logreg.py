import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self,input_size,num_classes):
        super(LogisticRegression,self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(input_size,num_classes)
    
    def forward(self,feature):
        output = self.linear(feature)
        return output