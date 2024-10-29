import torch.nn as nn


class LogisticRegression(nn.Module):
    """Logistic Regression model for classification.

    Args:
        input_size (int): The size of the input.
        num_classes (int): The number of classes for classification.

    Attributes:
        num_classes (int): The number of classes for classification.
        linear (nn.Linear): The linear layer for the logistic regression model.
    """

    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, feature):
        output = self.linear(feature)
        return output
