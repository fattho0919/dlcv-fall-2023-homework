import torch.nn as nn
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        self.class_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(400, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2)
        )

    def forward(self, input_data, alpha=0, train=True):
        feature = self.feature(input_data)
        class_output = self.class_classifier(feature)
        if train:
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_output = self.domain_classifier(reverse_feature)
            return class_output, domain_output
        else:
            return class_output