import torch
import torch.nn as nn

from torchvision.models import alexnet


class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Ready Pytorch documention
    to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one

    Download pretrained alexnet using pytorch's API (Hint: see the import
    statements)
    '''
    super(MyAlexNet, self).__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = nn.CrossEntropyLoss(reduction="sum")

    ###########################################################################
    # Student code begin
    ###########################################################################

    # freezing the layers by setting requires_grad=False
    # example: self.cnn_layers[idx].weight.requires_grad = False

    # take care to turn off gradients for both weight and bias

    model = alexnet(pretrained=True)
    for param in model.parameters():
      param.requires_grad = False


    self.cnn_layers = nn.Sequential(*list(model.children())[:-1])
    for layer in self.cnn_layers[0]:
      if (not isinstance(layer, nn.ReLU) and not isinstance(layer, nn.MaxPool2d)):
        layer.weight.requires_grad = False
        layer.bias.requires_grad = False
  
    self.fc_layers = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 15),
    )
    c = 0
    for layer in self.fc_layers:
      if (not isinstance(layer, nn.ReLU) and not isinstance(layer, nn.Dropout) and c != 5):
        layer.weight.requires_grad = False
        layer.bias.requires_grad = False
      c += 1


    ###########################################################################
    # Student code end
    ###########################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''

    model_output = None
    x = x.repeat(1, 3, 1, 1) # as AlexNet accepts color images

    ###########################################################################
    # Student code begin
    ###########################################################################

    x = self.cnn_layers(x)
    x = torch.flatten(x, 1)
    x = self.fc_layers(x)
    model_output = x

    ###########################################################################
    # Student code end
    ###########################################################################
    return model_output
