"""
Example file to show how different learning rates can be set in ONN.

First layer should all have the same learning rate. 
All layers after should have lower learning rates for higher q components.
"""

from Self_ONN import *

class Two_Layer_Model(nn.Module):
    """
    Example two layer ONN model.

    First layer should have all the same learning rate 

    Second layer (and any layers afer) components should have a learning rate corresponding to 1/q for best training stability.
    """

    def __init__(self, in_channels, out_channels):
        """
        q_order: the MacLaurin series order approximation
        """
        super(Two_Layer_Model, self).__init__()

        self.layer1 = Operator_Layer(in_channels, in_channels, q_order=5)           # Use same LR for first layer
        self.layer2 = Operator_Layer_Split(in_channels, out_channels, q_order=5)    # Use differing LR for second layer (and any following layers)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer2(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ONN = Two_Layer_Model(1, 32).to(device)

base_lr = 0.001     

divs = [1,2,3,4,5]      # Values to divide learning rate by, each dividor should be the power of the q component

optimizer = torch.optim.Adam(
    [
        {"params": ONN.layer1.parameters()},        # Works best if first layer all the same

        {"params": ONN.layer2.operators[0].parameters(), "lr": base_lr/divs[0]},       
        {"params": ONN.layer2.operators[1].parameters(), "lr": base_lr/divs[1]},
        {"params": ONN.layer2.operators[2].parameters(), "lr": base_lr/divs[2]},
        {"params": ONN.layer2.operators[3].parameters(), "lr": base_lr/divs[3]},
        {"params": ONN.layer2.operators[4].parameters(), "lr": base_lr/divs[4]},
    ],  
    lr=base_lr)