import torch
import torch.nn as nn

from Self_ONN import Operator_Layer

#conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

q_ord = 3
in_chans = 2
out_chans = 3
bs=64
ks=3
td=8    # Test dimensions

learn_weights = torch.rand(out_chans, in_chans*q_ord)

op_layer = Operator_Layer(in_chans, out_chans, ks=ks, q_order=q_ord)

test_input = torch.rand(bs,in_chans,td,td)
test_output = torch.zeros(bs, out_chans, td, td)

for oc in range(out_chans):
    for q in range(1, q_ord+1):
        pow = torch.pow(test_input, q)
        s = (q-1)*in_chans
        e = s + in_chans
        w = learn_weights[oc, s:e].reshape(1,-1,1,1)
        mul = torch.mul(pow, w)
        test_output[:, oc, :, :] += torch.sum(mul, 1)

s = int(ks/2)       
test_output = test_output[:,:,s:td-s, s:td-s]

print("test input")
print(test_input)
print("test output")
print(test_output)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = op_layer.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
lossFunction = nn.MSELoss()

accum_iter = 16     # 16 x 4, total bs = 64

img = test_input.to(device)
labels = test_output.to(device)

for epoch in range(100000):
    
    output = model(img)
    #print("output:", output.shape)
    #print("labels:", labels.shape)
    loss = lossFunction(output, labels)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    
    
    if (epoch + 1) % 100 == 0:
        print("Epoch:", epoch+1, "| Loss:", loss.item())
        #print(model.operator.weight)
        #print(model.operator.bias)

print(model.operator.weight)
print(model.operator.bias)
print("learn weights:")
print(learn_weights)
print(torch.sum(learn_weights, 1))