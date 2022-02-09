import torch
import torch.nn as nn

from Self_ONN import Operator_Layer
from models import Three_Layer_ONN
from torch.utils.data import DataLoader, Dataset
#from SRDataset import SRDataset
import scipy.io
from patchify import patchify
import numpy as np
from funs import bicubic_lr, hsi_normalize_full, tnsrTo2Dstack, stackToTnsr, scheduler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from piq import ssim, SSIMLoss, psnr
import math
#import tensorflow as tf

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class SRDataset(Dataset):
    def __init__(self, image_data, labels):
        self.image_data = image_data
        self.labels = labels
    def __len__(self):
        return (len(self.image_data))
    def __getitem__(self, index):
        image = self.image_data[index]
        label = self.labels[index]
        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.float)
        )


#LOADING, PATCHING, AND PREPARING PAVIA DATASET


hsi =  scipy.io.loadmat('D:\HSI_SR_datasets\PaviaU.mat').get('paviaU') 
#hsi = hsi/np.max(hsi)
################################################################################
# get image tiles
hr_tiles = []
patch_size = 64
TILES = patchify(hsi, (patch_size,patch_size,hsi.shape[2]), step=patch_size )
for i in range(TILES.shape[0]):
    for j in range(TILES.shape[1]):          
        hr_tiles.append(np.squeeze(TILES[i,j,:,:,:,:], axis = (0,)))


# Normalize each tile
hr_tiles_nor = []
for i in range(len(hr_tiles)):
    hr_tiles_nor.append(hsi_normalize_full(hr_tiles[i]))
    
    
# Create Low resolution stile
lr_tiles = []
ratio = 2
for i in list(range(len(hr_tiles))):
    print("Processing tile # " , i)
    lr_tiles.append(bicubic_lr(hr_tiles_nor[i], ratio))



###############################################################################
X = np.array(lr_tiles)
Y = np.array(hr_tiles_nor)
X = np.moveaxis(X, -1, 1)
Y = np.moveaxis(Y, -1, 1)
X = np.float32(X)
Y = np.float32(Y)

q_ord = 6     # Number of terms in MacLaurin series approximation
in_chans = 103
out_chans = 103
bs=64
ks=3
td=64    # Test dimensions

learn_weights = torch.rand(out_chans, in_chans*q_ord)

#op_layer = Operator_Layer(in_chans, out_chans, ks=ks, q_order=q_ord)
ONN_model = Three_Layer_ONN(in_chans, out_chans, ks=ks, q_order=q_ord)
#test_input = torch.rand(bs,in_chans,td,td)
#test_output = torch.zeros(bs, out_chans, td, td)



(x_train, x_val, y_train, y_val) = train_test_split(X, Y, test_size=0.25,random_state=0)
print('Training samples: ', x_train.shape[0])
print('Validation samples: ', x_val.shape[0])


x_train_torch = torch.from_numpy(x_train)
y_train_torch = torch.from_numpy(y_train)

x_val_torch = torch.from_numpy(x_val)
y_val_torch = torch.from_numpy(y_val)

# train and validation data
#train_data = SRDataset(x_train, y_train)
#val_data = SRDataset(x_val, y_val)
# train and validation loaders
#train_loader = DataLoader(train_data, batch_size=bs)
#val_loader = DataLoader(val_data, batch_size=bs)



#for oc in range(out_chans):
#    for q in range(1, q_ord+1):
#        pow = torch.pow(x_train_torch, q)
#        s = (q-1)*in_chans
#        e = s + in_chans
#        w = learn_weights[oc, s:e].reshape(1,-1,1,1)
#        mul = torch.mul(pow, w)
#        y_train_torch[:, oc, :, :] += torch.sum(mul, 1)

s = int(ks/2)       

#test_output = test_output[:,:,s:td-s, s:td-s]

#print("test input")
#print(test_input)
#print("test output")
#print(test_output)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ONN_model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#lossFunction = nn.MSELoss()
lossFunction = nn.MSELoss()

accum_iter = 16     # 16 x 4, total bs = 64

img = x_train_torch.to(device)
labels = y_train_torch.to(device)

for trial in range(0,5):
    
    for epoch in range(1000):
        
        output = model(img)
        #output[output < 0] = 0
        #print("output:", output.shape)
        #print("labels:", labels.shape)
        loss = lossFunction(output, labels)
        #ssim_metric = ssim(output, labels, data_range=np.finfo('float32').max)
        #ssimloss = SSIMLoss(data_range=np.finfo('float32').max)
        #out: torch.Tensor= loss(output, labels)
        #psnr_index = psnr(output, labels, data_range=np.finfo('float32').max, reduction='none')
        #print(ssim_metric.size())
    
        loss.backward()
    
        optimizer.step()
        optimizer.zero_grad()
        
        
        if (epoch + 1) % 5 == 0:
            print("Epoch:", epoch+1, "| Loss:", loss.item())#, "| SSIM:", ssim_metric.item(), "| PSNR:", psnr_index[0].item())
            #print(model.operator.weight)
            #print(model.operator.bias)
    
    #print("True weights model learned:")
    #print(model.operator.weight)
    #print("bias:", model.operator.bias)
    #print("target learning weights (should be in the centre of the filter, surrounding values should be ~0):")
    #print(learn_weights)
    #print("weight sum for each power term:", torch.sum(learn_weights, 1))
    
    
    # sample = output.cpu()[20,:,:,:]
    # sample = sample.permute(1, 2, 0)
    # sample = sample.detach().numpy()
    
    
    # samplein = labels.cpu()[20,:,:,:]
    # samplein = samplein.permute(1, 2, 0)
    # samplein = samplein.detach().numpy()
    
    # fig, axs = plt.subplots(2)
    # fig.suptitle('PSNR = ' + str(psnr(sample,samplein)) + ', SSIM = '+ str(ssim(sample,samplein)))
    # axs[0].imshow(sample[:,:,2],cmap='gray')
    # axs[1].imshow(samplein[:,:,2],cmap='gray')
    # #plt.imshow(  sample[:,:,2],cmap='gray')
    # #plt.imshow(  samplein[:,:,2],cmap='gray' )
    
    # #y_pred = stackToTnsr(samplein, 103) # Convert Stack into cube, to find SAM
    # cos = torch.nn.CosineSimilarity(dim=0)
    # m = cos(output.cpu()[15,:,:,:],labels.cpu()[15,:,:,:])
    # mn = np.average(m.detach().numpy())
    # sam = math.acos(mn)*180/math.pi
    
    # print(ssim(sample,samplein))
    # print(psnr(sample,samplein))
    
    
    # print(torch.max(labels))
    
    all_psnr = []
    all_ssim = []
    all_sam = []
    
    predicted_output = model(x_val_torch.cuda())
    test_size = predicted_output.size()[0]
    for i in range (0,test_size):
        predict = predicted_output.cpu()[i,:,:,:]
        predict = predict.permute(1, 2, 0)
        predict = predict.detach().numpy()
        grountruth = y_val_torch.cpu()[i,:,:,:]
        grountruth = grountruth.permute(1, 2, 0)
        grountruth = grountruth.detach().numpy()
        cos = torch.nn.CosineSimilarity(dim=0)
        m = cos(predicted_output.cpu()[i,:,:,:],y_val_torch.cpu()[i,:,:,:])
        mn = np.average(m.detach().numpy())
        sam = math.acos(mn)*180/math.pi
        all_psnr.append(psnr(predict,grountruth))
        all_ssim.append(ssim(predict, grountruth))
        all_sam.append(sam)
        fig, axs = plt.subplots(2)
        fig.suptitle('PSNR = ' + str(psnr(predict,grountruth)) + ', SSIM = '+ str(ssim(predict, grountruth)) + ', SAM = ' + str(sam))
        axs[0].imshow(predict[:,:,2],cmap='gray')
        axs[1].imshow(grountruth[:,:,2],cmap='gray')
    
    torch.save(model.state_dict(), 'D:/ONN/Self-ONN-main/PaviaU/MSE/'+str(q_ord)+'/qord'+str(q_ord)+'_'+str(trial)+'_L1'+'.pth')
    print('saved q_ord('+str(q_ord)+')'+'model: '+str(trial))


print(sum(all_psnr)/len(all_psnr))
print(sum(all_ssim)/len(all_ssim))
print(sum(all_sam)/len(all_sam))
