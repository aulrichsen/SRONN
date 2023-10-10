import json
import glob
import math

import cv2
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt 
import scipy
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

from Load_Data import get_data, hsi_normalize_full
from models import get_model
from training_setup import parse_train_opt
from Test_Model import test_model

DATA_DIR = 'C:/Users/psb15138/Documents/Uni/ONN/dataset'  # For my PC
hsi = scipy.io.loadmat(DATA_DIR + '/PaviaU.mat')#.get('paviaU') 


#hsi = hsi.get(list(hsi.keys())[-1])     # Extract only data (Remove header etc)
  
with open('best_results.json', 'r') as openfile:
    # Reading from json file
    best_results = json.load(openfile)

# Selection parameters
norm_type = 'none'
dataset = 'PaviaU'
is_residual = False
scale = 2
image_type = 'objective'

img_idx_1 = 5
img_idx_2 = 6
slice_idx = 80
dim=128

#img_idx = 0
#slice_idx = 40

def get_img_file(modelname, img_idx=0, slice_idx=40):
    if is_residual: modelname += '_residual'
    if norm_type != 'none': modelname += '_' + norm_type + '_norm'
    try:
        folder = best_results[modelname][dataset + ' x2']['folder']
    except:
        return False
    img_files = glob.glob(folder + '/' + image_type + '/*')
    for img_file in img_files:
        if f'Img {img_idx}, Slice {slice_idx}' in img_file:
            return img_file
    else:
        return False

def get_model_file(modelname):
    if is_residual: modelname += '_residual'
    if norm_type != 'none': modelname += '_' + norm_type + '_norm'
    try:
        folder = best_results[modelname][dataset + ' x2']['folder']
    except:
        return False
    files = glob.glob(folder + '/*')
    for file in files:
        if 'best_SSIM' in file:
            return file
    else:
        return False



'''
# Residual models, no norm
srcnn_img = "Final Results (to be put in papers)/Journal Paper/30_08_2022 23_27_56 SRCNN_residual PaviaU x2/objective/Img 0, Slice 40, PSNR 36.png"
sronn_img = "Final Results (to be put in papers)/Journal Paper/29_08_2022 11_59_47 SRONN_residual PaviaU x2/objective/Img 0, Slice 40, PSNR 36.png"
ssronn_img = "Final Results (to be put in papers)/Journal Paper/28_08_2022 23_22_18 SRONN_AEP_residual PaviaU x2/objective/Img 0, Slice 40, PSNR 36.png"
'''

device = "cuda" if torch.cuda.is_available() else "cpu"

_, _, _, _, lr_tiles, hr_tiles, _ = get_data(dataset, noise_var=0)  # Bug in code meant that despite initialising with noise_var=0.00005, no noise actually applied... (didn't call within get_dataloaders function), so 0 used instead to reproduce.

'''
ip_img = lr_tiles[img_idx].to(device)
tar_img = hr_tiles[img_idx].to(device)
print("ip_img:", ip_img.shape)
tar_slice = hr_tiles[img_idx][slice_idx, :,:].to(device)
print(tar_slice.shape)
'''

# Change directory to oneDrive
for model in best_results.keys():
    for ds in best_results[model].keys():
        if best_results[model][ds]['folder']:
            best_results[model][ds]['folder'] = 'C:/Users/psb15138/OneDrive - University of Strathclyde/PhD' + best_results[model][ds]['folder'].split('..')[-1]

srcnn_weights = get_model_file('SRCNN')
sronn_weights = get_model_file('SRONN')
ssronn_weights = get_model_file('SRONN_AEP')
channels = hr_tiles[0].shape[0]

print("Weights files")
print(srcnn_weights)
print(sronn_weights)
print(ssronn_weights)

def load_model(opt, weights):
    model, _ = get_model(opt, channels)
    model.load_state_dict(torch.load(weights))
    model.eval()
    model.to(device)
    return model

class Opt():
    def __init__(self):
        self.dataset = dataset
        self.model = 'SRCNN'
        self.is_residual = is_residual
        self.q = 1
        self.norm_type = norm_type

        self.init_type = 'normal'
        self.init_gain = 0.02
        self.weight_transfer = False
        self.checkpoint = False
        self.SISR = False
        self.scale = 2

opt = Opt()
opt.is_residual = is_residual

srcnn = load_model(opt, srcnn_weights)
opt.model = 'SRONN'
opt.q = 3
sronn = load_model(opt, sronn_weights)
opt.model = 'SRONN_AEP'
ssronn = load_model(opt, ssronn_weights)

#test_data = TensorDataset(lr_tiles.to(device), hr_tiles.to(device))
#test_dl = DataLoader(test_data, batch_size=64, shuffle=False)
#test_model(sronn, test_dl, opt, save_dir='test', disp_slices=[{'b': img_idx, 'c': slice_idx}])

#plt.imshow(tar_img_np)
#plt.show()

DATA_DIR = 'C:/Users/psb15138/Documents/Uni/ONN/dataset'  # For my PC
hsi = scipy.io.loadmat(DATA_DIR + '/PaviaU.mat')#.get('paviaU') 
hsi = hsi.get(list(hsi.keys())[-1])     # Extract only data (Remove header etc)

hsi = hsi_normalize_full(hsi)

#plt.imshow(hsi[:,:,slice_idx])
#plt.show()

def make_img_display(model, slice_idx, ip_img, tar_img):
    #print(model.name)
    with torch.no_grad():
        pred = torch.clamp(model(ip_img.unsqueeze(0)), min=0, max=1)
    
    ip_slice = ip_img[slice_idx, :, :]
    tar_slice = tar_img[slice_idx, :, :]
    #print("ip:", ip_slice.min(), ip_slice.max(), "srcnn:", pred.min(), pred.max(), "tar_slice:", tar_slice.min(), tar_slice.max())

    pred_slice = pred[0, slice_idx, :,:]

    spatial_absolute_difference = torch.abs(pred_slice - tar_slice)
    #print("spatial_absolute_difference sum:", spatial_absolute_difference.sum())

    #print(psnr(tar_slice.cpu().numpy(), pred_slice.cpu().numpy(), data_range=1))

    var_mean = torch.var_mean(spatial_absolute_difference, dim=(0,1), unbiased=False)

    output = torch.cat([ip_slice, pred_slice, tar_slice, spatial_absolute_difference*3], dim=1)
    #plt.imshow(output.cpu().numpy())
    #plt.title(model.name)
    #plt.show()
    return output, float(spatial_absolute_difference.sum()), var_mean


''' Find best tiles and slices for figures
for img_idx in range(lr_tiles.shape[0]):
    ip_img = lr_tiles[img_idx].to(device)
    tar_img = hr_tiles[img_idx].to(device)

    best_vm_diff = 0

    for slice_idx in range(3, ip_img.shape[0]-3):       # Ignore end slices
        srcnn_disp, srcnn_sad, srcnn_vm = make_img_display(srcnn, slice_idx, ip_img, tar_img)
        sronn_disp, sronn_sad, sronn_vm = make_img_display(sronn, slice_idx, ip_img, tar_img)
        ssronn_disp, ssronn_sad, ssronn_vm = make_img_display(ssronn, slice_idx, ip_img, tar_img)

        #if max(srcnn_sad-sronn_sad, srcnn_sad-ssronn_sad) > best_diff:
        #    best_diff = max(srcnn_sad-sronn_sad, srcnn_sad-ssronn_sad)
        #    best_img, best_slice = img_idx, slice_idx
        #    all_disp_sad = torch.cat([srcnn_disp, sronn_disp, ssronn_disp], dim=0)
        #    print(f"SAD | img: {img_idx} | slice {slice_idx} | SRCNN: {round(srcnn_sad, 2)} | SRONN: {round(sronn_sad, 2)} | sSRONN: {round(ssronn_sad, 2)}")
        

        if max(srcnn_vm[0]-sronn_vm[0], srcnn_vm[0]-ssronn_vm[0]) > best_vm_diff:
            best_vm_diff = max(srcnn_vm[0]-sronn_vm[0], srcnn_vm[0]-ssronn_vm[0])
            best_vm_img, best_vm_slice = img_idx, slice_idx
            all_disp_vm = torch.cat([srcnn_disp, sronn_disp, ssronn_disp], dim=0)
            diffs = srcnn_vm[0]-sronn_vm[0], srcnn_vm[0]-ssronn_vm[0]
       
        #if best_vm_diff == 0:
        #    print(best_vm_diff, slice_idx)
        #else:
        #    print(best_vm_diff, slice_idx, best_vm_slice)
            
    print(f"VM | img: {img_idx} | slice {best_vm_slice} | diffs {diffs}")
    plt.imshow(all_disp_vm.cpu().numpy())
    plt.title(f"VM | Img {best_vm_img}, Slice {best_vm_slice} SRCNN top, SRONN middle, sSRONN bottom")
    plt.show()
'''

# Disp images: img 4 slice 80, img 6 slice 26

'''
plt.imshow(all_disp_sad.cpu().numpy())
plt.title(f"SAD | Img {best_img}, Slice {best_slice} SRCNN top, SRONN middle, sSRONN bottom")
plt.show()
'''

def add_predictions_and_spatial_difference_to_figure(out_img, img_idx, slice_idx, hsi_patch_loc, global_offset=5, dim=128):
    def get_pred_slice(model, ip_img, slice_idx):
        with torch.no_grad():
            pred = torch.clamp(model(ip_img.unsqueeze(0)), min=0, max=1)
            
            pred_slice = pred[0, slice_idx, :,:]
        return pred_slice.cpu().numpy()

    tar_slice_np = hr_tiles[img_idx][slice_idx].numpy()
    
    ip_img = lr_tiles[img_idx].to(device)
    tar_img = hr_tiles[img_idx].to(device)
    srcnn_pred = get_pred_slice(srcnn, ip_img, slice_idx)
    sronn_pred = get_pred_slice(sronn, ip_img, slice_idx)
    ssronn_pred = get_pred_slice(ssronn, ip_img, slice_idx)

    ip_slice = lr_tiles[img_idx][slice_idx].numpy()
    ip_slice = cv2.resize(ip_slice, (dim, dim),interpolation=cv2.INTER_NEAREST)
    srcnn_pred = cv2.resize(srcnn_pred, (dim, dim),interpolation=cv2.INTER_NEAREST)
    sronn_pred = cv2.resize(sronn_pred, (dim, dim),interpolation=cv2.INTER_NEAREST)
    ssronn_pred = cv2.resize(ssronn_pred, (dim, dim),interpolation=cv2.INTER_NEAREST)
    tar_slice_np = cv2.resize(tar_slice_np, (dim, dim),interpolation=cv2.INTER_NEAREST)

    out_img[:, :hsi_patch_loc.shape[1], :] = hsi_patch_loc
    hsi_width = hsi_patch_loc.shape[1]
    for offset, img in zip(range(5, dim*5+10, dim+10), [ip_slice, srcnn_pred, sronn_pred, ssronn_pred, tar_slice_np]):
        for i in range(3):
            out_img[global_offset:dim+global_offset, hsi_width+offset:hsi_width+offset+dim, i] = img    # All cahnnels same for grayscale

    def get_spatial_absolute_difference(model, slice_idx, ip_img, tar_img):
        with torch.no_grad():
            pred = torch.clamp(model(ip_img.unsqueeze(0)), min=0, max=1)
        
        ip_slice = ip_img[slice_idx, :, :]
        tar_slice = tar_img[slice_idx, :, :]
        #print("ip:", ip_slice.min(), ip_slice.max(), "srcnn:", pred.min(), pred.max(), "tar_slice:", tar_slice.min(), tar_slice.max())

        pred_slice = pred[0, slice_idx, :,:]

        spatial_absolute_difference = torch.abs(pred_slice - tar_slice)
        return spatial_absolute_difference

    # Scale spatial difference map so that colour map between 0 and 0.3
    srcnn_disp = get_spatial_absolute_difference(srcnn, slice_idx, ip_img, tar_img) * 3.33333
    sronn_disp = get_spatial_absolute_difference(sronn, slice_idx, ip_img, tar_img) * 3.33333
    ssronn_disp = get_spatial_absolute_difference(ssronn, slice_idx, ip_img, tar_img) * 3.33333

    srcnn_disp = cv2.resize(srcnn_disp.cpu().numpy(), (dim, dim), interpolation=cv2.INTER_NEAREST)
    sronn_disp = cv2.resize(sronn_disp.cpu().numpy(), (dim, dim), interpolation=cv2.INTER_NEAREST)
    ssronn_disp = cv2.resize(ssronn_disp.cpu().numpy(), (dim, dim), interpolation=cv2.INTER_NEAREST)


    for offset, img in zip(range(dim+15, dim*4+15, dim+10), [srcnn_disp, sronn_disp, ssronn_disp]):
        img = (img*255).astype(np.uint8)    # Convert to uint8 for applyColorMap
        im_colour = cv2.cvtColor(cv2.applyColorMap(img, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        out_img[global_offset+dim+10:2*dim+global_offset+10, hsi_width+offset:hsi_width+offset+dim, :] = im_colour/255 

    # Add colour map
    colour_map = np.array([[*range(255, -1, -math.ceil(256/dim))]] * 10, dtype=np.uint8)
    colour_map = np.transpose(colour_map)
    colour_map = cv2.cvtColor(cv2.applyColorMap(colour_map, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    out_img[global_offset+dim+10:dim+colour_map.shape[0]+global_offset+10, hsi_width+offset+dim+10:hsi_width+offset+dim+10+colour_map.shape[1], :] = colour_map / 255

    return out_img

def get_patch_coords(hsi, tar_slice_np):
    # Locate patch in hsi
    lowest_mse = 1
    for x in range(hsi.shape[0] - tar_slice_np.shape[0]):
        for y in range(hsi.shape[1] - tar_slice_np.shape[1]):
            mse = ((hsi[x:x+tar_slice_np.shape[0], y:y+tar_slice_np.shape[1], slice_idx] - tar_slice_np)**2).mean(axis=None)
            if mse < lowest_mse:
                lowest_mse = mse
                lowest_mse_coords = (x, y)
            if mse == 0:
                break
            
    return lowest_mse_coords

def make_objective_SR_figure():
    tar_slice_np = hr_tiles[img_idx_1][slice_idx].numpy()
    x, y = get_patch_coords(hsi, tar_slice_np)


    # Draw tile box on full hsi
    hsi_patch_loc = np.concatenate([np.expand_dims(hsi[:,:,slice_idx], axis=2)]*3, axis=2)
    out_img = np.ones((hsi_patch_loc.shape[0], hsi_patch_loc.shape[1]+(dim+10)*5, 3))
    hsi_patch_loc[x:x+tar_slice_np.shape[0], y, 0] = hsi.max()
    hsi_patch_loc[x:x+tar_slice_np.shape[0], y+tar_slice_np.shape[1], 0] = hsi.max()
    hsi_patch_loc[x, y:y+tar_slice_np.shape[1], 0] = hsi.max()
    hsi_patch_loc[x+tar_slice_np.shape[0], y:y+tar_slice_np.shape[1], 0] = hsi.max()

    go = hsi_patch_loc.shape[0] - dim*4-48      # global offset for first pred image group
    out_img = add_predictions_and_spatial_difference_to_figure(out_img, img_idx_1, slice_idx, hsi_patch_loc, dim=dim, global_offset=go)
    # Draw Red box around predictions from red patch (since white, make B and G 0)
    out_img[go-3:go+dim*2+12, hsi_patch_loc.shape[1]+2:hsi_patch_loc.shape[1]+4, [1,2]] = 0   # Left vertical line
    out_img[go-3:go+dim*2+12, -4:-2, [1,2]] = 0   # Right vertical line
    out_img[go+dim*2+11:go+dim*2+13, hsi_patch_loc.shape[1]+2:-2, [1,2]] = 0    # Bottom horizontal line
    out_img[go-3:go-1, hsi_patch_loc.shape[1]+2:-2, [1,2]] = 0  # top horizontal line


    #plt.imshow(out_img, cmap='gray')
    #plt.show()

    tar_slice_np = hr_tiles[img_idx_2][slice_idx].numpy()

    x, y = get_patch_coords(hsi, tar_slice_np)

    # Draw tile box on full hsi
    hsi_patch_loc[x:x+tar_slice_np.shape[0], y, 1] = hsi.max()
    hsi_patch_loc[x:x+tar_slice_np.shape[0], y+tar_slice_np.shape[1], 1] = hsi.max()
    hsi_patch_loc[x, y:y+tar_slice_np.shape[1], 1] = hsi.max()
    hsi_patch_loc[x+tar_slice_np.shape[0], y:y+tar_slice_np.shape[1], 1] = hsi.max()

    go = hsi_patch_loc.shape[0] - dim*2 - 13 # global offset for second pred image group (300)
    out_img = add_predictions_and_spatial_difference_to_figure(out_img, img_idx_2, slice_idx, hsi_patch_loc, global_offset=go, dim=dim)
    # Draw green box around predictions from green patch (since white, make R and B 0)
    out_img[go-3:go+dim*2+12, hsi_patch_loc.shape[1]+2:hsi_patch_loc.shape[1]+4, [0,2]] = 0     # Left vertical line
    out_img[go-3:go+dim*2+12, -4:-2, [0,2]] = 0     # Right vertical line
    out_img[go+dim*2+11:go+dim*2+13, hsi_patch_loc.shape[1]+2:-2, [0,2]] = 0      # Bottom horizontal line
    out_img[go-3:go-1, hsi_patch_loc.shape[1]+2:-2, [0,2]] = 0        # Top horizontal line

    #plt.imshow(hsi_patch_loc)
    #plt.show()


    title = dataset 
    if is_residual: title += ' residual'
    if norm_type != 'none': title += ' ' + norm_type

    #plt.imshow(out_img)
    #plt.title(title)
    #plt.show()


    out_img = (out_img*255).astype(np.uint8)    # Convert to uint8

    out_img=cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 0, 0)   # Black 
    thickness = 1   # Line thickness of 2 px

    row=40
    org = (395, row)
    out_img = cv2.putText(out_img, 'LR', org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (520, row)
    out_img = cv2.putText(out_img, 'SRCNN', org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (656, row)
    out_img = cv2.putText(out_img, 'SRONN', org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (790, row)
    out_img = cv2.putText(out_img, 'sSRONN', org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (945, row)
    out_img = cv2.putText(out_img, 'HR', org, font, fontScale, color, thickness, cv2.LINE_AA)

    # Colour map scales
    fontScale = 0.2
    col = 910
    org = (col, 314)     # x, y  /  cols, rows
    out_img = cv2.putText(out_img, '0', org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (col, 274)
    out_img = cv2.putText(out_img, '0.1', org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (col, 234)
    out_img = cv2.putText(out_img, '0.2', org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (col, 194)
    out_img = cv2.putText(out_img, '0.3', org, font, fontScale, color, thickness, cv2.LINE_AA)

    org = (col, 605)
    out_img = cv2.putText(out_img, '0', org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (col, 565)
    out_img = cv2.putText(out_img, '0.1', org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (col, 525)
    out_img = cv2.putText(out_img, '0.2', org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (col, 485)
    out_img = cv2.putText(out_img, '0.3', org, font, fontScale, color, thickness, cv2.LINE_AA)


    cv2.imwrite(title+'.jpg', out_img)
    cv2.imshow(title, out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_true_predictions_to_figure(out_img, img_idx, slice_idx, hsi_patch_loc, global_offset=5, dim=128):
    def get_pred_slice(model, ip_img, slice_idx):
        with torch.no_grad():
            pred = torch.clamp(model(ip_img.unsqueeze(0)), min=0, max=1)
            
            pred_slice = pred[0, slice_idx, :,:]
        return pred_slice.cpu().numpy()

    ip_img = hr_tiles[img_idx]
    ip_img = ip_img.permute(1,2,0).numpy()
   
    w, h = ip_img.shape[0]*2, ip_img.shape[1]*2
    img_up = cv2.resize(ip_img, (w, h), interpolation=cv2.INTER_CUBIC)   # 2x bicubic upsampling

    img_up = torch.Tensor(img_up).to(device)
    img_up = img_up.permute(2,0,1)

    srcnn_pred = get_pred_slice(srcnn, img_up, slice_idx)
    sronn_pred = get_pred_slice(sronn, img_up, slice_idx)
    ssronn_pred = get_pred_slice(ssronn, img_up, slice_idx)

    ip_slice = hr_tiles[img_idx][slice_idx].numpy()
    ip_slice = cv2.resize(ip_slice, (dim, dim), interpolation=cv2.INTER_CUBIC)

    out_img[:, :hsi_patch_loc.shape[1], :] = hsi_patch_loc
    hsi_width = hsi_patch_loc.shape[1]
    for offset, img in zip(range(5, dim*4+10, dim+10), [ip_slice, srcnn_pred, sronn_pred, ssronn_pred]):
        for i in range(3):
            out_img[global_offset:dim+global_offset, hsi_width+offset:hsi_width+offset+dim, i] = img    # All cahnnels same for grayscale

    return out_img

def make_true_SR_figure():
    hsi_patch_loc = np.stack([hsi[:,:,slice_idx]]*3, axis=2)  # Make grayscale
    out_img = np.ones((hsi_patch_loc.shape[0], hsi_patch_loc.shape[1]+(dim+10)*4, 3))
        
    for img_idx, colour in zip([*range(4)], ["RED", "GREEN", "BLUE", "YELLOW"]):
        tar_slice_np = hr_tiles[img_idx][slice_idx].numpy()
        x, y = get_patch_coords(hsi, tar_slice_np)

        if colour == "RED":
            c_dim = 0
            nc_dim = [1,2]
        elif colour == "GREEN":
            c_dim = 1
            nc_dim = [0,2]
        elif colour == "BLUE":
            c_dim = 2
            nc_dim = [0,1]
        elif colour == "YELLOW":
            c_dim = [0, 1]
            nc_dim = 2

        # Draw tile box on full hsi
        hsi_patch_loc[x:x+tar_slice_np.shape[0], y, c_dim] = hsi.max()
        hsi_patch_loc[x:x+tar_slice_np.shape[0], y+tar_slice_np.shape[1], c_dim] = hsi.max()
        hsi_patch_loc[x, y:y+tar_slice_np.shape[1], c_dim] = hsi.max()
        hsi_patch_loc[x+tar_slice_np.shape[0], y:y+tar_slice_np.shape[1], c_dim] = hsi.max()

        go = hsi_patch_loc.shape[0] + (4-img_idx)*(-dim-12)      # global offset for pred image group
        #print(go, colour, c_dim, nc_dim)
        out_img = add_true_predictions_to_figure(out_img, img_idx, slice_idx, hsi_patch_loc, dim=dim, global_offset=go)
        # Draw Red box around predictions from red patch (since white, make B and G 0)
        out_img[go-3:go+dim+3, hsi_patch_loc.shape[1]+2:hsi_patch_loc.shape[1]+4, nc_dim] = 0   # Left vertical line
        out_img[go-3:go+dim+3, -4:-2, nc_dim] = 0   # Right vertical line
        out_img[go+dim+1:go+dim+3, hsi_patch_loc.shape[1]+2:-2, nc_dim] = 0    # Bottom horizontal line
        out_img[go-3:go-1, hsi_patch_loc.shape[1]+2:-2, nc_dim] = 0  # top horizontal line


    #plt.imshow(out_img, cmap='gray')
    #plt.show()

    '''
    tar_slice_np = hr_tiles[img_idx_2][slice_idx].numpy()

    x, y = get_patch_coords(hsi, tar_slice_np)

    # Draw tile box on full hsi
    hsi_patch_loc[x:x+tar_slice_np.shape[0], y, 1] = hsi.max()
    hsi_patch_loc[x:x+tar_slice_np.shape[0], y+tar_slice_np.shape[1], 1] = hsi.max()
    hsi_patch_loc[x, y:y+tar_slice_np.shape[1], 1] = hsi.max()
    hsi_patch_loc[x+tar_slice_np.shape[0], y:y+tar_slice_np.shape[1], 1] = hsi.max()

    go = hsi_patch_loc.shape[0] - dim*2 - 13 # global offset for second pred image group (300)
    out_img = add_true_predictions_to_figure(out_img, img_idx_2, slice_idx, hsi_patch_loc, global_offset=go, dim=dim)
    # Draw green box around predictions from green patch (since white, make R and B 0)
    out_img[go-3:go+dim*2+12, hsi_patch_loc.shape[1]+2:hsi_patch_loc.shape[1]+4, [0,2]] = 0     # Left vertical line
    out_img[go-3:go+dim*2+12, -4:-2, [0,2]] = 0     # Right vertical line
    out_img[go+dim*2+11:go+dim*2+13, hsi_patch_loc.shape[1]+2:-2, [0,2]] = 0      # Bottom horizontal line
    out_img[go-3:go-1, hsi_patch_loc.shape[1]+2:-2, [0,2]] = 0        # Top horizontal line
    '''

    #plt.imshow(hsi_patch_loc)
    #plt.show()


    title = dataset 
    if is_residual: title += ' residual'
    if norm_type != 'none': title += ' ' + norm_type
    title += ' true SR'

    #plt.imshow(out_img)
    #plt.title(title)
    #plt.show()


    out_img = (out_img*255).astype(np.uint8)    # Convert to uint8

    out_img=cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 0, 0)   # Black 
    thickness = 1   # Line thickness of 2 px

    row=40
    org = (380, row)
    out_img = cv2.putText(out_img, 'BICUBIC', org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (520, row)
    out_img = cv2.putText(out_img, 'SRCNN', org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (656, row)
    out_img = cv2.putText(out_img, 'SRONN', org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (790, row)
    out_img = cv2.putText(out_img, 'sSRONN', org, font, fontScale, color, thickness, cv2.LINE_AA)
   

    cv2.imwrite(title+'.jpg', out_img)
    cv2.imshow(title, out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


make_objective_SR_figure()
#make_true_SR_figure()