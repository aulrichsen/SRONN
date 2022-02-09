import cv2
import numpy as np
import pywt
import tensorflow as tf
import math

def bicubic_lr (img, ratio):
# Function to create Low Resolution images using Bicubic Interpolation
    if ratio == 2:
        sigma = 0.8943
    else:
        sigma = 1.6986

    [h, w, d] = img.shape
    
    img = cv2.GaussianBlur(img, (0, 0), sigma, sigma) # run in a loop
    
    im_down = img[::ratio,::ratio, :]
    img_up = cv2.resize(im_down, (w, h),interpolation=cv2.INTER_CUBIC)
    return img_up


def bilinear_lr (img, ratio):
# Function to create Low Resolution images using Bilinear Interpolation
    if ratio == 2:
        sigma = 0.8943
    else:
        sigma = 1.6986

    [h, w, d] = img.shape
    
    img = cv2.GaussianBlur(img, (0, 0), sigma, sigma) # run in a loop
    
    im_down = img[::ratio,::ratio, :]
    img_up = cv2.resize(im_down, (w, h),interpolation=cv2.INTER_LINEAR)
    return img_up



def wavelet_lr (img, ratio, bands):
    if ratio == 2:
        sigma = 0.8943
    else:
        sigma = 1.6986

    [h, w, d] = img.shape

    img = cv2.GaussianBlur(img, (0, 0), sigma, sigma) # run in a loop

    im_down = img[::ratio,::ratio, :]
    img_up = cv2.resize(im_down, (w, h),interpolation=cv2.INTER_CUBIC)
    LL=[]
    HL = []
    LH = []
    HH = []
    All = []
    for i in range (bands):
        ll, (lh,hl,hh) = pywt.dwt2(img_up[:,:,i], 'haar')
        #[h, w, d] = img.shape
        #ll = cv2.resize(ll, (w, h),interpolation=cv2.INTER_CUBIC)
        #lh = cv2.resize(lh, (w, h),interpolation=cv2.INTER_CUBIC)
        #hl = cv2.resize(hl, (w, h),interpolation=cv2.INTER_CUBIC)
        #hh = cv2.resize(hh, (w, h),interpolation=cv2.INTER_CUBIC)
        LL.append(ll)
        LH.append(lh)
        HL.append(hl)
        HH.append(hh)
    #LL = np.array(LL)
    #LL = np.moveaxis(LL, 0, -1)    
    #print(LL.shape)  
    #LH = np.array(LH)
    #LH = np.moveaxis(LH, 0, -1)  
    
    #HL = np.array(HL)
    #HL = np.moveaxis(HL, 0, -1)  
    
    #HH = np.array(HH)
    #HH = np.moveaxis(HH, 0, -1) 
    All.append(np.array(LL))
    All.append(np.array(LH))
    All.append(np.array(HL))
    All.append(np.array(HH))
    
    return All





def wavelet_lr_bicubic (img, ratio, bands):
    if ratio == 2:
        sigma = 0.8943
    else:
        sigma = 1.6986

    [h, w, d] = img.shape

    img = cv2.GaussianBlur(img, (0, 0), sigma, sigma) # run in a loop

    im_down = img[::ratio,::ratio, :]
    img_up = cv2.resize(im_down, (w, h),interpolation=cv2.INTER_CUBIC)
    LL=[]
    HL = []
    LH = []
    HH = []
    All = []
    for i in range (bands):
        ll, (lh,hl,hh) = pywt.dwt2(img_up[:,:,i], 'haar')
        [h, w, d] = img.shape
        #ll_up = cv2.resize(ll, (w, h),interpolation=cv2.INTER_CUBIC)
        #lh_up = cv2.resize(lh, (w, h),interpolation=cv2.INTER_CUBIC)
        #hl_up = cv2.resize(hl, (w, h),interpolation=cv2.INTER_CUBIC)
        #hh_up = cv2.resize(hh, (w, h),interpolation=cv2.INTER_CUBIC)
        LL.append(ll)
        LH.append(lh)
        HL.append(hl)
        HH.append(hh)

    All.append(np.array(LL))
    All.append(np.array(LH))
    All.append(np.array(HL))
    All.append(np.array(HH))
    
    return All




def wavelet_hr (img, bands):

    LL=[]
    HL = []
    LH = []
    HH = []
    All = []
    for i in range (bands):
        #[h, w] = img[:,:,i].shape
        #img_up = cv2.resize(img[:,:,i], (124, 124),interpolation=cv2.INTER_CUBIC)
        ll, (lh,hl,hh) = pywt.dwt2(img[:,:,i], 'haar')
        LL.append(ll)
        LH.append(lh)
        HL.append(hl)
        HH.append(hh)

    All.append(np.array(LL))
    All.append(np.array(LH))
    All.append(np.array(HL))
    All.append(np.array(HH))
    
    return All


def hsi_normalize_full(hsi):
  tmp = hsi - np.min(hsi)
  img = tmp/np.max(tmp)
  return img


def tnsrTo2Dstack (tnsr):
    # Funtion to convert 3D data cubes into 2D image stacks
    [b,r,c,d] = tnsr.shape
    stack2d = np.zeros((b*d,r,c), dtype = float)
    ctr = 0
    for i in range(b):
        for j in range(d):
            stack2d[ctr,:,:] = tnsr[i,:,:,j]
            ctr = ctr + 1
            
    return stack2d

def stackToTnsr (stack, num_bands):    
    [num_imgs,h,w] = stack.shape
    btch_size = num_imgs//num_bands
    tnsr = np.zeros((btch_size,h,w,num_bands), dtype = float)
    ctr = 0
    for i in range(btch_size):
        for j in range(num_bands):
            tnsr[i,:,:,j] = stack[ctr,:,:]
            #print(str(i) + " , " + str(j))
            ctr = ctr+1

    return tnsr

        

def scheduler(epoch, lr):
  #if epoch % 10 == 0:
  #    lr = lr /(tf.math.exp(-0.1))
  if epoch == 10:
      lr = lr /2
  elif epoch == 20:
      lr = lr /4
  #elif epoch < 150:
  #    lr = 1e-5 / 4
  #elif epoch < 200:
  #

  else:
      lr = lr
      
  return lr



#def scheduler(epoch):
#  if epoch == 50:
#      lr = 1e-5
#  elif epoch >50 and epoch < 100:
#      lr = 1e-5 * tf.math.exp(-0.1)
#  elif epoch >= 100 and epoch < 150:
#      lr = 1e-5 * tf.math.exp(-0.01)
#  elif epoch >= 150 and epoch < 200:
#      lr = 1e-5 * tf.math.exp(-0.001)
#  else:
#      lr = 1e-5 * tf.math.exp(-0.0001)
      
#  return lr


#def scheduler(epoch,lr):
#  if epoch < 50:
#      lr = 1e-5
#  elif epoch >=50 and epoch < 100:
#      lr = lr * tf.math.exp(-0.1)
#  elif epoch >=100 and epoch < 150:
#      lr = lr * tf.math.exp(-0.1)
#  elif epoch >=150 and epoch < 200:
#      lr = lr * tf.math.exp(-0.1)
#  else:
#      lr = lr * tf.math.exp(-0.1)
      
#  return lr

def scheduler2(epoch, lr):
   lr_init = 1e-5/16
   interval = 50
   
   #lr = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
   lr = lr_init * (1./2.)**( epoch // interval)
  
   return lr
















