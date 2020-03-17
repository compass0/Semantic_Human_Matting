'''
    test camera 

Author: Zhengwei Li
Date  : 2018/12/28
'''
import time
import cv2
import torch 
import argparse
import numpy as np
import os 
import torch.nn.functional as F
from model import network

parser = argparse.ArgumentParser(description='human matting')
parser.add_argument('--model', default='./ckpt/human_matting/model/model_obj.pth', help='preTrained model')
parser.add_argument('--size', type=int, default=256, help='input size')
parser.add_argument('--without_gpu', action='store_true', default=False, help='no use gpu')

args = parser.parse_args()

torch.set_grad_enabled(False)

    
#################################
#----------------
if args.without_gpu:
    print("use CPU !")
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print("----------------------------------------------------------")
        print("|       use GPU !      ||   Available GPU number is {} !  |".format(n_gpu))
        print("----------------------------------------------------------")

        device = torch.device('cuda:0,1')

#################################
#---------------
def load_model(args):
    print('Loading model from {}...'.format(args.model))
    if args.without_gpu:
        myModel = torch.load(args.model, map_location=lambda storage, loc: storage)
    else:
        myModel = torch.load(args.model)
    myModel.eval()
    myModel.to(device)
    
    return myModel

def seg_process(args, image, background, net):

    # opencv
    origin_h, origin_w, c = image.shape
    image_resize = cv2.resize(image, (args.size,args.size), interpolation=cv2.INTER_CUBIC)
    image_resize = (image_resize - (104., 112., 121.,)) / 255.0



    tensor_4D = torch.FloatTensor(1, 3, args.size, args.size)
    
    tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2,0,1))
    inputs = tensor_4D.to(device)

    t0 = time.time()

    trimap, alpha = net(inputs)
  

    if args.without_gpu:
        alpha_np = alpha[0,0,:,:].data.numpy()
    else:
        alpha_np = alpha[0,0,:,:].cpu().data.numpy()


    alpha_np = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)

    #fg = np.multiply(alpha_np[..., np.newaxis], image)
    
    bg = background

    # mask 추출 
    bg_gray = np.multiply(1-alpha_np[..., np.newaxis], image)
    bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(bg_gray, 50, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    mask_inv = cv2.bitwise_not(mask)

    # background 합성된 iamge
    bg = cv2.bitwise_and(bg, bg, mask=mask)
    fg = cv2.bitwise_and(image, image, mask=mask_inv)
    out1 = cv2.add(bg,fg)
    cv2.imwrite("./result/out1.png", out1)

    # backround 제거된 image (with alpha channel)
    b,g,r = cv2.split(image)
    rgba = [b,g,r,mask_inv]
    out2 = cv2.merge(rgba,4)
    cv2.imwrite("./result/out2.png", out2)

    
   # bg[:,:,0] = bg_gray
   # bg[:,:,1] = bg_gray
   # bg[:,:,2] = bg_gray

    #fg[fg<=0] = 0
    #fg[fg>255] = 255
    #fg = fg.astype(np.uint8)
    #out = cv2.addWeighted(fg, 0.7, bg, 0.3, 0)
    
    #out = fg + bg
    #out[out<0] = 0
    #out[out>255] = 255
    #out = out.astype(np.uint8)

    return out2


def test(args, net):

    frame = cv2.imread('./test2.jpg',cv2.IMREAD_COLOR)
    bg = cv2.imread('./bg.jpg',cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, (400,600), interpolation=cv2.INTER_CUBIC)
    bg= cv2.resize(bg, (400,600), interpolation=cv2.INTER_CUBIC)
    #frame = cv2.flip(frame,1)
    frame_seg = seg_process(args, frame, bg, net)
    print(frame_seg.shape)

    # show a frame
    cv2.imwrite('./result/test_t.jpg',frame_seg)


def main(args):

    myModel = load_model(args)
    test(args, myModel)


if __name__ == "__main__":
    main(args)


