import os
import argparse
import copy
import numpy as np
import chainer
from chainer.datasets import ConcatenatedDataset
from chainer.datasets import TransformDataset
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.links import FasterRCNNVGG16

from chainercv import transforms
from chainercv.visualizations import vis_bbox
import sys
#import matplotlib.pyplot as plt
print('start')
def print_(x):
      print(x)

chainer.config.cv_resize_backend = "cv2"
INFER_DIR = 'voc07_trans'


args = sys.argv
config_file = os.path.splitext(args[1])[0]
config_module = __import__(config_file)
Config = config_module.Config
config = Config()

TARGET_MODEL_DIR  = config.TARGET_MODEL_DIR

ATTACK_MODEL_DIR  = config.ATTACK_MODEL_DIR


print(ATTACK_MODEL_DIR)
MAX_LEN          = config.MAX_LEN
CANVAS_TYPE      = config.CANVAS_TYPE
ATTACK_MODEL     = config.ATTACK_MODEL
if 'ori' in ATTACK_MODEL_DIR:
    CANVAS_TYPE = 'original'
else:
    CANVAS_TYPE = 'uniform'
if 'shallow' in ATTACK_MODEL_DIR:
    ATTACK_MODEL = 'shallow'

LOG_SCORE = config.LOG_SCORE
if 'LOG_SCORE_2' in ATTACK_MODEL_DIR:
    LOG_SCORE = 2
if LOG_SCORE == 2:
    def logscore(a):
        r = -np.log2(1-a+1e-20)
        return r
elif LOG_SCORE>0:
    def logscore(a):
        r = -np.log(1-a)
        return r
else:
    def logscore(a):
        r= a
        raise ZeroDivisionError
        return r
print('CANVAS_TYPE {}'.format(CANVAS_TYPE))  
print('ATTACK_MODEL {}'.format(ATTACK_MODEL))              
print('LOG_SCORE {}'.format(LOG_SCORE))


EVAL_LAST= False
id_ = 'ssd_500k_lrdrop_right_pretrainv7'
if 'left' in TARGET_MODEL_DIR or 'target' in TARGET_MODEL_DIR:
    subset = 'left'
elif 'right' in TARGET_MODEL_DIR or 'shadow' in TARGET_MODEL_DIR:
    subset = 'right'
else:
    raise ZeroDivisionError
TARGET_MODEL_TYPE = config.TARGET_MODEL_TYPE
if 'res50' in TARGET_MODEL_DIR:
    TARGET_MODEL_TYPE = 'ssd_res50'
elif 'ssd512' in TARGET_MODEL_DIR:
    TARGET_MODEL_TYPE = 'ssd512_vgg'
elif 'fastercnn' in TARGET_MODEL_DIR:
    TARGET_MODEL_TYPE = 'FR_vgg'
    MIN_SIZE=600 # 600 by default
    MAX_SIZE=800 # 800 by default
    batch_size  = 1
    
print('TARGET_MODEL_TYPE : {}'.format(TARGET_MODEL_TYPE))

START   = config.START
END     = config.END
INTERVAL= config.INTERVAL


gpu_id = config.gpu_id

batch_size  = config.batch_size
if not os.path.exists(INFER_DIR):
    os.makedirs(INFER_DIR)

if TARGET_MODEL_TYPE =='ssd300_vgg':
    model = SSD300(n_fg_class=21,pretrained_model='imagenet')
elif TARGET_MODEL_TYPE =='ssd512_vgg':
    model = SSD512(n_fg_class=21,pretrained_model='imagenet')
    #END = 450000
elif TARGET_MODEL_TYPE=='ssd_res50':
    from ssd_res50 import SSD300_Resnet
    model = SSD300_Resnet(n_fg_class=21,pretrained_model='imagenet')
elif TARGET_MODEL_TYPE =='FR_vgg':
    model = FasterRCNNVGG16(n_fg_class=len(voc_bbox_label_names),
                                  pretrained_model='imagenet',min_size=MIN_SIZE,max_size=MAX_SIZE)
model.to_gpu(gpu_id)
train07 = VOCBboxDataset(data_dir='auto',year='2007', split='trainval',use_difficult=True,return_difficult=False)
train12 = VOCBboxDataset(data_dir='auto',year='2012', split='trainval',use_difficult=True,return_difficult=False)

train = ConcatenatedDataset(train07,train12)
test   = VOCBboxDataset(data_dir='auto',year='2007', split='test',use_difficult=True,return_difficult=True)
test_iter = chainer.iterators.SerialIterator(test,batch_size,repeat=False,shuffle=False)

print('image prepare')
if subset == 'left':
    train07_left = train07[:len(train07)//2]
    #train12_left = train12[:len(train12)//2]
    test_left    = test[:len(test)//2]
    print('load images')
    test_imgs_subset    = [item[0] for item in test_left   ]
    train07_imgs_subset = [item[0] for item in train07_left]
    #test12_img  = [item[0] for item in test12 ]

elif subset == 'right':
    train07_right = train07[len(train07)//2:]
    #train12_right = train12[len(train12)//2:]
    test_right    = test[len(test)//2:]
    print('load images')
    test_imgs_subset    = [item[0] for item in test_right   ]
    train07_imgs_subset = [item[0] for item in train07_right]




model.to_cpu()
model.to_gpu()

model.use_preset('evaluate')
model.score_thresh                       = config.model_score_thresh
model.nms_thresh                         = config.model_nms_thresh
if TARGET_MODEL_TYPE =='FR_vgg': #RPN Setting
    model.score_thresh                       = config.model_score_thresh
    model.nms_thresh                         = config.model_nms_thresh

    model.rpn.proposal_layer.nms_thresh      = config.model_rpn_proposal_layer_nms_thresh
    model.rpn.proposal_layer.n_test_pre_nms  = config.model_rpn_proposal_layer_n_test_pre_nms
    model.rpn.proposal_layer.n_test_post_nms = config.model_rpn_proposal_layer_n_test_post_nms
    MAX_LEN           = config.MAX_LEN      
    if 'normal' in TARGET_MODEL_DIR:
        START=25000
        END  =400000
        INTERVAL=25000
    else:
        START   =100000
        END     =800000
        INTERVAL=100000
if EVAL_LAST:
    START=END
def generate_pointsets(imgs,model,num_logit_feature=1,max_len=MAX_LEN,input_size = 300,regard_in_set=True):
    
    size300_train07_dataset = []
    max_feat_len = 0
    min_feat_len = 50000
    mark = 0
    jump = 10
    with chainer.using_config('train', False),chainer.function.no_backprop_mode():
        while ( mark  < len(imgs ) ):
            tr_bboxes_list=[]
            tr_labels_list=[]
            tr_scores_list=[]
            tr_img_sizes= []
            x = []
            sizes = []
            #for img in imgs[mark:mark+jump]:
                
            bboxes,labels,scores = model.predict(imgs[mark:mark+10])
            tr_bboxes_list +=bboxes
            tr_labels_list +=labels
            tr_scores_list +=scores
            for img in imgs[mark:mark+10]:
                _,H,W = img.shape
                tr_img_sizes.append((H,W))
            
                
           
            bbox_list  = tr_bboxes_list
            bbox_size  = tr_img_sizes
            score_list = tr_scores_list
            label_list = tr_labels_list
            bbox_size300 = np.zeros_like(bbox_list,dtype=np.float32)
            bbox_size300_list = []
            for bboxes,size,score,labels in zip(bbox_list,bbox_size,score_list,label_list):
                bboxes_300 = np.zeros((bboxes.shape[0],bboxes.shape[1]+num_logit_feature*2),dtype=np.float32)
                bboxes_300[:,0]  =  bboxes[:,0] / size[0]
                bboxes_300[:,2]  =  bboxes[:,2] / size[0]

                bboxes_300[:,1]  =  bboxes[:,1] / size[1]
                bboxes_300[:,3]  =  bboxes[:,3] / size[1]
                bboxes_300[:,4] =  score
                bboxes_300[:,4+num_logit_feature] =  labels

                bbox_size300_list.append(bboxes_300)
                
                
            max_feat_len_now = max([len(item) for item in bbox_size300_list])
            if max_feat_len < max_feat_len_now :
                max_feat_len = max_feat_len_now
                print('max feat len : {}'.format(max_feat_len))
            
            min_feat_len_now = min([len(item) for item in bbox_size300_list])
            if min_feat_len > min_feat_len_now:
                min_feat_len = min_feat_len_now
                #print('min feat len : {}'.format(min_feat_len))
                      
            bbox_size300_list = np.array( [ item[:max_len] for item in bbox_size300_list   ] )
            size300_train07=np.array([np.pad(item,((0,max_len-len(item)),(0,0)),'constant') for item in bbox_size300_list])
            
            if regard_in_set:
                size300_train07_dataset += [(a,b) for a,b in zip(size300_train07, np.ones(len(size300_train07),dtype=np.float32)  )]
            else:
                size300_train07_dataset += [(a,b) for a,b in zip(size300_train07, np.zeros(len(size300_train07),dtype=np.float32) )]
            if mark +  10 > len(imgs):
                jump = len(imgs) - mark
            mark += jump
            #print(mark)
    return size300_train07_dataset


import numpy as np
import os
import sys
import os
import copy
import numpy as np
#import matplotlib.pyplot as plt
import logging
#from skimage import io

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
import sys
import time

if ATTACK_MODEL == 'alex':
    from torchvision.models import alexnet
    net = alexnet(num_classes =2)
elif ATTACK_MODEL == 'shallow':
    from shallow_model import shallowNet
    net = shallowNet(num_classes =2)
else:
    raise ZeroDivisionError
net.to(torch.float)
if torch.cuda.is_available():
    print_('Use GPU')
    net.cuda()
#net.train();   
net.eval();
net.load_state_dict(torch.load(ATTACK_MODEL_DIR))
epoch = 0
for iter in range(START,END + 1,INTERVAL):
    load_dir  = TARGET_MODEL_DIR.replace('100000',str(iter))
    chainer.serializers.load_npz(os.path.join(load_dir), model,)
    print_('current iter : {}'.format(iter))

    # size300_meta_train_dataset = []

    # temp=generate_pointsets(train07_imgs_subset[:len(train07_imgs_subset)/2+1],model,regard_in_set=True,num_logit_feature=1)
    # size300_meta_train_dataset += temp
    # temp=generate_pointsets(test_imgs_subset[:len(test_imgs_subset)/2+1],model,regard_in_set=False,num_logit_feature=1)
    # size300_meta_train_dataset += temp


    size300_meta_evaluate_dataset = []
    size300_meta_evaluate_dataset += generate_pointsets(train07_imgs_subset,model,regard_in_set=True,num_logit_feature=1)
    size300_meta_evaluate_dataset += generate_pointsets(test_imgs_subset,model,regard_in_set=False,num_logit_feature=1)

    #a_train = np.array(size300_meta_train_dataset)
    a_eval  = np.array(size300_meta_evaluate_dataset)


    SAVE_MODEL        = config.SAVE_MODEL       #True
    TRANSFORM         = config.TRANSFORM        #True
    PRETRAIN          = config.PRETRAIN         #False
    NORMALIZE_CANVAS  = config.NORMALIZE_CANVAS #False
    shuffle_boxes     = config.shuffle_boxes    #False


    input_size = 300
    train_test_shuffle = 0 # 0: normal 1:shuffle 2: even odd shuffle 3 : odd even shuffle
    QUICK      = 0
    SMALL_SET  = 0
    EPOCHS     = 21
    if TRANSFORM:
        EPOCHS = 61
    if SMALL_SET>0:
        a_train = a_train[len(a_train)//2-SMALL_SET:len(a_train)//2+SMALL_SET]
    if PRETRAIN:
        id_     += '_pretrain_'
    ball_size = input_size*0.1




    np.random.seed(777)
   


    def make_canvas_circle_data(a_data,ball_size=ball_size,QUICK=0):
        H,W = (input_size,input_size)
        y, x = np.mgrid[0:H, 0:W]
        (xc,yc) = (input_size//2,input_size//2)
        domain_radius =  ball_size/2
        domain_radius = int(domain_radius)
        sigma_x,sigma_y = (ball_size,ball_size)

        a = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
        c = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)

        domain  =  (x-xc)**2 + (y-yc)**2 <= domain_radius**2
       
        g  =     np.where(domain, np.ones_like(x) , np.zeros_like(x))



        y, x = np.mgrid[0:H, 0:W]
        canvas_dataset_train=[]
        tt = time.time()
        for item in a_data:

            feats,label = item
            #break

            canvas = np.zeros((input_size,input_size),np.float32)
            if QUICK:
                num_feats = QUICK
            else:
                num_feats = len(feats)
            for i in range(num_feats):
                feat = feats[i]
                if np.sum(feat) < 1e-5:
                    break
                x0,y0,x1,y1=[int(item*input_size) for item in feat[:4] ]
                score  =feat[4]
                if score < 1e-8:
                    break
                x_c,y_c =  ( (x0+x1)//2,(y0+y1)//2)
                
                if LOG_SCORE:
                    score = logscore(score)
                if CANVAS_TYPE == 'uniform':
                    canvas[x_c-domain_radius:x_c+domain_radius+1,y_c-domain_radius:y_c+domain_radius+1] += score     
                elif CANVAS_TYPE == 'original':
                    canvas[x0:x1+1,y0:y1+1] += score     

        #break
            canvas_dataset_train.append([canvas,label])
        #break
        canvas_dataset_train = np.array(canvas_dataset_train)
        print_('time spent : {}'.format(time.time()-tt))
        return np.array(canvas_dataset_train)

        

    canvas_dataset_eval =make_canvas_circle_data( a_eval,QUICK=QUICK)

    
    canvas_dataset_eval =[list(item) for item in canvas_dataset_eval]


    class canvas_dataset(torch.utils.data.Dataset):
        def __init__(self, canvas_dataset,transform = False):
            self.canvas_dataset = canvas_dataset
            self.transform      = transform
        def __len__(self):
            'Denotes the total number of samples'
            return len(self.canvas_dataset)
        def __getitem__(self,i):
            images = self.canvas_dataset[i][0]
            labels = self.canvas_dataset[i][1]
            labels = int(labels)
            if self.transform:
                if np.random.randint(2):
                    if np.random.randint(2): # 50 % prob.
                        images = np.fliplr(images)
                if np.random.randint(2):
                    rot_k = np.random.randint(4)
                    images = np.rot90(images,k=rot_k)
            images = images[...,np.newaxis]
            images = np.transpose(images,[2,0,1])
            images = np.tile(images,[3,1,1])
            return images,labels
    # train = canvas_dataset(canvas_dataset_train,transform=TRANSFORM)
    # valid = canvas_dataset(canvas_dataset_valid, transform=False   )
    # test  = canvas_dataset(canvas_dataset_test , transform=False   )
    eval  = canvas_dataset(canvas_dataset_eval , transform=False   )

    # train_loader = torch.utils.data.DataLoader(train, batch_size=8,shuffle=True )
    # valid_loader = torch.utils.data.DataLoader(valid, batch_size=8,shuffle=False)
    # test_loader  = torch.utils.data.DataLoader(test , batch_size=8,shuffle=False)
    eval_loader  = torch.utils.data.DataLoader(eval , batch_size=8,shuffle=False)

    if PRETRAIN:
        pretrain = canvas_dataset(canvas_dataset_pretrain,transform=TRANSFORM)
        pretrain_loader = torch.utils.data.DataLoader(pretrain, batch_size=8,shuffle=True )
    
    
    step_start = 0  
    weight_decay = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=weight_decay)
    cross_entropy_loss = nn.CrossEntropyLoss()
    train_loss = 0
    train_loss_batch = 0
    def print_summary(summary):
            
            print_(' ')
            print_(summary['tag'])
            print_('iter : {}      '.format(summary['iter']     ))
            print_('recall_in  : {}'.format(summary['recall_in'] ))     
            print_('recall_out : {}'.format(summary['recall_out']))
            print_('recall_all : {}'.format(summary['recall_all']))    
            print_('pred_in    : {}'.format(summary['pred_in']   ))   
            print_('f1_score   : {}'.format(summary['f1_score']  ))    
            print_('accuracy   : {}'.format(summary['accuracy']  ))   
    def get_acc(data,loader,net,tag='test' , print_all=True):
    
            correct_num     = 0
            correct_num_in  = 0
            correct_num_out = 0
            pred_as_in_num  = 0
            pred_as_out_num = 0
            total_num_in    = 0
            total_num_out   = 0 
            for batch in loader: #Test
                with torch.no_grad():
                    #print(0)
                    images = batch[0]
                    labels = batch[1]
                    if torch.cuda.is_available():
                        images = images.cuda()
                        labels = labels.cuda()
                    labels_pred = net.forward(images)
                    labels_pred = labels_pred.cpu().numpy()
                    pred_as_in_num  += np.sum( np.argmax(labels_pred,axis=1) == 1 )
                    correct_num     += np.sum( np.argmax(labels_pred,axis=1) == labels.cpu().numpy() )
                    correct_num_in  += np.sum( np.logical_and( np.argmax(labels_pred,axis=1) == labels.cpu().numpy() , labels.cpu().numpy()==1 ))
                    total_num_in    += np.sum(labels.cpu().numpy() ==1)
                    total_num_out   += np.sum(labels.cpu().numpy() ==0)
                    correct_num_out += np.sum( np.logical_and( np.argmax(labels_pred,axis=1) == labels.cpu().numpy() , labels.cpu().numpy()==0 ))  
           
            total_num = total_num_in+total_num_out
            recall_in = float(correct_num_in ) / total_num_in
            pred_in   = float(correct_num_in ) / pred_as_in_num  if pred_as_in_num > 0.5 else 0 
            summary    = {}
            summary['tag'       ] = tag 
            summary['iter'      ] = iter 
            summary['total_num' ] = total_num
            summary['recall_in' ] = recall_in
            summary['recall_out'] = float(correct_num_out) / total_num_out
            summary['recall_all'] = ( summary['recall_in' ] + summary['recall_out'] ) / 2
            summary['pred_in'   ] = pred_in
            summary['accuracy'  ] = float(correct_num)     / total_num
            summary['f1_score'  ] = 2*recall_in*pred_in/( recall_in+pred_in ) if recall_in+pred_in> 1e-5 else 0
            if print_all:
                print_summary(summary)

            return summary['accuracy'] , summary
    
    
    
    best_acc=0
  
    _,best_eval_summary=get_acc(eval ,eval_loader ,net,tag='eval' ,print_all=True)

    






