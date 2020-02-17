from importlib import import_module
import os
import sys
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

from chainercv.links import SSD300
from chainercv.links import SSD512

from chainercv.links import FasterRCNNVGG16
from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain
from chainer.training.extensions.value_observation import observe_value
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
#import chainer.functions as F
from chainer.backends import cuda

from chainercv import transforms
from chainercv.visualizations import vis_bbox

import chainer.functions as F
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
from chainercv.utils import non_maximum_suppression
from chainercv.transforms.image.resize import resize

args = sys.argv
config_file = os.path.splitext(args[1])[0]
#print(config_file)
config_module = __import__(config_file)
Config = config_module.Config
#from config import Config
config = Config()

#import matplotlib.pyplot as plt
print('start')
def print_(x):
      print(x)
chainer.config.cv_resize_backend = "cv2"
INFER_DIR = 'voc07_trans'
SHADOW_MODEL_DIR  = config.SHADOW_MODEL_DIR
id_            =  config.id_

if 'left' in SHADOW_MODEL_DIR or 'target' in SHADOW_MODEL_DIR:
    subset = 'left'
elif 'right' in SHADOW_MODEL_DIR or 'shadow' in SHADOW_MODEL_DIR:
    subset = 'right'
else:
    raise ZeroDivisionError

    
SHADOW_MODEL_TYPE  = config.SHADOW_MODEL_TYPE


ATTACK_MODEL=config.ATTACK_MODEL 

print('SHADOW_MODEL_TYPE : {}'.format(SHADOW_MODEL_TYPE))
    



if not os.path.exists(INFER_DIR):
    os.makedirs(INFER_DIR)

gpu_id      = config.gpu_id
batch_size  = config.batch_size

PREDICT_ONE_EACH_BOX = config.PREDICT_ONE_EACH_BOX

if SHADOW_MODEL_TYPE == 'ssd300_vgg':
    model = SSD300(n_fg_class=21,pretrained_model='imagenet')
if SHADOW_MODEL_TYPE == 'ssd512_vgg':
    model = SSD512(n_fg_class=21,pretrained_model='imagenet')
if SHADOW_MODEL_TYPE == 'ssd_res50':
    model = SSD300_Resnet(n_fg_class=21,pretrained_model='imagenet')
if SHADOW_MODEL_TYPE == 'FR_vgg':
    MIN_SIZE=config.MIN_SIZE 
    MAX_SIZE=config.MAX_SIZE 
    model = FasterRCNNVGG16(n_fg_class=len(voc_bbox_label_names),
                                  pretrained_model='imagenet',min_size=MIN_SIZE,max_size=MAX_SIZE)
model.to_gpu(gpu_id)

if PREDICT_ONE_EACH_BOX:
    def _suppress_each_box(self, raw_cls_bbox, raw_prob):
        xp =  np #model.xp
        bbox = []
        label = []
        prob = []
        best_class = raw_prob.argmax(axis=1)
        self.raw_cls_bbox  = raw_cls_bbox   
        if True:
            best_class = best_class[:len(raw_cls_bbox)]
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4)) 
            
            cls_bbox_l = xp.array([cls_bbox_l[i,item,:] for i,item in enumerate(best_class)])
            
            
            non_bg_mask = best_class>0
            #prob_l = raw_prob[:, best_class]
            prob_l = xp.array([raw_prob[i,item] for i,item in enumerate(best_class)])
            self.prob_l = prob_l
           
            
            mask = np.logical_and( prob_l > self.score_thresh , best_class > 0) 
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            #best_class_mask = best_class[mask]
            keep = non_maximum_suppression(
                cls_bbox_l, self.nms_thresh, prob_l)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            #label.append((l - 1) * np.ones((len(keep),)))
            
            
            self.out = prob_l
            self.best_class = best_class
            self.mask = mask
            self.keep = keep
            self.raw_prob = raw_prob
            
            
            label.append(best_class[mask][keep] - 1 )

            prob.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        prob = np.concatenate(prob, axis=0).astype(np.float32)
        return bbox, label, prob
    model._suppress_each_box = _suppress_each_box.__get__(model)
    def predict_each_box(self, imgs):
            prepared_imgs = []
            sizes = []
            for img in imgs:
                size = img.shape[1:]
                img = self.prepare(img.astype(np.float32))
                prepared_imgs.append(img)
                sizes.append(size)

            bboxes = []
            labels = []
            scores = []
            for img, size in zip(prepared_imgs, sizes):
                with chainer.using_config('train', False), \
                        chainer.function.no_backprop_mode():
                    img_var = chainer.Variable(self.xp.asarray(img[None]))
                    scale = img_var.shape[3] / size[1]
                    roi_cls_locs, roi_scores, rois, _ = self.forward(
                        img_var, scales=[scale])
                # We are assuming that batch size is 1.
                roi_cls_loc = roi_cls_locs.array
                roi_score = roi_scores.array
                roi = rois / scale

                # Convert predictions to bounding boxes in image coordinates.
                # Bounding boxes are scaled to the scale of the input images.
                mean = self.xp.tile(self.xp.asarray(self.loc_normalize_mean),
                                    self.n_class)
                std = self.xp.tile(self.xp.asarray(self.loc_normalize_std),
                                   self.n_class)
                roi_cls_loc = (roi_cls_loc * std + mean).astype(np.float32)
                roi_cls_loc = roi_cls_loc.reshape((-1, self.n_class, 4))
                roi = self.xp.broadcast_to(roi[:, None], roi_cls_loc.shape)
                cls_bbox = loc2bbox(roi.reshape((-1, 4)),
                                    roi_cls_loc.reshape((-1, 4)))
                cls_bbox = cls_bbox.reshape((-1, self.n_class * 4))
                # clip bounding box
                cls_bbox[:, 0::2] = self.xp.clip(cls_bbox[:, 0::2], 0, size[0])
                cls_bbox[:, 1::2] = self.xp.clip(cls_bbox[:, 1::2], 0, size[1])
                #print(roi_score)
                prob = chainer.functions.softmax(roi_score).array

                raw_cls_bbox = cuda.to_cpu(cls_bbox)
                raw_prob = cuda.to_cpu(prob)

                bbox, label, prob = self._suppress_each_box(raw_cls_bbox, raw_prob)
                bboxes.append(bbox)
                labels.append(label)
                scores.append(prob)

            return bboxes, labels, scores
    model.predict_each_box = predict_each_box.__get__(model)

train07 = VOCBboxDataset(data_dir='auto',year='2007', split='trainval',use_difficult=True,return_difficult=False)
train12 = VOCBboxDataset(data_dir='auto',year='2012', split='trainval',use_difficult=True,return_difficult=False)

train  = ConcatenatedDataset(train07,train12)
test   = VOCBboxDataset(data_dir='auto',year='2007', split='test',use_difficult=True,return_difficult=True)

test_iter = chainer.iterators.SerialIterator(test,batch_size,repeat=False,shuffle=False)

print('image prepare')
if subset == 'left':
    train07_left = train07[:len(train07)//2]
    test_left    = test[:len(test)//2]
    print('load images')
    test_imgs_subset    = [item[0] for item in test_left   ]
    train07_imgs_subset = [item[0] for item in train07_left]

elif subset == 'right':
    train07_right = train07[len(train07)//2:]
    test_right    = test[len(test)//2:]
    print('load images')
    test_imgs_subset    = [item[0] for item in test_right   ]
    train07_imgs_subset = [item[0] for item in train07_right]


model.to_cpu()
model.to_gpu()
model.use_preset('evaluate')
model.score_thresh                       = config.model_score_thresh
model.nms_thresh                         = config.model_nms_thresh

if SHADOW_MODEL_TYPE == 'FR_vgg':

    model.rpn.proposal_layer.nms_thresh      = config.model_rpn_proposal_layer_nms_thresh
    model.rpn.proposal_layer.n_test_pre_nms  = config.model_rpn_proposal_layer_n_test_pre_nms
    model.rpn.proposal_layer.n_test_post_nms = config.model_rpn_proposal_layer_n_test_post_nms

lr_attack = 1e-5

input_size = 300
train_test_shuffle = 0 # 0: normal 1:shuffle 2: even odd shuffle 3 : odd even shuffle
SAVE_MODEL        = config.SAVE_MODEL       #True
TRANSFORM         = config.TRANSFORM        #True
PRETRAIN          = config.PRETRAIN         #False
NORMALIZE_CANVAS  = config.NORMALIZE_CANVAS #False
shuffle_boxes     = config.shuffle_boxes    #False
if shuffle_boxes:
    shuffle_sort  = config.shuffle_sort     #False
LOG_SCORE         = config.LOG_SCORE     #2 # 0 for not using logscore

if LOG_SCORE == 2:
    def logscore(a):
        r = -np.log2(1-a)
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
MAX_LEN           = config.MAX_LEN      
CANVAS_TYPE       = config.CANVAS_TYPE  
QUICK             = config.QUICK
SMALL_SET   = 0
EPOCHS      = 61
if True:

    if TRANSFORM:
        EPOCHS = 201 #61
    if SMALL_SET>0:
        a_train = a_train[len(a_train)//2-SMALL_SET:len(a_train)//2+SMALL_SET]
    if 'shallow' in ATTACK_MODEL:
        id_     += '_shallowCNN_'
    if PRETRAIN:
        id_     += '_pretrain_'
    if CANVAS_TYPE == 'original':
        id_ += '_'+CANVAS_TYPE+'_'
    if MAX_LEN <5000 :
        id_     += '_MAX_LEN_{}_'.format(MAX_LEN)

    if LOG_SCORE:
        if type( LOG_SCORE ) == bool:
            id_ += '_'+'LOG_SCORE'+'_'
        else:
            id_ += '_'+'LOG_SCORE_'
    if PREDICT_ONE_EACH_BOX:
         id_    += '_predictEach_'
ball_size = input_size*0.1

    

def generate_pointsets(imgs,model,num_logit_feature=1,max_len=MAX_LEN,input_size = 300,regard_in_set=True,shuffle_boxes = shuffle_boxes):
    
    size300_train07_dataset = []
    max_feat_len = 0
    min_feat_len = 5000
    mark = 0
    jump = 10
    with chainer.using_config('train', False),chainer.function.no_backprop_mode():
        feat_len_info = []
        while ( mark  < len(imgs ) ):
            tr_bboxes_list=[]
            tr_labels_list=[]
            tr_scores_list=[]
            tr_img_sizes= []
            x = []
            sizes = []
            if PREDICT_ONE_EACH_BOX:
                bboxes,labels,scores = model.predict_each_box(imgs[mark:mark+10])
            else:
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
                
            feat_len_info    +=  [len(item) for item in bbox_size300_list]
            
            max_feat_len_now = max([len(item) for item in bbox_size300_list])
            if max_feat_len < max_feat_len_now :
                max_feat_len = max_feat_len_now
                print('max feat len : {}'.format(max_feat_len))
            
            min_feat_len_now = min([len(item) for item in bbox_size300_list])
            if min_feat_len > min_feat_len_now:
                min_feat_len = min_feat_len_now
                      
                      
            if shuffle_boxes:
                bbox_size300_list = [ item[ np.random.permutation(len(item)) ] for item in bbox_size300_list]                      
            bbox_size300_list = np.array( [ item[:max_len] for item in bbox_size300_list   ] )
            size300_train07=np.array([np.pad(item,((0,max_len-len(item)),(0,0)),'constant') for item in bbox_size300_list])
            
            if regard_in_set:
                size300_train07_dataset += [(a,b) for a,b in zip(size300_train07, np.ones(len(size300_train07),dtype=np.float32)  )]
            else:
                size300_train07_dataset += [(a,b) for a,b in zip(size300_train07, np.zeros(len(size300_train07),dtype=np.float32) )]
            if mark +  10 > len(imgs):
                jump = len(imgs) - mark
            mark += jump
    return size300_train07_dataset


import numpy as np
import os
import copy
import numpy as np
import logging

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
import time
if True:
    load_dir  = SHADOW_MODEL_DIR
    chainer.serializers.load_npz(os.path.join(load_dir), model,)
    #print_('current iter : {}'.format(iter))

    size300_meta_train_dataset = []

    temp=generate_pointsets(train07_imgs_subset[:len(train07_imgs_subset)//2+1],model,regard_in_set=True,num_logit_feature=1)
    size300_meta_train_dataset += temp
    temp=generate_pointsets(test_imgs_subset[:len(test_imgs_subset)//2+1],model,regard_in_set=False,num_logit_feature=1)
    size300_meta_train_dataset += temp


    size300_meta_test_dataset = []
    size300_meta_test_dataset += generate_pointsets(train07_imgs_subset[len(train07_imgs_subset)//2+1:],model,regard_in_set=True,num_logit_feature=1)
    size300_meta_test_dataset += generate_pointsets(test_imgs_subset[len(test_imgs_subset)//2+1:],model,regard_in_set=False,num_logit_feature=1)

    a_train = np.array(size300_meta_train_dataset)
    a_test  = np.array(size300_meta_test_dataset)


    np.random.seed(777)
 

    def make_canvas_circle_data(a_data,ball_size=ball_size,QUICK=0):
        H,W = (input_size,input_size)
        y, x = np.mgrid[0:H, 0:W]
        (xc,yc) = (input_size//2,input_size//2)
        domain_radius =  ball_size/2
        domain_radius = int(domain_radius)
        sigma_x,sigma_y = (ball_size,ball_size)



        y, x = np.mgrid[0:H, 0:W]
        canvas_dataset_train=[]
        tt = time.time()
        for item in a_data:

            feats,label = item

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
            canvas_dataset_train.append([canvas,label])
        canvas_dataset_train = np.array(canvas_dataset_train)
        print_('time spent : {}'.format(time.time()-tt))
        return np.array(canvas_dataset_train)
    if PRETRAIN:
        canvas_dataset_train_pretrain=[]
        NUM_PRETRAIN_SET=500
        for _ in range(NUM_PRETRAIN_SET):
            domain_radius =  int(ball_size//2)
            label =   np.random.randint(2)
            num_feats = np.random.randint(1,5)
            canvas = np.zeros((input_size,input_size),np.float32)
            if label == 1:
                for i in range(num_feats):
                    x_c, y_c = np.random.randint(ball_size,high=input_size-ball_size,size=2)
                    #shots = np.random.randint(10,40)
                    if np.random.rand() > 0.9:
                        shots = np.random.randint(1,4+1)
                        for _ in range(shots):
                            x_tur,y_tur = np.random.randint(-4,4+1,size=2)
                            score  = np.random.rand()*0.1+0.9
                            x_tur ,y_tur = x_c+x_tur,y_c+y_tur
                            canvas[x_tur-domain_radius:x_tur+domain_radius+1,y_tur-domain_radius:y_tur+domain_radius+1] += score  
                        if NORMALIZE_CANVAS:
                            canvas = canvas/canvas.mean()
                        canvas_dataset_train_pretrain.append([canvas,label])
                    else:
                        shots = np.random.randint(1,8+1)
                        for _ in range(shots):
                            x_tur,y_tur = np.random.randint(-4,4+1,size=2)
                            score  = np.random.rand()*0.3
                            x_tur ,y_tur = x_c+x_tur,y_c+y_tur
                            canvas[x_tur-domain_radius:x_tur+domain_radius+1,y_tur-domain_radius:y_tur+domain_radius+1] += score  
                        if NORMALIZE_CANVAS:
                            canvas = canvas/canvas.mean()
                        canvas_dataset_train_pretrain.append([canvas,label])
            else:
                for i in range(num_feats):
                    x_c, y_c = np.random.randint(ball_size,high=input_size-ball_size,size=2)
                    #shots = np.random.randint(10,30)
                    #shots = np.random.randint(5,20)
                    shots = np.random.randint(1,12+1)
                    if np.random.rand() > 0.95:
                        shots = np.random.randint(1,4+1)
                        for _ in range(shots):
                            x_tur,y_tur = np.random.randint(-4,4+1,size=2)
                            score  = np.random.rand()*0.2+0.8
                            x_tur ,y_tur = x_c+x_tur,y_c+y_tur
                            canvas[x_tur-domain_radius:x_tur+domain_radius+1,y_tur-domain_radius:y_tur+domain_radius+1] += score  
                        if NORMALIZE_CANVAS:
                            canvas = canvas/canvas.mean()
                        canvas_dataset_train_pretrain.append([canvas,label])
                    else:
                        shots = np.random.randint(1,8+1)

                        for _ in range(shots):
                            x_tur,y_tur = np.random.randint(-10,10+1,size=2)
                            score  = np.random.rand()*0.3
                            x_tur ,y_tur = x_c+x_tur,y_c+y_tur
                            canvas[x_tur-domain_radius:x_tur+domain_radius+1,y_tur-domain_radius:y_tur+domain_radius+1] += score  
                        if NORMALIZE_CANVAS:
                            canvas = canvas/canvas.mean()
                        canvas_dataset_train_pretrain.append([canvas,label])
        canvas_dataset_pretrain =[list(item) for item in canvas_dataset_train_pretrain]
        
        
    canvas_dataset_train=make_canvas_circle_data(a_train,QUICK=QUICK)
    canvas_dataset_test =make_canvas_circle_data( a_test,QUICK=QUICK)
    canvas_dataset_train =[list(item) for item in canvas_dataset_train]
    canvas_dataset_test =[list(item) for item in canvas_dataset_test]

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
    train = canvas_dataset(canvas_dataset_train,transform=TRANSFORM)
    test  = canvas_dataset(canvas_dataset_test , transform=False   )

    train_loader = torch.utils.data.DataLoader(train, batch_size=8,shuffle=True )
    test_loader  = torch.utils.data.DataLoader(test , batch_size=8,shuffle=False)
    
    if PRETRAIN:
        pretrain = canvas_dataset(canvas_dataset_pretrain,transform=TRANSFORM)
        pretrain_loader = torch.utils.data.DataLoader(pretrain, batch_size=8,shuffle=True )
    
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
    net.train();
    step_start = 0  
    #weight_decay = 0.0005
    weight_decay = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_attack, weight_decay=weight_decay)
    cross_entropy_loss = nn.CrossEntropyLoss()
    train_loss = 0
    train_loss_batch = 0
    def print_summary(summary):
            
            print_(' ')
            print_(summary['tag'])
            print_('epoch: {}      '.format(summary['epoch']     ))
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
            summary['epoch'     ] = epoch 
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
    if PRETRAIN:
        print('pretrain started')
        best_acc=0
        for epoch in range(5):
            train_loss_batch = 0
            net.train()
            for batch in pretrain_loader:
                images = batch[0]
                labels = batch[1]
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                labels_pred = net.forward(images)
                # backward
                loss = cross_entropy_loss(labels_pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss = loss.data.item()
                train_loss_batch += train_loss 
            print_('train loss : {}'.format(train_loss_batch))
          
            if epoch % 1 == 0:
                net.eval()
                test_acc ,test_summary =get_acc(test ,test_loader ,net,tag='test' ,print_all=False)
                if test_acc >best_acc:
                   
                    best_acc =test_acc
                    print_('###################')
                    print_('best test acc : {}'.format(best_acc))
                    _ , best_test_summary = get_acc(test ,test_loader ,net,tag='test'  ,print_all=True)

                    print_('###################')

            if epoch % 10 == 0:
                get_acc(train,train_loader,net,tag='train')
        print('pretrain ended')
    
    
    best_acc=0
    for epoch in range(EPOCHS):
        train_loss_batch = 0
        net.train()
        for batch in train_loader:
            images = batch[0]
            labels = batch[1]
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            labels_pred = net.forward(images)
            loss = cross_entropy_loss(labels_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = loss.data.item()
            train_loss_batch += train_loss 
        print_('train loss : {}'.format(train_loss_batch))
     
        net.eval()
     
        test_acc ,test_summary =get_acc(test ,test_loader ,net,tag='test' ,print_all=False)
        if test_acc >best_acc:
            if SAVE_MODEL:
                if TRANSFORM:
                    torch.save(net.state_dict(),'mia_{}_{}.npy'.format(id_,epoch))
                else:
                    torch.save(net.state_dict(),'mia_noAUg_{}_{}.npy'.format(id_,epoch))
            best_acc =test_acc
            best_test_summary  = test_summary
            print_('###################')
            print_('best test acc : {}'.format(best_acc))
            get_acc(test ,test_loader ,net,tag='test'  ,print_all=True)
            print_('###################')

        if epoch % 10 == 0:
            get_acc(train,train_loader,net,tag='train')
        #train_acc.append(correct_num/ len(dataset_train))
    # At the end
    get_acc(test ,test_loader ,net,tag='test' ,print_all=True)
    
    #print_summary(best_valid_summary)
    print_summary(best_test_summary)

  
    del net
   
   










