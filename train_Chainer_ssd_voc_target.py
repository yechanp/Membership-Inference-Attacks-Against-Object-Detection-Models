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
import os 
import sys
from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import port DetectionVOCEvaluatorVOCBboxDataset
from chainercv.extensions im
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import transforms
from ssd_transform import _Transform
argv = sys.argv
continuous = False


chainer.config.cv_resize_backend = "cv2"

if len(argv) >1 :
   gpu_id = int(argv[1])
   if len(argv)>2:
      continuous = bool(argv[2])
else:
   gpu_id = 0
print('gpu_id is {}'.format(gpu_id))
SAVE_PATH = 'ssd300_model_vocall_trval_lrdrop_target.npz'
print('save path is {}'.format(SAVE_PATH))
iters  = 800000+1
batch_size  = 8


model = SSD300(
            n_fg_class=21,
            pretrained_model='imagenet')
model.to_gpu(gpu_id)


train07 = VOCBboxDataset(data_dir='auto',year='2007', split='trainval',use_difficult=True,return_difficult=False)
train12 = VOCBboxDataset(data_dir='auto',year='2012', split='trainval',use_difficult=True,return_difficult=False)

train07_left = train07[:len(train07)//2]
train12_left = train12[:len(train12)//2]

train_left = ConcatenatedDataset(train07_left,train12_left)
train = TransformDataset(train_left,_Transform(model.coder,model.insize,model.mean))
train_iter = chainer.iterators.SerialIterator(train,batch_size,repeat=True,shuffle=True)

test = VOCBboxDataset(data_dir='auto',year='2007', split='test',use_difficult=True,return_difficult=True)
test_iter = chainer.iterators.SerialIterator(test,batch_size,repeat=False,shuffle=False)



class MultiboxTrainChain(chainer.Chain):
    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k
    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)
        return loss
train_chain = MultiboxTrainChain(model)




optimizer = chainer.optimizers.MomentumSGD(1e-3)
optimizer.setup(train_chain)
for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))




evaluator = DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=voc_bbox_label_names,)

class save_model(training.Extension): #ver 1.2
    def __init__(self,model,save_name,before_iter=0,saved_dir='saved_model/',save_after=0):
        self.model = model
        self.save_name=save_name
        self.saved_dir=saved_dir
        self.before_iter = before_iter
        self.save_after= save_after
    def __call__(self,trainer):
        curr_iter = trainer.updater.iteration+self.before_iter
        if curr_iter>self.save_after:
            chainer.serializers.save_npz(self.saved_dir+self.save_name[:-4]+'_'+str(curr_iter)+'.npz', model,)


steps = [200000 , 400000]
lr_trigger= triggers.ManualScheduleTrigger(steps, 'iteration')


updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=gpu_id)
trainer = training.Trainer(
        updater, (iters, 'iteration'), 'ssd_result')
trainer.extend(extensions.ExponentialShift('lr', 0.1), trigger=lr_trigger)
#trainer.extend(evaluator,trigger=(50000, 'iteration'))
trainer.extend(training.extensions.LogReport(log_name='ssd_report'+SAVE_PATH,trigger=(1000, 'iteration')))
trainer.extend(extensions.observe_lr(), trigger=(1000, 'iteration'))
trainer.extend(training.extensions.PrintReport(['iteration','lr' , 'main/loss', 'main/loss/loc','main/loss/conf']))
trainer.extend(save_model(model,SAVE_PATH,save_after=0),trigger=(50000,'iteration'))

if continuous:
   chainer.serializers.load_npz(os.path.join(SAVE_PATH), model,)
trainer.run()
chainer.serializers.save_npz(os.path.join(SAVE_PATH), model,)


# In[12]:



