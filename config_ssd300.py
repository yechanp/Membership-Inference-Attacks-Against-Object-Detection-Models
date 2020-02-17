class Config():
    gpu_id     = 1
    batch_size = 8 #1 
    
    
    SHADOW_MODEL_DIR = 'saved_model/ssd300_model_vocall_trval_lrdrop_right_500000.npz'
    TARGET_MODEL_DIR = 'saved_model/ssd300_model_vocall_trval_left_100000.npz'
    ATTACK_MODEL_DIR = 'attackmodel'
    id_           = 'ssd_500k_normal_right'#'FR_200k_normal_right_rpnNMS070'
    SHADOW_MODEL_TYPE  = 'ssd300_vgg' #  ['ssd300_vgg'  , 'FR_vgg' ]  
    TARGET_MODEL_TYPE  = 'ssd300_vgg' #  ['ssd300_vgg'  , 'FR_vgg' ]  

    ATTACK_MODEL  = 'alex'  # ['alex, 'shallow']
    
    PREDICT_ONE_EACH_BOX = False
    SAVE_MODEL        = True
    TRANSFORM         = True
    PRETRAIN          = False
    NORMALIZE_CANVAS  = False
    shuffle_boxes     = False
    if shuffle_boxes:
        shuffle_sort  = False
    LOG_SCORE         = 2 # 0 for not using logscore
    MAX_LEN           = 5000# [300,5000,6000]
    CANVAS_TYPE = 'original' #CANVAS_TYPE  = ['original' , 'uniform']
    QUICK       = 0
    
    if 'ssd' in SHADOW_MODEL_TYPE:
        model_score_thresh= 0.01
        model_nms_thresh  = 1.00
        
        START   = 50000
        END     = 500000 #500000
        INTERVAL= 50000
        pass
    
    if 'FR' in SHADOW_MODEL_TYPE:
    
        MIN_SIZE=600 # 600 by default
        MAX_SIZE=800 # 800 by default
        
           
        START   =25000
        END     =400000
        INTERVAL=25000
        
        model_score_thresh= 0.01
        model_nms_thresh  = 1.00
        model_rpn_proposal_layer_nms_thresh      = 0.70    #0.8  # 0.7
        model_rpn_proposal_layer_n_test_pre_nms  = 6000    # 6000
        model_rpn_proposal_layer_n_test_post_nms = 6000    # 300
