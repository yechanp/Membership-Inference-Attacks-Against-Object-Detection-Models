class Config():
    gpu_id     = 1
    batch_size = 1 
    
    
    LOAD_DIR_META = 'saved_model/fastercnn_model_right_200000.npz'
    id_           = 'FR_200k_normal_right_rpnNMS070'
    SHADOW_MODEL  = 'FR_vgg'  # FR_vgg for Faster R CNN
    ATTACK_MODEL  = 'alex'  # 'shallow'

    
    PREDICT_ONE_EACH_BOX = False
    SAVE_MODEL        = True
    TRANSFORM         = True
    PRETRAIN          = False
    NORMALIZE_CANVAS  = False
    shuffle_boxes     = False
    if shuffle_boxes:
        shuffle_sort  = False
    LOG_SCORE         = 2 # 0 for not using logscore
    MAX_LEN           = 6000# 5000
    CANVAS_TYPE = 'original' #CANVAS_TYPE = 'original'
    QUICK       = 0
    
    if 'ssd' in SHADOW_MODEL:
        pass
    
    if 'FR' in SHADOW_MODEL:
    
        MIN_SIZE=600 # 600 by default
        MAX_SIZE=800 # 800 by default
        
        model_score_thresh= 0.01
        model_nms_thresh  = 1.00
        model_rpn_proposal_layer_nms_thresh      = 0.70    #0.8  # 0.7
        model_rpn_proposal_layer_n_test_pre_nms  = 6000    # 6000
        model_rpn_proposal_layer_n_test_post_nms = 6000    # 300
