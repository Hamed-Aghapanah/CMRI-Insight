"""
Here is a dice loss for keras which is smoothed to approximate a linear (L1) loss.
It ranges from 1 to 0 (no error), and returns results similar to binary crossentropy
"""

# define custom loss and metric functions 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

tf.executing_eagerly()
# tf.enable_eager_execution()
# import tensorflow as tf
# tf.enable_eager_execution()

#  for contour loss if cant find contour
diss_constant=1


w_loss = [0, 1, 0, 0]  #['MSE' 'dice' 'CrossEntropy'  'MAE']
# =============================================================================
# Section 1  :  acc_coef
# =============================================================================
def acc_coef (y_true, y_pred):
    loss_val = 0
    if w_loss[0] != 0:#mse
        temp =np.square(y_true-y_pred)
        temp = K.mean(temp)
        loss_val = loss_val +  w_loss[0]*temp
    if w_loss[1] != 0:# Dice
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        smooth = 1
        temp = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
        loss_val = loss_val + w_loss[1]*temp
    if w_loss[2] != 0: # 'CrossEntropy':
        if y_true == 1:
            temp =-K.log(y_pred)
        else:
            temp= -K.log(1 - y_pred)
        loss_val = loss_val + w_loss[2]*temp        
    if w_loss[3] != 0: #'MAE':
         temp = K.sum(y_true*K.log(y_pred),axis=0)
         loss_val = loss_val + temp*w_loss[1]
    return loss_val
def loss_coef(y_true, y_pred):
    return 1-acc_coef(y_true, y_pred)
# =============================================================================
# Section 2  :  MultiClass Dice 
# =============================================================================
def dice_coef_multi(y_true, y_pred, smooth=1, nclass=2):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    dice=np.zeros([nclass,1])     
    y_true = y_true+1
    y_pred =  y_pred+1
       
    for i in range(1, nclass+1):
        i_y_pred = K.y_pred.copy()
        i_y_true = y_true.copy()
        K.place(i_y_true,i_y_true!=i, [0])
        np.place(i_y_true,i_y_true==i, [1])
        np.place(i_y_pred,i_y_pred!=i, [0])
        np.place(i_y_pred,i_y_pred==i, [1])        
        intersection = K.sum(K.abs(i_y_true * i_y_pred), axis=-1)
        dice[i] = (2. * intersection + smooth) / (K.sum(K.square(i_y_true),-1)\
            + K.sum(K.square(i_y_pred),-1) + smooth)
    return dice

def dice_coef_loss_multi(y_true, y_pred):
    return 1-dice_coef_multi(y_true, y_pred)
# =============================================================================
# Section 3  :  Dice Coefficient
# =============================================================================    
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    try:
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        dicee=(2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    except:
        
        try:
            y_pred = y_pred.numpy()
            y_true = y_true.numpy()
        except:1    
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    under =(K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    dicee=(2. * intersection + smooth) / under
    
    
    # saed
    # dicee=(2. * np.mean(intersection) + smooth) / np.mean(under)
    # print('dicee = ',dicee)    
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    dicee = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

    if  K.sum(K.abs(y_true))==0 or  K.sum(K.abs(y_pred))==0:
        dicee=1
    return dicee

def dice_coef_np(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    try:
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        dicee=(2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    except:
        
        try:
            y_pred = y_pred.numpy()
            y_true = y_true.numpy()
        except:1    
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    under =(K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    dicee=(2. * intersection + smooth) / under
    
    
    # saed
    dicee=(2. * np.mean(intersection) + smooth) / np.mean(under)
    # print('dicee = ',dicee)    
        # intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        # temp = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    return dicee

def dice_coef_loss(y_true, y_pred):
    # print(type (1-dice_coef(y_true, y_pred)))
    return 1-dice_coef(y_true, y_pred)
# =============================================================================
# Section 4  :  tversky
# =============================================================================        
def tversky(y_true, y_pred,smooth=1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)
# =============================================================================
# Section 5  : jaccard_coef_logloss
# =============================================================================  
def jaccard_coef_logloss(y_true, y_pred, smooth=1e-10):
    """ Loss function based on jaccard coefficient.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    keras tensor
        tensor containing negative logarithm of jaccard coefficient.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    falsepos = K.sum(y_pred) - truepos
    falseneg = K.sum(y_true) - truepos
    jaccard = (truepos + smooth) / (smooth + truepos + falseneg + falsepos)
    return -K.log(jaccard + smooth)
# =============================================================================
# Section 6  :  losss
# =============================================================================  
def losss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))
        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))
    return tversky_loss(y_true, y_pred) + dice_loss(y_true, y_pred)
# =============================================================================
# Section 7  :  directed hausdorff  distance  (Weighted_Hausdorff_loss)
# =============================================================================        
def hausdorff(y_true, y_pred ):
    from scipy.spatial.distance import directed_hausdorff
    import numpy as np
    u = K.flatten(y_true)
    v = K.flatten(y_pred)
    # u=np.array(u)
    # v=np.array(v)
    # true_pos = K.sum(y_true_pos * y_pred_pos)
    # false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    # false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    # alpha = 0.7
    H = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return H



def tf_repeat(tensor, repeats):
    """
    Args:
    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input
    Returns:
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor
def Weighted_Hausdorff_loss(y_true, y_pred):
    w=128;            h=128
    
    import math
    import numpy as np
    import tensorflow as tf
    from sklearn.utils.extmath import cartesian
    
    resized_height = 144  
    resized_width  = 256
    max_dist = math.sqrt(resized_height**2 + resized_width**2)
    n_pixels = resized_height * resized_width
    all_img_locations = tf.convert_to_tensor(cartesian([np.arange(resized_height), np.arange(resized_width)]),
                                                       tf.float32)



    # https://arxiv.org/pdf/1806.07564.pdf
    #prob_map_b - y_pred
    #gt_b - y_true
    # https://www.catalyzex.com/paper/arxiv:1806.07564/code
    terms_1 = []
    terms_2 = []
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.squeeze(y_pred, axis=-1)
    batch_size=len(y_true)
#     y_true = tf.reduce_mean(y_true, axis=-1)
#     y_pred = tf.reduce_mean(y_pred, axis=-1)
    for b in range(batch_size):
        gt_b = y_true[b]
        prob_map_b = y_pred[b]
        # Pairwise distances between all possible locations and the GTed locations
        n_gt_pts = tf.reduce_sum(gt_b)
        gt_b = tf.where(tf.cast(gt_b, tf.bool))
        gt_b = tf.cast(gt_b, tf.float32)
        d_matrix = tf.sqrt(tf.maximum(tf.reshape(tf.reduce_sum(gt_b*gt_b, axis=1), (-1, 1)) + tf.reduce_sum(all_img_locations*all_img_locations, axis=1)-2*(tf.matmul(gt_b, tf.transpose(all_img_locations))), 0.0))
        d_matrix = tf.transpose(d_matrix)
        # Reshape probability map as a long column vector,
        # and prepare it for multiplication
        p = tf.reshape(prob_map_b, (n_pixels, 1))
        n_est_pts = tf.reduce_sum(p)
        p_replicated = tf_repeat(tf.reshape(p, (-1, 1)), [1, n_gt_pts])
        eps = 1e-6
        alpha = 4
        # Weighted Hausdorff Distance
        term_1 = (1 / (n_est_pts + eps)) * tf.reduce_sum(p * tf.reshape(tf.reduce_min(d_matrix, axis=1), (-1, 1)))
        d_div_p = tf.reduce_min((d_matrix + eps) / (p_replicated**alpha + eps / max_dist), axis=0)
        d_div_p = tf.clip_by_value(d_div_p, 0, max_dist)
        term_2 = tf.reduce_mean(d_div_p, axis=0)
        terms_1.append(term_1)
        terms_2.append(term_2)
    terms_1 = tf.stack(terms_1)
    terms_2 = tf.stack(terms_2)
    terms_1 = tf.Print(tf.reduce_mean(terms_1), [tf.reduce_mean(terms_1)], "term 1")
    terms_2 = tf.Print(tf.reduce_mean(terms_2), [tf.reduce_mean(terms_2)], "term 2")
    res = terms_1 + terms_2
    return res

    all_img_locations = tf.convert_to_tensor(cartesian([np.arange(w),
                                               np.arange(h)]), dtype=tf.float32)
    max_dist = math.sqrt(w ** 2 + h ** 2)

    def hausdorff_loss(y_true, y_pred):
        def loss(y_true, y_pred):
            w=128;            h=128
            eps = 1e-6
            y_true = K.reshape(y_true, [w, h])
            gt_points = K.cast(tf.where(y_true > 0.5), dtype=tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
            y_pred = K.flatten(y_pred)
            p = y_pred
            p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p, axis=-1), 
                                                num_gt_points))
            d_matrix = cdist(all_img_locations, gt_points)
            num_est_pts = tf.reduce_sum(p)
            term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

            d_div_p = K.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)
            d_div_p = K.clip(d_div_p, 0, max_dist)
            term_2 = K.mean(d_div_p, axis=0)

            return term_1 + term_2

        batched_losses = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (y_true, y_pred),
                                   dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))

    return hausdorff_loss


# =============================================================================
# Section 7  :  confusion_m 
# =============================================================================        
def confusion_matrix_saed(y1,yy1,class_no):
    y1_index=np.where (y1>=class_no);y1[y1_index] = class_no-1
    y1_index=np.where (y1<0);y1[y1_index] = 0
    y1_index=np.where (yy1>=class_no);yy1[y1_index] = class_no-1
    y1_index=np.where (yy1<0);yy1[y1_index] = 0
    confusion_m=np.zeros ([class_no,class_no])
    # print (confusion_m)
    for i in range(len (y1)):
        # print (int(y1[i]),int(yy1[i]))
        # print (confusion_m)
        confusion_m [int(y1[i]),int(yy1[i])] = confusion_m[int(y1[i]),int(yy1[i])] +1
    return confusion_m

def Precision(y_true, y_pred ):
    th=0.5
    a= np.where (y_true>th);y_true=0*y_true;y_true[a]=1
    a= np.where (y_pred>th);y_pred=0*y_pred;y_pred[a]=1
    from sklearn.metrics import confusion_matrix
    if np.size (np.shape(y_pred))==3:
        y1=np.array (y_pred[:,:,0].flatten())
    if np.size (np.shape(y_pred))==2:
        y1=np.array (y_pred[:,:].flatten())
    # y1=np.array (y_pred[:,:,0].flatten())
    yy1=np.array ( y_true .flatten())
    confusion_m = confusion_matrix_saed(y1,yy1,2 )
    epsilon = 0.001
    TP=confusion_m[0,0] 
    TN=confusion_m[1,1] 
    FN=confusion_m[0,1] 
    FP=confusion_m[1,0] 
    
    if FP  ==0:
        Precision=1
    if not FP==0:
        Precision  = TP / (FP + TP)
    # print('Precision',TP,FP,Precision)
    # precision =np.mean( TP /  (TP +FN))
    # Recall  = TP / (FN + TP) 
    # recall = np.mean(tp / tp_and_fn)
    # Accuracy_Score = (TP + TN)/ (TP + FN + TN + FP) 
    # F1_Score = 2* Precision  * Recall / (Precision  + Recall) 
    return Precision

def Recall(y_true, y_pred ):
    th=0.5
    a= np.where (y_true>th);y_true=0*y_true;y_true[a]=1
    a= np.where (y_pred>th);y_pred=0*y_pred;y_pred[a]=1
    from sklearn.metrics import confusion_matrix
    # y1=np.array (y_pred[:,:,0].flatten())
    
    if np.size (np.shape(y_pred))==3:
        y1=np.array (y_pred[:,:,0].flatten())
    if np.size (np.shape(y_pred))==2:
        y1=np.array (y_pred[:,:].flatten())
    yy1=np.array ( y_true .flatten())
    confusion_m = confusion_matrix_saed(y1,yy1,2 )
    epsilon = 0.001
    TP=confusion_m[0,0] 
    TN=confusion_m[1,1] 
    FN=confusion_m[0,1] 
    FP=confusion_m[1,0] 
    # Precision  = TP / (FP + TP)
    # precision =np.mean( TP /  (TP +FN))
    
    if FN ==0:
        Recall=1
    if not FN==0:
        Recall  = TP / (FN + TP)

    # recall = np.mean(tp / tp_and_fn)
    # Accuracy_Score = (TP + TN)/ (TP + FN + TN + FP) 
    # F1_Score = 2* Precision  * Recall / (Precision  + Recall) 
    return Recall

def Accuracy_Score(y_true, y_pred ):
    th=0.5
    a= np.where (y_true>th);y_true=0*y_true;y_true[a]=1
    a= np.where (y_pred>th);y_pred=0*y_pred;y_pred[a]=1
    from sklearn.metrics import confusion_matrix
    
    if np.size (np.shape(y_pred))==3:
        y1=np.array (y_pred[:,:,0].flatten())
    if np.size (np.shape(y_pred))==2:
        y1=np.array (y_pred[:,:].flatten())
    yy1=np.array ( y_true .flatten())
    confusion_m = confusion_matrix_saed(y1,yy1,2 )
    epsilon = 0.001
    TP=confusion_m[0,0] 
    TN=confusion_m[1,1] 
    FN=confusion_m[0,1] 
    FP=confusion_m[1,0] 
    # Precision  = TP / (FP + TP)
    # precision =np.mean( TP /  (TP +FN))
    # Recall  = TP / (FN + TP) 
    # recall = np.mean(tp / tp_and_fn)
    
    if TP + FN + TN + FP ==0:
        Accuracy_Score=1
    if not TP + FN + TN + FP ==0:
        Accuracy_Score = (TP + TN)/ (TP + FN + TN + FP) 
    # F1_Score = 2* Precision  * Recall / (Precision  + Recall) 
    return Accuracy_Score


def Specificity(y_true, y_pred ):
    th=0.5
    a= np.where (y_true>th);y_true=0*y_true;y_true[a]=1
    a= np.where (y_pred>th);y_pred=0*y_pred;y_pred[a]=1
    from sklearn.metrics import confusion_matrix
    # y1=np.array (y_pred[:,:,0].flatten())
    if np.size (np.shape(y_pred))==3:
        y1=np.array (y_pred[:,:,0].flatten())
    if np.size (np.shape(y_pred))==2:
        y1=np.array (y_pred[:,:].flatten())
    yy1=np.array ( y_true .flatten())
    confusion_m = confusion_matrix_saed(y1,yy1,2 )
    epsilon = 0.001
    TP=confusion_m[0,0] 
    TN=confusion_m[1,1] 
    FN=confusion_m[0,1] 
    FP=confusion_m[1,0] 
    
    # Precision  = TP / (FP + TP)
    # precision =np.mean( TP /  (TP +FN))
    # Recall  = TP / (FN + TP) 
    # recall = np.mean(tp / tp_and_fn)
    # Accuracy_Score = (TP + TN)/ (TP + FN + TN + FP) 
    
    if FP==0:
        Specificity1=1
    if not FP==0:
        Specificity1=TN/(TN+FP)
    # print(TN,TN,FP)
    # print(confusion_m)
    # print(Specificity1)
    # F1_Score = 2* Precision  * Recall / (Precision  + Recall) 
    return Specificity1


def Sensitivity(y_true, y_pred ):
    th=0.5
    a= np.where (y_true>th);y_true=0*y_true;y_true[a]=1
    a= np.where (y_pred>th);y_pred=0*y_pred;y_pred[a]=1
    from sklearn.metrics import confusion_matrix
    # y1=np.array (y_pred[:,:,0].flatten())
    if np.size (np.shape(y_pred))==3:
        y1=np.array (y_pred[:,:,0].flatten())
    if np.size (np.shape(y_pred))==2:
        y1=np.array (y_pred[:,:].flatten())
    yy1=np.array ( y_true .flatten())
    confusion_m = confusion_matrix_saed(y1,yy1,2 )
    epsilon = 0.001
    TP=confusion_m[0,0] +epsilon
    TN=confusion_m[1,1] +epsilon
    FN=confusion_m[0,1] +epsilon
    FP=confusion_m[1,0] +epsilon
    # Precision  = TP / (FP + TP)
    # precision =np.mean( TP /  (TP +FN))
    # Recall  = TP / (FN + TP) 
    # recall = np.mean(tp / tp_and_fn)
    # Accuracy_Score = (TP + TN)/ (TP + FN + TN + FP) 
    # Specificity1=TN/(TN+FP)
    
    if FN==0:
        Sensitivity1=1
    if not FN==0:
        Sensitivity1 =TP/(TP+FN)
    # print('sen',TP ,TP,FN )
    # print(confusion_m)
    # F1_Score = 2* Precision  * Recall / (Precision  + Recall) 
    return Sensitivity1


def F1_Score(y_true, y_pred ):
    # th=0.5
    # a= np.where (y_true>th);y_true=0*y_true;y_true[a]=1
    # a= np.where (y_pred>th);y_pred=0*y_pred;y_pred[a]=1
    # from sklearn.metrics import confusion_matrix
    # # y1=np.array (y_pred[:,:,0].flatten())
    
    
    # if np.size (np.shape(y_pred))==3:
    #     y1=np.array (y_pred[:,:,0].flatten())
    # if np.size (np.shape(y_pred))==2:
    #     y1=np.array (y_pred[:,:].flatten())
        
    # yy1=np.array ( y_true .flatten())
    # confusion_m = confusion_matrix_saed(y1,yy1,2 )
    # # print(np.shape(confusion_m))
    # # print((confusion_m))
    # epsilon = 0.001
    # TP=confusion_m[0,0] +epsilon
    # TN=confusion_m[1,1] +epsilon
    # FN=confusion_m[0,1] +epsilon
    # FP=confusion_m[1,0] +epsilon
    # # print(TP)
    # # print(111)
    # Precision  = TP / (FP + TP)
    # # precision =np.mean( TP /  (TP +FN))
    # Recall  = TP / (FN + TP) 
    # # recall = np.mean(tp / tp_and_fn)
    # # Accuracy_Score = (TP + TN)/ (TP + FN + TN + FP) 
    Precision1 =Precision(y_true, y_pred)
    Recall1=    Recall(y_true, y_pred)
    
    # print(Precision1,Recall1)
    
    # print(20*'saed')
    # F1_Score = 2* Precision1  * Recal1l / (Precision1  + Recall1) 
    if Precision1  + Recall1 ==0:
        F1_Score=1
    if not Precision1  + Recall1 ==0:
        F1_Score = 2* Precision1  * Recall1 / (Precision1  + Recall1) 
    return F1_Score

def F1_Score1(y_true, y_pred ):
    th=0.5
    a= np.where (y_true>th);y_true=0*y_true;y_true[a]=1
    a= np.where (y_pred>th);y_pred=0*y_pred;y_pred[a]=1
    from sklearn.metrics import confusion_matrix
    y1=np.array (y_pred.flatten())
    yy1=np.array ( y_true .flatten())
    confusion_m = confusion_matrix_saed(y1,yy1,2 )
    # print(np.shape(confusion_m))
    # print((confusion_m))
    epsilon = 0.01
    TP=confusion_m[0,0] +epsilon
    TN=confusion_m[1,1] +epsilon
    FN=confusion_m[0,1] +epsilon
    FP=confusion_m[1,0] +epsilon
    # print(TP)
    # print(111)
    Precision  = TP / (FP + TP)
    # precision =np.mean( TP /  (TP +FN))
    Recall  = TP / (FN + TP) 
    # recall = np.mean(tp / tp_and_fn)
    # Accuracy_Score = (TP + TN)/ (TP + FN + TN + FP) 
    F1_Score = 2* Precision  * Recall / (Precision  + Recall) 
    return F1_Score


   
        
        

def iou( predictions, labels, smooth=1,th=0.5):
    
    for innnnn in range(len (predictions)):
        idx = np.where (predictions[innnnn] <th);predictions[innnnn][idx]=0
        idx = np.where (predictions[innnnn] >=th);predictions [innnnn][idx]=1
    for innnnn in range(len (labels)):
        idx = np.where (labels[innnnn] <th);labels[innnnn][idx]=0
        idx = np.where (labels[innnnn] >=th);labels [innnnn][idx]=1    
        
    iou = []
    # print('labels.shape[-1]',labels.shape[-1])
    for i in range(labels.shape[-1]):
      try: label_1D = K.flatten(labels[:, :, i])
      except:label_1D = K.flatten(labels )
      
      try: pred_1D  = K.flatten(predictions[:, :, i])
      except:pred_1D  = K.flatten(predictions )
                                  
      # label_1D = K.flatten(labels[:, :, i])
      # pred_1D  = K.flatten(predictions[:, :, i])
      intersection = K.sum(label_1D * pred_1D)
      union = K.sum(label_1D) + K.sum(pred_1D) - K.sum((label_1D * pred_1D))
      # print('intersection',intersection,'union',union)
      try:
          iou_coff = (intersection + smooth) / (union + smooth)
          iou.append( float(iou_coff))
      except:
          iou_coff = 0
          iou.append( float(iou_coff))
    return iou
# =============================================================================
#     
# =============================================================================
def mask2contour (mask,n=100):
    img = np.uint8 (mask)
    import cv2
    img[np.where (img>0.5)] =1
    img[np.where (img<0.5)] =0
    # plt.imshow(img,cmap='gray')
    # contours, _ = cv2.findContours(img)
    # idx = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1][0]
    idx = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) 
    x=[];y=[];
    try:
        a= len(idx[0][0] )
        for i in range(a):
            x.append(idx[0][0][i][0][0])
            y.append(idx[0][0][i][0][1])
    except:2 
    return x,y
# =============================================================================
# 
# =============================================================================
def mask2(predictions, labels,n=100):
    x1=[];y1=[];x2=[];y2=[];
    predictions[np.where (predictions>0.5)] =1
    predictions[np.where (predictions<0.5)] =0
    
    labels[np.where (labels>0.5)] =1
    labels[np.where (labels<0.5)] =0
    
    try:    x1,y1 = mask2contour (predictions)
    except:1
    try:x2,y2 = mask2contour (labels)
    except:1
    from saed import roundd
    import scipy
    
    try:x1  = scipy.signal.resample (x1,n)
    except:1
    try:x2  = scipy.signal.resample (x2,n)
    except:1
    try:y1  = scipy.signal.resample (y1,n)
    except:1
    try:y2  = scipy.signal.resample (y2,n)
    except:1
    
    return x1,y1,x2,y2
# =============================================================================
# New  loss function  curvature   regular  MAE  DSSIM  iou
# =============================================================================
def contour_loss_function  (predictions, labels,n=100):
    x1,y1 = mask2contour (predictions)
    x2,y2 = mask2contour (labels)
    scape=False
    if len(x1)==0 and not len(x2)==0:scape=True;disss=diss_constant*len(x2)
    if len(x2)==0 and not len(x1)==0:scape=True;disss=diss_constant*len(x1)
    if len(x2)==0 and len(x1)==0:scape=True;disss=0
    if not scape:
        from saed import roundd
        import scipy
        
        x1  = scipy.signal.resample (x1,n)
        x2  = scipy.signal.resample (x2,n)
        y1  = scipy.signal.resample (y1,n)
        y2  = scipy.signal.resample (y2,n)
            
                
        disss=[]
        for i in range(n):
            d=[]
            for j in range(n):
                dis= (x1[i] - x2[j])**2 + (y1[i] - y2[j])**2
                d.append(dis)
            i1 = np.argmin(d)
            mm=roundd(np.min(d),3)
            disss.append(mm)
        disss=np.mean(disss)/n   
    return disss




def curvature_loss_function(predictions, labels,n=100):
    # print(np.shape(predictions))
    
    H=np.zeros([3,3])
    H[0,0]=-1;    H[2,0]=-1;    H[0,2]=-1;    H[2,0]=-1;
    H[1,0]=5;    H[1,2]=5;    H[0,1]=5;    H[2,1]=5;
    H[1,1]=16;
    from scipy.ndimage import convolve
    
    # T=labels
    P_C = convolve(predictions, H)   
    T_C = convolve(labels, H)   
    l_cur =0
    epsilon =10**-6
    T   =labels.flatten()
    T_C = T_C.flatten()
    P_C=P_C.flatten()
    # plt.figure()
    # plt.subplot(131);    plt.imshow(predictions)
    # plt.subplot(132);    plt.imshow(labels)
    # plt.subplot(133)
    # plt.plot(T,'r*')
    # plt.plot(T_C,'g*')
    # plt.plot(P_C,'b*')
    # ss
    for i in range(len ( T_C)):
        if not T_C[i]==0 and not P_C[i] == T_C[i]:
            l_cur=l_cur+np.abs ((P_C[i] *T[i]) /(T_C[i]  ))
            # print(P_C[i] ,T[i] , T_C[i] , l_cur)
    # sss
    return l_cur

 


def  regular_loss_function(predictions, labels,n=100):
    T=labels.flatten()
    P=predictions.flatten()
    reg_loss=0
    N=len ( T)
    for i in range(N):
        if not P[i]==0 and  T[i] ==0:
            reg_loss=reg_loss-1*(T[i] * np.log(0.01+P[i]))/N 
            - 2*T[i] * P[i]/ (T[i] + P[i])
            
            reg_loss=reg_loss+1*( T[i] *(1-P[i])  +P[i] *(1-T[i]))/N 
            - T[i] * P[i]/ (T[i] + P[i])
            
    return 10000*reg_loss



def  MAE_loss_function(predictions, labels,n=100):
    T=labels.flatten()
    P=predictions.flatten()
    MAE_loss=0
    N=len ( T)
    for i in range(N):
        MAE_loss=MAE_loss+ (P[i] -T[i])**2
    return 1000*MAE_loss/N



def  DSSIM_loss_function(P, T,n=100):
    c1=10**-6;    c2=10**-6
    # c1=2.55 ;    c2=7.65 
    m1=np.mean(P);    m2=np.mean(T)
    s1=np.std(P);    s2=np.std(T)
    x1= (2*m1*m2 +c1)*(2*s1*s2 +c2)
    x2=(m1**2 +m2**2+c1)*(s1**2 +s2**2+c2)
    
    DSSIM=1-(x1) / (x2)
    # print(x1,x2,DSSIM)
    return  1000*DSSIM

def iou_loss_function( predictions, labels, smooth=1,th=0.5):
    # epsilon=10**6
    # for innnnn in range(len (predictions)):
    #     idx = np.where (predictions[innnnn] <th);        predictions[innnnn][idx]=0
    #     idx = np.where (predictions[innnnn] >=th);        predictions [innnnn][idx]=1
    # for innnnn in range(len (labels)):
    #     idx = np.where (labels[innnnn] <th);        labels[innnnn][idx]=0
    #     idx = np.where (labels[innnnn] >=th);        labels [innnnn][idx]=1    
        
    # iou = []
    # # print('labels.shape[-1]',labels.shape[-1])
    # for i in range(labels.shape[-1]):
    #   try: label_1D = K.flatten(labels[:, :, i])
    #   except:label_1D = K.flatten(labels )
      
    #   try: pred_1D  = K.flatten(predictions[:, :, i])
    #   except:pred_1D  = K.flatten(predictions )
     
    #   intersection = K.sum(label_1D * pred_1D)
    #   union = K.sum(label_1D) + K.sum(pred_1D) - K.sum((label_1D * pred_1D))
    #   # print('intersection',intersection)
    #   # print('union',union)
    #   try:
    #       iou_coff = (intersection + smooth) / (union + smooth)
    #       iou.append( float(iou_coff))
    #   except:
    #       iou_coff = 0
    #       iou.append( float(iou_coff))
    # iou=np.mean(iou)
    # IOU_loss=1/(epsilon +iou)
    IOU_loss = 1-np.mean (iou(predictions, labels))
    return 100*IOU_loss    

# =============================================================================
#  saed  loss

# dice_coef    ++++++++++++++  tensorflow.python.framework.ops.EagerTensor'

# contour 
# curvature


# regular          NaN
# MAE 

# DSSIM        ++++++++++++++++  
#  iou

 

# =============================================================================
def dice_coef_loss_saed(y_true, y_pred):
    # print(type (1-dice_coef(y_true, y_pred)))
    # ssss
    return 1- np.mean ( dice_coef(y_true, y_pred))

def contour_loss_function_saed   (predictions, labels ):
    # print( type( predictions) )
    # print( type( labels) )
    
    predictions = predictions.numpy()
    labels = labels.numpy()
    
    contour_loss=0
    for i in range( len(predictions)):
        P=predictions[i]
        T=labels[i]
        P=np.squeeze(P);        T=np.squeeze(T)
        contour_loss  = contour_loss+ contour_loss_function( P, T  )
    contour_loss_tensor = K.constant(contour_loss)
    return contour_loss_tensor


def curvature_loss_function_saed   (predictions, labels ): 
    predictions = predictions.numpy()
    labels = labels.numpy()
    
    curvature_loss=0
    for i in range( len(predictions)):
        P=predictions[i]
        T=labels[i]
        P=np.squeeze(P);        T=np.squeeze(T)
        
        curvature_loss  = curvature_loss_function( P, T  )
    curvature_loss_tensor = K.constant(curvature_loss)
    return curvature_loss_tensor

def  regular_loss_function_saed(predictions, labels,n=100):
         
    predictions = predictions.numpy()
    labels = labels.numpy()
    reg_loss=0
    for i in range( len(predictions)):
        P=predictions[i]
        T=labels[i]
        P=np.squeeze(P);        T=np.squeeze(T)
        reg_loss  = reg_loss+ regular_loss_function( P, T  )
    reg_loss_tensor = K.constant(reg_loss)   
    return reg_loss_tensor

 
def  MAE_loss_function_saed(predictions, labels ):
    predictions = predictions.numpy()
    labels = labels.numpy()
    MAE_loss=0
    for i in range( len(predictions)):
        P=predictions[i]
        T=labels[i]
        P=np.squeeze(P);        T=np.squeeze(T)
        MAE_loss  = MAE_loss+ MAE_loss_function( P, T  )
    MAE_loss_tensor = K.constant(MAE_loss)
    return 1000*MAE_loss_tensor


def  DSSIM_loss_function_saed(P, T,n=100):
    c1=10**-6;    c2=10**-6
    c1=2.55 ;    c2=7.65 
    m1=K.mean(P);    m2=K.mean(T)
    s1=K.std(P);    s2=K.std(T)
    DSSIM=1-((2*m1*m2 +c1)*(2*s1*s2 +c2)) / ((m1**2 +m2**2+c1)*(s1**2 +s2**2+c2)) 
    return DSSIM



def iou_loss_function_saed( predictions, labels):
    predictions = predictions.numpy()
    labels = labels.numpy()
    IOU_loss=0
    for i in range( len(predictions)):
        P=predictions[i]
        T=labels[i]
        P=np.squeeze(P);        T=np.squeeze(T)
        IOU_loss  = IOU_loss+ iou_loss_function( P, T  )
    # MAE_loss_tensor = K.constant(MAE_loss)
    IOU_loss_tensor = K.constant(IOU_loss)
    return 1000*IOU_loss_tensor



# =============================================================================
# 



# =============================================================================
    
