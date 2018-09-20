import os, re, glob
from collections import OrderedDict
import numpy as np
import scipy.misc
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from keras.callbacks import Callback

def build(self, tp, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        # Image size must be dividable by 2 multiple times
        if self.img_size / 2**6 != int(self.img_size / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(shape=(self.img_size*4, self.img_size*4, 3),name='input_image')
        weight_true = KL.Input(shape=(self.img_size, self.img_size, 1))
        #input_image = KL.Input(shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        #input_up = KL.Conv2DTranspose(3, (7,7), strides=(4,4), padding='same',
        #                              name = 'head_input_convt')(input_image)

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        _, C2, C3, C4, C5 = resnet_graph(input_image, "resnet101", stage5=True)
     
        #C1 = C1_graph(input_image, ksize=config.C1_KSIZE, depth=config.C1_DEPTH)
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])

        '''
        for nb_p1 in range(2):    
            P2 = KL.Conv2DTranspose(256, (2, 2), strides=(2, 2), activation="relu",
                               name="head_fpn_p2ct_{}".format(nb_p1))(P2)
        P1 = KL.Add(name="head_fpn_p1add")([P2,
            KL.Conv2D(256, (1, 1), name = 'head_fpn_c1p1')(C1)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        
        P1 = KL.Conv2D(256, (1, 1), padding="SAME", activation='relu',
                       name="head_fpn_p1")(P1)
        '''
        P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
        
        P2_true = KL.Conv2D(256, (3, 3), padding="SAME", activation='relu',
                            name="fpn_p2_true_conv0")(P2)          
        output_y_true = KL.Conv2D(1, (1, 1), activation='tanh', name='output_y_true_conv')(P2_true)
        output_x_true = KL.Conv2D(1, (1, 1), activation='tanh', name='output_x_true_conv')(P2_true)
        output_dr_true= KL.Conv2D(1, (1, 1), activation='tanh', name='output_dr_true_conv')(P2_true)
        output_dl_true = KL.Conv2D(1, (1, 1), activation='tanh', name='output_dl_true_conv')(P2_true)

        weight_pred = KL.Lambda(lambda x: inst_weight(*x, config = config), name="weight_pred")(
                    [output_y_true, output_x_true, output_dr_true, output_dl_true])
        weight = KL.Maximum()([weight_pred, weight_true])

        output_y_true = KL.Concatenate(name='output_y_true')([output_y_true, weight_true])
        output_x_true = KL.Concatenate(name='output_x_true')([output_x_true, weight_true])
        output_dr_true= KL.Concatenate(name='output_dr_true')([output_dr_true, weight_true])
        output_dl_true = KL.Concatenate(name='output_dl_true')([output_dl_true, weight_true])
        
        P2_pred = KL.Conv2D(256, (3, 3), padding="SAME", activation='relu',
                            name="fpn_p2_pred_conv0")(P2)            
        output_mask = KL.Conv2D(1, (1, 1), activation='sigmoid', 
                                name='output_mask')(P2_pred)    
        output_y = KL.Conv2D(1, (1, 1), activation='tanh', name='output_y_conv')(P2_pred)
        output_x = KL.Conv2D(1, (1, 1), activation='tanh', name='output_x_conv')(P2_pred)
        output_dr= KL.Conv2D(1, (1, 1), activation='tanh', name='output_dr_conv')(P2_pred)
        output_dl = KL.Conv2D(1, (1, 1), activation='tanh', name='output_dl_conv')(P2_pred)
        output_ly = KL.Conv2D(1, (1, 1), activation='tanh', name='output_ly_conv')(P2_pred)
        output_lx = KL.Conv2D(1, (1, 1), activation='tanh', name='output_lx_conv')(P2_pred)
        output_ldr= KL.Conv2D(1, (1, 1), activation='tanh', name='output_ldr_conv')(P2_pred)
        output_ldl = KL.Conv2D(1, (1, 1), activation='tanh', name='output_ldl_conv')(P2_pred)

        output_y = KL.Concatenate(name='output_y')([output_y, weight])
        output_x = KL.Concatenate(name='output_x')([output_x, weight])
        output_dr= KL.Concatenate(name='output_dr')([output_dr, weight])
        output_dl = KL.Concatenate(name='output_dl')([output_dl, weight])
        output_ly = KL.Concatenate(name='output_ly')([output_ly, weight])
        output_lx = KL.Concatenate(name='output_lx')([output_lx, weight])
        output_ldr= KL.Concatenate(name='output_ldr')([output_ldr, weight])
        output_ldl = KL.Concatenate(name='output_ldl')([output_ldl, weight])
        
        output_inst = [output_y_true, output_x_true, output_dr_true, output_dl_true, 
                       output_y, output_x, output_dr, output_dl, 
                       output_ly,output_lx,output_ldr,output_ldl]
        #output_inst =[output_ly, output_lx, output_ldr, output_ldl][self.nb_start:
         #               self.nb_start+self.nb_feature]
    
        model = KM.Model(inputs=[input_image, weight_true], 
                         outputs=[output_mask]+ output_inst)

    return model
	
	

	
	
import keras.backend as K                                    
from keras.layers import BatchNormalization, Activation, Input, Dropout, \
                         Conv2D, MaxPooling2D,Conv2DTranspose, Concatenate, \
                         Reshape, Lambda, Maximum
from keras import Model, optimizers,losses, activations, models
import tensorflow as tf

def inst_weight(output_y, output_x, output_dr, output_dl, config=None):
    dy = output_y[:,2:,2:]-output_y[:, :-2,2:] + \
         2*(output_y[:,2:,1:-1]- output_y[:,:-2,1:-1]) + \
         output_y[:,2:,:-2]-output_y[:,:-2,:-2]
    dx = output_x[:,2:,2:]- output_x[:,2:,:-2] + \
         2*( output_x[:,1:-1,2:]- output_x[:,1:-1,:-2]) +\
         output_x[:,:-2,2:]- output_x[:,:-2,:-2]
    ddr=  (output_dr[:,2:,2:]-output_dr[:,:-2,:-2] +\
           output_dr[:,1:-1,2:]-output_dr[:,:-2,1:-1]+\
           output_dr[:,2:,1:-1]-output_dr[:,1:-1,:-2])*K.constant(2)
    ddl=  (output_dl[:,2:,:-2]-output_dl[:,:-2,2:] +\
           output_dl[:,2:,1:-1]-output_dl[:,1:-1,2:]+\
           output_dl[:,1:-1,:-2]-output_dl[:,:-2,1:-1])*K.constant(2)
    dpred = K.concatenate([dy,dx,ddr,ddl],axis=-1)
    dpred = K.spatial_2d_padding(dpred)
    weight_fg = K.cast(K.all(dpred>K.constant(config.GRADIENT_THRES), axis=3, 
                          keepdims=True), K.floatx())
    
    weight = K.clip(K.sqrt(weight_fg*K.prod(dpred, axis=3, keepdims=True)), 
                    config.WEIGHT_AREA/config.CLIP_AREA_HIGH, 
                    config.WEIGHT_AREA/config.CLIP_AREA_LOW)
    weight +=(1-weight_fg)*config.WEIGHT_AREA/config.BG_AREA
    weight = K.conv2d(weight, K.constant(config.GAUSSIAN_KERNEL),
                      padding='same')
return K.stop_gradient(weight)