import tensorflow as tf


class Channel_attention(tf.keras.layers.Layer):
    """ 
    Channel attention module 
    
    Fu, Jun, et al. "Dual attention network for scene segmentation." 
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    """

    def __init__(
        self,
        gamma_initializer=tf.zeros_initializer(),
        gamma_regularizer=None,
        gamma_constraint=None,
        **kwargs
    ):
        super(Channel_attention, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def get_config(self):
        config = super(Channel_attention, self).get_config()
        config.update(
            {
                'gamma_initializer': self.gamma_initializer,
                'gamma_regularizer': self.gamma_regularizer,
                'gamma_constraint': self.gamma_constraint
            }
        )
        return config

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(1, ),
            initializer=self.gamma_initializer,
            name='gamma',
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint
        )
        super(Channel_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()

        proj_query = tf.keras.layers.Reshape(
            (input_shape[1] * input_shape[2], input_shape[3])
        )(inputs)
        proj_key = tf.keras.backend.permute_dimensions(proj_query, (0, 2, 1))
        energy = tf.keras.backend.batch_dot(proj_query, proj_key)
        attention = tf.keras.activations.softmax(energy)

        outputs = tf.keras.backend.batch_dot(attention, proj_query)
        outputs = tf.keras.layers.Reshape(
            (input_shape[1], input_shape[2], input_shape[3])
        )(outputs)
        # outputs = self.gamma * outputs + inputs
        outputs = self.gamma * outputs

        return outputs

class Position_attention(tf.keras.layers.Layer):
    """ 
    Position attention module 
        
    Fu, Jun, et al. "Dual attention network for scene segmentation." 
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    """

    def __init__(
        self,
        ratio=8,
        gamma_initializer=tf.zeros_initializer(),
        gamma_regularizer=None,
        gamma_constraint=None,
        **kwargs
    ):
        super(Position_attention, self).__init__(**kwargs)
        self.ratio = ratio
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def get_config(self):
        config = super(Position_attention, self).get_config()
        config.update(
            {
                'ratio': self.ratio,
                'gamma_initializer': self.gamma_initializer,
                'gamma_regularizer': self.gamma_regularizer,
                'gamma_constraint': self.gamma_constraint
            }
        )
        return config

    def build(self, input_shape):
        super(Position_attention, self).build(input_shape)
        self.query_conv = tf.keras.layers.Conv2D(
            filters=input_shape[-1] // self.ratio,
            kernel_size=(1, 1),
            use_bias=False,
            kernel_initializer='he_normal'
        )
        self.key_conv = tf.keras.layers.Conv2D(
            filters=input_shape[-1] // self.ratio,
            kernel_size=(1, 1),
            use_bias=False,
            kernel_initializer='he_normal'
        )
        self.value_conv = tf.keras.layers.Conv2D(
            filters=input_shape[-1],
            kernel_size=(1, 1),
            use_bias=False,
            kernel_initializer='he_normal'
        )
        self.gamma = self.add_weight(
            shape=(1, ),
            initializer=self.gamma_initializer,
            name='gamma',
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()

        proj_query = tf.keras.layers.Reshape(
            (input_shape[1] * input_shape[2], input_shape[3] // self.ratio)
        )(self.query_conv(inputs))
        proj_query = tf.keras.backend.permute_dimensions(proj_query, (0, 2, 1))
        proj_key = tf.keras.layers.Reshape(
            (input_shape[1] * input_shape[2], input_shape[3] // self.ratio)
        )(self.key_conv(inputs))
        energy = tf.keras.backend.batch_dot(proj_key, proj_query)
        attention = tf.keras.activations.softmax(energy)

        proj_value = tf.keras.layers.Reshape(
            (input_shape[1] * input_shape[2] , input_shape[3] )
        )(self.value_conv(inputs))

        outputs = tf.keras.backend.batch_dot(attention, proj_value)
        outputs = tf.keras.layers.Reshape(
            (input_shape[1], input_shape[2], input_shape[3] )
        )(outputs)
        # outputs = self.gamma * outputs + inputs
        outputs = self.gamma * outputs 

        return outputs

class coef_layer(tf.keras.layers.Layer):
    def __init__(self, name="coef_layer", **kwargs):
        super(coef_layer, self).__init__(name=name, **kwargs)
        self.w_init = tf.random_normal_initializer()
        self.w = None

    def build(self, input_shape):
        # Initialize w as a scalar trainable variable
        self.w = self.add_weight(
            shape=(1,),
            initializer=self.w_init,
            trainable=True,
            name="w"
        )
    
    def call(self, inputs):
        return inputs * tf.nn.softplus(self.w)

class VIT_function(tf.keras.layers.Layer):
    def __init__(
        self,
        image_size=(None, 128, 128, 1),
        patch_size=32,
        num_patches=128,
        projection_dim=1280,
        vanilla=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vanilla = vanilla  # Flag to switch to vanilla patch extractor
        self.image_size = image_size  # Expected input image size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.flatten_patches = tf.keras.layers.Reshape((num_patches, -1))
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        # Validate the input shape dimensions
        if len(input_shape) != 4:
            raise ValueError("Input shape must have 4 dimensions: (batch, height, width, channels)")
        if input_shape[1] != self.image_size[1] or input_shape[2] != self.image_size[2]:
            raise ValueError(f"Input height and width must match {self.image_size[1:3]}")

    def crop_shift_pad(self, images, mode):
        # Create shifted crops based on the mode
        height, width = self.image_size[1], self.image_size[2]
        if mode == "left-up":
            crop_height, crop_width = self.half_patch, self.half_patch
            shift_height, shift_width = 0, 0
        elif mode == "left-down":
            crop_height, crop_width = 0, self.half_patch
            shift_height, shift_width = self.half_patch, 0
        elif mode == "right-up":
            crop_height, crop_width = self.half_patch, 0
            shift_height, shift_width = 0, self.half_patch
        else:  # "right-down"
            crop_height, crop_width = 0, 0
            shift_height, shift_width = self.half_patch, self.half_patch

        # Crop the image and apply padding
        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=height - self.half_patch,
            target_width=width - self.half_patch,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=height,
            target_width=width,
        )
        return shift_pad

    def call(self, images):
        if not self.vanilla:
            # Concatenate shifted images along the channel axis
            images = tf.concat(
                [
                    images,
                    self.crop_shift_pad(images, mode="left-up"),
                    self.crop_shift_pad(images, mode="left-down"),
                    self.crop_shift_pad(images, mode="right-up"),
                    self.crop_shift_pad(images, mode="right-down"),
                ],
                axis=-1,
            )
        # Extract patches from images and flatten them
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        flat_patches = self.flatten_patches(patches)
        if not self.vanilla:
            # Normalize and project the patches
            tokens = self.layer_norm(flat_patches)
            tokens = self.projection(tokens)
        else:
            # Directly project the patches
            tokens = self.projection(flat_patches)

        # Reshape tokens back to the original image size (without batch size)
        tokens = tf.keras.layers.Reshape(self.image_size[1:])(tokens)
        tokens = tf.keras.activations.softmax(tokens)
        return tokens

    def get_config(self):
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
            "vanilla": self.vanilla,
        })
        return config

class CardSegNet:
    def __init__(self, num_class=4, input_shape=(128, 128,3) , ENABLE_ADS= False):
        self.num_class   = num_class
        self.input_shape = input_shape
        self.ENABLE_ADS = ENABLE_ADS
    
    def downsample2(self,img):
        x = tf.keras.layers.MaxPooling2D((2, 2))(img)
        return x
    
    def init_layer(self,layer):
        session = tf.keras.get_session()
        weights_initializer = tf.variables_initializer(layer.weights)
        session.run(weights_initializer)
    
    def conv2d_block(self,input_tensor, n_filters=10, kernel_size = 3, batchnorm = True):
        x = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                  kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    def conv_block_simple(self,prevlayer, filters, prefix, strides=(1, 1)):
        conv = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
        conv = tf.keras.layers.BatchNormalization(name=prefix + "_bn")(conv)
        conv = tf.keras.layers.Activation('relu', name=prefix + "_activation")(conv)
        return conv
    
    def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
        conv = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
        conv = tf.keras.layers.Activation('relu', name=prefix + "_activation")(conv)
        return conv
    
    def identity_block(self,input_tensor, kernel_size=3, filters=[3,3,3], stage='stage1', block='b1'):
        filters1, filters2, filters3 = filters
        if tf.keras.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = tf.keras.layers.Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = tf.keras.layers.add([x, input_tensor])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    def BB_Resnet(self):
        base_model = tf.keras.applications.resnet50.ResNet50(include_top=False , input_shape=self.input_shape, pooling='avg', weights="imagenet")
        resnet_base = tf.keras.models.Model(base_model.input, base_model.layers[142].output)
        
        for l in resnet_base.layers:
            l.trainable = False
        
        conv1 = resnet_base.layers[0].output
        conv2 = resnet_base.layers[4].output
        conv3 = resnet_base.layers[38].output
        conv4 = resnet_base.layers[80].output
        conv5 = resnet_base.layers[142].output
        
        up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv5), conv4], axis=-1)
        conv6 = self.conv_block_simple(up6, 256, "conv6_1")
        conv6 = self.conv_block_simple(conv6, 256, "conv6_2")

        up7 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv6), conv3], axis=-1)
        conv7 = self.conv_block_simple(up7, 192, "conv7_1")
        conv7 = self.conv_block_simple(conv7, 192, "conv7_2")

        up8 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv7), conv2], axis=-1)
        conv8 = self.conv_block_simple(up8, 128, "conv8_1")
        conv8 = self.conv_block_simple(conv8, 128, "conv8_2")

        up9 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D()(conv8), conv1], axis=-1)
        conv9 = self.conv_block_simple(up9, 64, "conv9_1")
        conv9 = self.conv_block_simple(conv9, 64, "conv9_2")

        # up10 = UpSampling2D()(conv9) TODO: WHY this commented?
        conv10 = self.conv_block_simple(conv9, 32, "conv10_1")
        conv10 = self.conv_block_simple(conv10, 32, "conv10_2")
        x = tf.keras.layers.SpatialDropout2D(0.2)(conv10)
        x = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(x)
        model = tf.keras.models.Model(resnet_base.input, x)
        
        return model 

    def back_bone(self, input_tensor, name = 'resnet'):
        if name == "resnet":
            BB_model = self.BB_Resnet()
            f001 = BB_model(input_tensor)
            return f001
        else:
            raise NotImplementedError

    def Encoder_Block0(
        self,
        input_tensor, 
        n_filters =10, 
        kernel_size = 3,
        batchnorm = True,
        dual_attention_enable_Encoder_Block0='sc',
        section_name_Encoder_Block0 = 'section_name_Encoder_Block0'
    ):
        x = self.conv2d_block(
            input_tensor, 
            n_filters, 
            kernel_size = kernel_size, 
            batchnorm = batchnorm
        )
        
        x = self.conv2d_block(
            x, 
            n_filters, 
            kernel_size = kernel_size, 
            batchnorm = batchnorm
        )
        
        x =self.VDAB_block(
            x,
            3,
            kernel_size = kernel_size, 
            batchnorm = True,
            dual_attention_enable = dual_attention_enable_Encoder_Block0,
            section_name = section_name_Encoder_Block0
        )
        
        return x
    
    def Encoder_Block(self,input_tensor, n_filters=10, kernel_size = 3, batchnorm = True,
                      dual_attention_enable_Encoder_Block='sc',
                      section_name_Encoder_Block = 'section_name_Encoder_Block'
                      ):
        x = tf.keras.layers.MaxPooling2D((2, 2))(input_tensor)
        x = self.conv2d_block(x, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        x = self.conv2d_block(x, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        x = self.VDAB_block(x ,3,
                        kernel_size = 3, batchnorm = True,
                                        dual_attention_enable=dual_attention_enable_Encoder_Block ,
                                        section_name = section_name_Encoder_Block
                                        )
        return x
    
    def Decoder_Block(self,A2, B3, n_filters=10, kernel_size = 3, batchnorm = True,
                    dual_attention_enable_Decoder_Block='vsc',
                    section_name_Decoder_Block = 'section_name_Decoder_Block'
                       ):
        
        
        input_tensor = tf.keras.layers.concatenate([A2, B3], axis=-1)
        
        # x = MaxPooling2D((2, 2))(input_tensor)
        x = self.conv2d_block(input_tensor, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        x = self.conv2d_block(x, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        x=  tf.keras.layers.UpSampling2D( size=(2, 2),  interpolation="nearest")(x)
        x = self.conv2d_block(x, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        x = self.VDAB_block(x ,3,
                        kernel_size = 3, batchnorm = True,
                                        dual_attention_enable=dual_attention_enable_Decoder_Block ,
                                        section_name = section_name_Decoder_Block
                                        )
        return x
    
    def Decoder_Block0(self,A1,B2 ,  n_filters=10, kernel_size = 3, batchnorm = True,
                       dual_attention_enable_Decoder_Block0='vsc',
                       section_name_Decoder_Block0 = 'section_name_Decoder_Block0'
                       ):
        x = tf.keras.layers.concatenate([A1, B2], axis=-1)
        
        D1 = self.VDAB_block(x ,n_filters,
                        kernel_size = 3, batchnorm = True,
                                        dual_attention_enable=dual_attention_enable_Decoder_Block0 ,
                                        section_name = section_name_Decoder_Block0
                                        )
        
        
        # x = conv2d_block(input_tensor, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        x = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(D1)
        

        o1 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid',name= "Right_Ventricle")(x)
        o2 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid',name= "Myocard")(x)
        o3 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid',name= "Left_Ventricle")(x)
        
        # out_title
        
        return o1,o2,o3
    
    def Decoder_Block1(self,input_tensor, n_filters=10, kernel_size = 3, batchnorm = True
                        ,
                        
                        dual_attention_enable_Decoder_Block1='vsc',
                        section_name_Decoder_Block1 = 'section_name_Decoder_Block1'
                        ):
        x=  tf.keras.layers.UpSampling2D( size=(2, 2),  interpolation="nearest")(input_tensor)
        x = self.conv2d_block(x, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        
        x = self.VDAB_block(x ,n_filters,
                        kernel_size = 3, batchnorm = True,
                                        dual_attention_enable=dual_attention_enable_Decoder_Block1 ,
                                        section_name = section_name_Decoder_Block1
                                        )
        
        return x

    def conv2d_block2(self,input_tensor, n_filters=10, kernel_size = 3, batchnorm = True):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        x = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                  kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        # second layer
        x = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                  kernel_initializer = 'he_normal', padding = 'same')(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def VDAB_block(self,input_tensor, n_filters=10, kernel_size = 3, batchnorm = True,
                    dual_attention_enable='vsc',
                    section_name = 'section name'):
        x_att_cross=True

        
        input_tensor = self.downsample2 (input_tensor)
        
        
        
        ATTENTION=0
        
        att_show=0
        if 'c' in dual_attention_enable:
            x_CAM=Channel_attention()(input_tensor)
            beta_x_att_ch = coef_layer (name=section_name + '_channel')(x_CAM)
            ATTENTION=ATTENTION+beta_x_att_ch
            att_show=att_show+1
            
        if 's' in dual_attention_enable:
            x_PAM=Position_attention()(input_tensor)
            alpha_x_att_pos = coef_layer(name=section_name + '_position')(x_PAM)
            ATTENTION=ATTENTION+alpha_x_att_pos
            att_show=att_show+10
            
        if 'v' in dual_attention_enable:
            x_VIT = VIT_function(image_size=input_tensor.shape,
            patch_size=32,
            num_patches=input_tensor.shape[1],
            projection_dim=input_tensor.shape[2]*input_tensor.shape[3],
            vanilla=True)(input_tensor)
            
            
            gamma_x_att_vit = coef_layer(name=section_name + '_VIT')(x_VIT)
            ATTENTION=ATTENTION+gamma_x_att_vit
            att_show=att_show+100
            
            
        if att_show>0:
            out_att_block       =  input_tensor  +  ATTENTION
            if x_att_cross:
                out_att_block   = input_tensor  * ATTENTION
        
        if att_show==0:
            out_att_block=input_tensor
        
        out_att_block = tf.keras.layers.UpSampling2D((2,2))( out_att_block )
            
        
        return out_att_block

    def DAB2in(self,Di, Ej ,n_f ,
                    kernel_size = 3, batchnorm = True,
                    dual_attention_enable_DAB2in='vsc',
                    section_name_DAB2in = 'section name DAB2in' ):
        DiEj = tf.keras.layers.concatenate([Di, Ej], axis=-1)
        
        Ai = self.VDAB_block(DiEj ,n_f,
                        kernel_size = 3, batchnorm = True,
                                        dual_attention_enable=dual_attention_enable_DAB2in ,
                                        section_name = section_name_DAB2in
                                        )
        return Ai

    def MSDAB(self,E4, E3, E2,E1 ,n_f ,dual_attention_enable_MSDAB='vsc',section_name_MSDAB = 'section_MSDAB'):
        
        E42 = tf.keras.layers.AveragePooling2D( pool_size=(1, 1),  strides=None, padding="valid", data_format=None)(E4)
        E32 = tf.keras.layers.AveragePooling2D( pool_size=(2, 2),  strides=None, padding="valid", data_format=None)(E3)
        E22 = tf.keras.layers.AveragePooling2D( pool_size=(4, 4),  strides=None, padding="valid", data_format=None)(E2)
        E12 = tf.keras.layers.AveragePooling2D( pool_size=(8, 8),  strides=None, padding="valid", data_format=None)(E1)
        
        DiEj = tf.keras.layers.concatenate([E42, E32, E22,E12], axis=-1)
        Ai = self.VDAB_block(
            DiEj,
            n_f,
            dual_attention_enable = dual_attention_enable_MSDAB,
            section_name = section_name_MSDAB   
        )
        return Ai
    
    def __call__(
        self,
        backbone_name = 'resnet', 
        dual_attention_enable_model = 'vsc',
        n_filters = 10, 
    ):

        input_img = tf.keras.layers.Input(
            shape = self.input_shape, 
            name = 'img'
        )
        
        if backbone_name.lower() == "resnet":
            F0 = self.back_bone(input_img, backbone_name)  
            E1 = self.Encoder_Block0(
                F0,
                dual_attention_enable_Encoder_Block0 = dual_attention_enable_model,
                section_name_Encoder_Block0 = 'section_Encoder1'
            )
        else:
            raise NotImplementedError("This part should be implemented again!")
            E1 = self.Encoder_Block0(input_img,dual_attention_enable_Encoder_Block0=dual_attention_enable_model,
            section_name_Encoder_Block0 = 'section_Encoder1')
            
        E2 = self.Encoder_Block(
            E1,
            dual_attention_enable_Encoder_Block = dual_attention_enable_model,
            section_name_Encoder_Block = 'section_Encoder2'
        )
        
        E3 = self.Encoder_Block(
            E2,
            dual_attention_enable_Encoder_Block = dual_attention_enable_model,
            section_name_Encoder_Block = 'section_Encoder3'
        )
        
        E4 = self.Encoder_Block(
            E3,
            dual_attention_enable_Encoder_Block = dual_attention_enable_model,
            section_name_Encoder_Block = 'section_Encoder4'
        )
        
        A4 = self.MSDAB(
            E4, 
            E3, 
            E2,
            E1,
            n_filters * 8,
            dual_attention_enable_MSDAB = dual_attention_enable_model
        )
        
        E4 =self.Decoder_Block1(
            A4,
            dual_attention_enable_Decoder_Block1 = dual_attention_enable_model,
            section_name_Decoder_Block1 = 'section_Decoder4'
        )

        A3 = self.DAB2in(
            E3,
            E4,
            n_filters * 8,
            dual_attention_enable_DAB2in = dual_attention_enable_model,
            section_name_DAB2in = 'section_name_A3'
        )
        
        E3 = self.Decoder_Block(
            A3,
            E4,
            dual_attention_enable_Decoder_Block = dual_attention_enable_model,
            section_name_Decoder_Block = 'section_Decoder3'
        )
        
        A2 = self.DAB2in(
            E2,
            E3,
            n_filters * 8,
            dual_attention_enable_DAB2in = dual_attention_enable_model,
            section_name_DAB2in = 'section_name_A2'
        )

        E2 = self.Decoder_Block(
            A2,
            E3,
            dual_attention_enable_Decoder_Block = dual_attention_enable_model,
            section_name_Decoder_Block = 'section_Decoder2'
        )
        
        A1 = self.DAB2in(
            E1, 
            E2,
            n_filters * 8,
            dual_attention_enable_DAB2in = dual_attention_enable_model,
            section_name_DAB2in = 'section_name_A1'
        )
        
        Output  = tf.keras.layers.Conv2D(self.num_class, (1, 1), activation = None)(A1)
        
        Output = tf.keras.layers.BatchNormalization()(Output)
        Output  = tf.keras.layers.Activation('sigmoid')(Output)
        
        if not self.ENABLE_ADS:
            model = tf.keras.models.Model(
                inputs=[input_img], 
                outputs=[Output],
            )
        
        else:
            raise NotImplementedError("This part should be checked again!")
            layer_names = [layer.name for layer in model.layers]
            ADS_layers=[]
            
            cnt=0
            for layer_name in layer_names:
                if 'section_name' in layer_name:
                    ADS_layers.append(layer_name)
                    ADS_out = model.get_layer(ADS_layers[-1]).output
                    
                    x1 = tf.keras.layers.Conv2D(4, kernel_size=(3, 3), activation='relu', padding='same')(ADS_out)
                    
                    while not x1.shape [1] == 128:
                        x1=tf.keras.layers.UpSampling2D()(x1)
                    
                    cnt=cnt+1
                     
            output001 = model.get_layer('section_name_A1_channel').output
            output001 = tf.keras.layers.Conv2D(self.num_class, (1, 1))(output001)
            output001C = tf.keras.layers.Activation("softmax", name="output001C")(output001)
            output001C=tf.keras.layers.UpSampling2D()(output001C)
            
            output001 = model.get_layer('section_name_A1_position').output
            output001 = tf.keras.layers.Conv2D(self.num_class, (1, 1))(output001)
            output001P = tf.keras.layers.Activation("softmax", name="output001P")(output001)
            output001P=tf.keras.layers.UpSampling2D()(output001P)
            
            output001 = model.get_layer('section_name_A1_VIT').output
            output001 = tf.keras.layers.Conv2D(self.num_class, (1, 1))(output001)
            output001V = tf.keras.layers.Activation("softmax", name="output001V")(output001)
            output001V=tf.keras.layers.UpSampling2D()(output001V)
            
            
            output002 = model.get_layer('section_name_A2_channel').output
            output002 = tf.keras.layers.Conv2D(self.num_class, (1, 1))(output002)
            output002C = tf.keras.layers.Activation("softmax", name="output002C")(output002)
            output002C=tf.keras.layers.UpSampling2D()(output002C)
            output002C=tf.keras.layers.UpSampling2D()(output002C)

            output002 = model.get_layer('section_name_A2_position').output
            output002 = tf.keras.layers.Conv2D(self.num_class, (1, 1))(output002)
            output002P = tf.keras.layers.Activation("softmax", name="output002P")(output002)
            output002P=tf.keras.layers.UpSampling2D()(output002P)
            output002P=tf.keras.layers.UpSampling2D()(output002P)
            
            
            output002 = model.get_layer('section_name_A2_VIT').output
            output002 = tf.keras.layers.Conv2D(self.num_class, (1, 1))(output002)
            output002V = tf.keras.layers.Activation("softmax", name="output002V")(output002)
            output002V=tf.keras.layers.UpSampling2D()(output002V)
            output002V=tf.keras.layers.UpSampling2D()(output002V)
            
            
            
            output003 = model.get_layer('section_name_A3_channel').output
            output003 = tf.keras.layers.Conv2D(self.num_class, (1, 1))(output003)
            output003C = tf.keras.layers.Activation("softmax", name="output003C")(output003)
            output003C=tf.keras.layers.UpSampling2D()(output003C)
            output003C=tf.keras.layers.UpSampling2D()(output003C)
            output003C=tf.keras.layers.UpSampling2D()(output003C)
            

            output003 = model.get_layer('section_name_A3_position').output
            output003 = tf.keras.layers.Conv2D(self.num_class, (1, 1))(output003)
            output003P = tf.keras.layers.Activation("softmax", name="output003P")(output003)
            output003P=tf.keras.layers.UpSampling2D()(output003P)
            output003P=tf.keras.layers.UpSampling2D()(output003P)
            output003P=tf.keras.layers.UpSampling2D()(output003P)
            
            
            output003 = model.get_layer('section_name_A3_VIT').output
            output003 = tf.keras.layers.Conv2D(self.num_class, (1, 1))(output003)
            output003V = tf.keras.layers.Activation("softmax", name="output003V")(output003)
            output003V=tf.keras.layers.UpSampling2D()(output003V)
            output003V=tf.keras.layers.UpSampling2D()(output003V)
            output003V=tf.keras.layers.UpSampling2D()(output003V)
            
            outputs_to_return = tf.keras.layers.concatenate([ Output ,output001C,output001P,output001V,output002C,output002P,output002V , output003C,output003P,output003V])
            
            model = tf.keras.models.Model(inputs=[input_img], outputs=[outputs_to_return ])
                    
        return model

if __name__ == "__main__":
    model = CardSegNet(input_shape=(128, 128, 3))()
    model.summary()
    output = model.predict(tf.zeros((1, 128, 128, 3)))
    print(output.shape)