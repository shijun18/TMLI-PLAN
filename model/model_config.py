MODEL_CONFIG = {
    'unet':{
        'simplenet':{
            'in_channels':1,
            'encoder_name':'simplenet',
            'encoder_depth':5,
            'encoder_channels':[32,64,128,256,512],  #[1,2,4,8,16]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[8,4,2,1]
            'upsampling':1,
            'classes':2,
            'aux_classifier': False
        },
        'resnet18':{
            'in_channels':1,
            'encoder_name':'resnet18',
            'encoder_depth':5,
            'encoder_channels':[64,64,128,256,512],  #[2,4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[16,8,4,2]
            'upsampling':2,
            'classes':2,
            'aux_classifier': False
        },
        'swin_transformer':{
            'in_channels':1,
            'encoder_name':'swin_transformer',
            'encoder_depth':4,
            'encoder_channels':[96,192,384,768],  #[4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64], #[16,8,4]
            'upsampling':4,
            'classes':2,
            'aux_classifier': False
        },
        'swinplusr18':{
            'in_channels':1,
            'encoder_name':'swinplusr18',
            'encoder_depth':5,
            'encoder_channels':[64,64,128,256,512],  #[2,4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[16,8,4,2]
            'upsampling':2,
            'classes':2,
            'aux_classifier': False
        } 
    },
    # att unet
    'att_unet':{
        'simplenet':{
            'in_channels':1,
            'encoder_name':'simplenet',
            'encoder_depth':5,
            'encoder_channels':[32,64,128,256,512],  #[1,2,4,8,16]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[8,4,2,1]
            'upsampling':1,
            'classes':2,
            'aux_classifier': False
        },
        'swin_transformer':{
            'in_channels':1,
            'encoder_name':'swin_transformer',
            'encoder_depth':4,
            'encoder_channels':[96,192,384,768],  #[4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64], #[16,8,4]
            'upsampling':4,
            'classes':2,
            'aux_classifier': False
        },
        'resnet18':{
            'in_channels':1,
            'encoder_name':'resnet18',
            'encoder_depth':5,
            'encoder_channels':[64,64,128,256,512],  #[2,4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[16,8,4,2]
            'upsampling':2,
            'classes':2,
            'aux_classifier': False
        }
    },
    # res unet
    'res_unet':{
        'simplenet':{
            'in_channels':1,
            'encoder_name':'simplenet_res',
            'encoder_depth':5,
            'encoder_channels':[32,64,128,256,512],  #[1,2,4,8,16]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[8,4,2,1]
            'upsampling':1,
            'classes':2,
            'aux_classifier': False
        },
        'resnet18':{
            'in_channels':1,
            'encoder_name':'resnet18',
            'encoder_depth':5,
            'encoder_channels':[64,64,128,256,512],  #[2,4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[16,8,4,2]
            'upsampling':2,
            'classes':2,
            'aux_classifier': False
        },
        'swinplusr18':{
            'in_channels':1,
            'encoder_name':'swinplusr18',
            'encoder_depth':5,
            'encoder_channels':[64,64,128,256,512],  #[2,4,8,16,32]
            'encoder_weights':None,
            'decoder_use_batchnorm':True,
            'decoder_attention_type':None,
            'decoder_channels':[256,128,64,32], #[16,8,4,2]
            'upsampling':2,
            'classes':2,
            'aux_classifier': False
        }
    },
    # deeplabv3+
    'deeplabv3+':{
        'swinplusr18':{
            'in_channels':1,
            'encoder_name':'swinplusr18',
            'encoder_weights':None,
            'encoder_depth':5,
            'encoder_channels':[64,64,128,256,512],  #[2,4,8,16,32]
            'encoder_output_stride':32, #[8,16,32]
            'decoder_channels':256, #[4]
            'decoder_atrous_rates':(12, 24, 36),
            'upsampling':4,
            'classes':2,
            'aux_classifier': False
        }
    },
    #bisenetv1
    'bisenetv1':{
        'resnet18':{
            'in_channels': 1,
            'encoder_name': 'resnet18',
            'encoder_weights': None,
            'encoder_depth': 5,
            'encoder_channels':[64,64,128,256,512], #[2,4,8,16,32]
            'encoder_outindice':[-2,-1],
            'decoder_use_batchnorm': True,
            'decoder_channels': [64,64,64,128],
            'upsampling': 8,
            'classes': 1,
            'aux_classifier': False
        },
        'swin_transformer':{
            'in_channels': 1,
            'encoder_name': 'swin_transformer',
            'encoder_weights': None,
            'encoder_depth': 4,
            'encoder_channels':[96,192,384,768],  #[4,8,16,32]
            'encoder_outindice':[-2,-1],
            'decoder_use_batchnorm': True,
            'decoder_channels': [64,64,64,128],
            'upsampling': 8,
            'classes': 1,
            'aux_classifier': False
        },
        'mobilenetv3_large_075':{
            'in_channels': 1,
            'encoder_name': 'mobilenetv3_large_075',
            'encoder_weights': None,
            'encoder_depth': 5,
            'encoder_channels':[16,24,32,88,720],  #[2,4,8,16,32]
            'encoder_outindice':[-2,-1],
            'decoder_use_batchnorm': True,
            'decoder_channels': [64,64,64,128],
            'upsampling': 8,
            'classes': 1,
            'aux_classifier': False
        },
        'xception':{
            'in_channels': 1,
            'encoder_name': 'xception',
            'encoder_weights': None,
            'encoder_depth': 5,
            'encoder_channels':[64,128,256,728,2048],  #[2,4,8,16,32]
            'encoder_outindice':[-2,-1],
            'decoder_use_batchnorm': True,
            'decoder_channels': [64,64,64,128],
            'upsampling': 8,
            'classes': 1,
            'aux_classifier': False
        }
    },
    # bisenetv2
    'bisenetv2':{
        'resnet18':{
            'in_channels': 1,
            'encoder_name': 'resnet18',
            'encoder_weights': None,
            'encoder_depth': 5,
            'encoder_channels':[64,64,128,256,512], #[2,4,8,16,32]
            'encoder_outindice':[-1],
            'decoder_use_batchnorm': True,
            'decoder_channels': [64,128],
            'upsampling': 8,
            'classes': 1,
            'aux_classifier': False
        },
        'swin_transformer':{
            'in_channels': 1,
            'encoder_name': 'swin_transformer',
            'encoder_weights': None,
            'encoder_depth': 4,
            'encoder_channels':[96,192,384,768],  #[4,8,16,32]
            'encoder_outindice':[-1],
            'decoder_use_batchnorm': True,
            'decoder_channels': [64,128],
            'upsampling': 8,
            'classes': 1,
            'aux_classifier': False
        },
        'mobilenetv3_large_075':{
            'in_channels': 1,
            'encoder_name': 'mobilenetv3_large_075',
            'encoder_weights': None,
            'encoder_depth': 5,
            'encoder_channels':[16,24,32,88,720],  #[2,4,8,16,32]
            'encoder_outindice':[-1],
            'decoder_use_batchnorm': True,
            'decoder_channels': [64,128],
            'upsampling': 8,
            'classes': 1,
            'aux_classifier': False
        },
        'xception':{
            'in_channels': 1,
            'encoder_name': 'xception',
            'encoder_weights': None,
            'encoder_depth': 5,
            'encoder_channels':[64,128,256,728,2048],  #[2,4,8,16,32]
            'encoder_outindice':[-1],
            'decoder_use_batchnorm': True,
            'decoder_channels': [64,128],
            'upsampling': 8,
            'classes': 1,
            'aux_classifier': False
        }
    },
    # sfnet
    'sfnet':{
        'simplenet':{
            'in_channels':1,
            'encoder_name':'simplenet',
            'encoder_weights':None,
            'encoder_depth':5,
            'encoder_channels':[32,64,128,256,512],  #[1,2,4,8,16]
            'num_stage':4,
            'decoder_use_batchnorm':True,
            'decoder_channels':[128], 
            'upsampling':2,
            'classes':2,
            'aux_classifier': False
        },
        'resnet18':{
            'in_channels':1,
            'encoder_name':'resnet18',
            'encoder_weights':None,
            'encoder_depth':5,
            'encoder_channels':[64,64,128,256,512],  #[2,4,8,16,32]
            'num_stage':4,
            'decoder_use_batchnorm':True,
            'decoder_channels':[128], #[16,8,4,2]
            'upsampling':4,
            'classes':2,
            'aux_classifier': False
        },
        'swin_transformer':{
            'in_channels': 1,
            'encoder_name': 'swin_transformer',
            'encoder_weights': None,
            'encoder_depth': 4,
            'encoder_channels':[96,192,384,768],  #[4,8,16,32]
            'num_stage':4,
            'decoder_use_batchnorm': True,
            'decoder_channels': [128],
            'upsampling':4,
            'classes':1,
            'aux_classifier':False
        },
    },
    # icnet
    'icnet':{
        'resnet18':{
            'in_channels':1,
            'encoder_name':'resnet18',
            'encoder_weights':None,
            'encoder_depth':5,
            'encoder_channels':[64,64,128,256,512],  #[2,4,8,16,32]
            'encoder_outindice':[2,4],
            'decoder_use_batchnorm':True,
            'decoder_channels':[32,64,128], #[16,8,4,2]
            'upsampling':4,
            'classes':2,
            'aux_classifier': False
        }
    }
}