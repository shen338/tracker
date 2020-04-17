import torchreid
import torch 
from torchvision.transforms import *
import numpy as np

class REID(object):
    
    def __init__(self, model='resnet18', height=128, width=128):
        
        torchreid.models.show_avai_models()
        
        self.height = height
        self.width = width
        
        reid_datamanager = torchreid.data.ImageDataManager(
            root='./fake_dataset',
            sources='market1501',
            targets='market1501',
            height=height,
            width=width,
            transforms=['random_flip', 'random_crop']
        )
        
        self.reid_model = torchreid.models.build_model(
            name=model,
            num_classes=reid_datamanager.num_train_pids,
            loss='softmax',
            pretrained=True
        )
        
        self.reid_model = self.reid_model.cuda()
        # self.reid_model = self.reid_model.eval()
        
        self.reid_engine = torchreid.engine.ImageSoftmaxEngine(
            reid_datamanager,
            self.reid_model,
            optimizer=None,
            scheduler=None,
            label_smooth=True
        )
        
        self.transform_tr, self.transform_te = self.build_transforms(self.height, self.width)  
        
        
    def build_transforms(
        self, 
        height,
        width,
        transforms='random_flip',
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
        **kwargs
    ):
        """Builds train and test transform functions.

        Args:
            height (int): target image height.
            width (int): target image width.
            transforms (str or list of str, optional): transformations applied to model training.
                Default is 'random_flip'.
            norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
            norm_std (list or None, optional): normalization standard deviation values. Default is
                ImageNet standard deviation values.
        """
        if transforms is None:
            transforms = []

        if isinstance(transforms, str):
            transforms = [transforms]

        if not isinstance(transforms, list):
            raise ValueError(
                'transforms must be a list of strings, but found to be {}'.format(
                    type(transforms)
                )
            )

        if len(transforms) > 0:
            transforms = [t.lower() for t in transforms]

        if norm_mean is None or norm_std is None:
            norm_mean = [0.485, 0.456, 0.406] # imagenet mean
            norm_std = [0.229, 0.224, 0.225] # imagenet std
        normalize = Normalize(mean=norm_mean, std=norm_std)

        print('Building train transforms ...')
        transform_tr = []

        print('+ resize to {}x{}'.format(height, width))
        transform_tr += [Resize((height, width))]

        if 'random_flip' in transforms:
            print('+ random flip')
            transform_tr += [RandomHorizontalFlip()]

        if 'random_crop' in transforms:
            print('+ random crop (enlarge to {}x{} and ' \
                  'crop {}x{})'.format(int(round(height*1.125)), int(round(width*1.125)), height, width))
            transform_tr += [Random2DTranslation(height, width)]

        if 'random_patch' in transforms:
            print('+ random patch')
            transform_tr += [RandomPatch()]

        if 'color_jitter' in transforms:
            print('+ color jitter')
            transform_tr += [
                ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)
            ]

        print('+ to torch tensor of range [0, 1]')
        transform_tr += [ToTensor()]

        print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
        transform_tr += [normalize]

        if 'random_erase' in transforms:
            print('+ random erase')
            transform_tr += [RandomErasing(mean=norm_mean)]

        transform_tr = Compose(transform_tr)

        print('Building test transforms ...')
        print('+ resize to {}x{}'.format(height, width))
        print('+ to torch tensor of range [0, 1]')
        print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))

        transform_te = Compose([
            Resize((height, width)),
            # ToTensor(),
            normalize,
        ])

        return transform_tr, transform_te      
        
    def extract_feature(self, img):
        
        # input shape (B, C, H, W)
        
        if len(img.shape) == 3: 
            img = np.expand_dims(img, axis=0)
            
        # img = self.transform_te(img)
        return self.reid_engine._extract_features(img)
        
        