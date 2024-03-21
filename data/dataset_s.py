#external libs
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import ImageFilter
import os
import random
from os.path import join as ospj
from glob import glob 
#torch libs
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
#local libs
from utils.utils import get_norm_values, chunks
from models.image_extractor import get_image_extractor
from itertools import product
import random
import pickle
import re
random.seed(777)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImageLoader:
    def __init__(self, root):
        self.root_dir = root

    def __call__(self, img):
        intab = r'[?*|:><]'
        img = re.sub(intab, "", img)
        img_root = ospj(self.root_dir, img)
        img = Image.open(img_root).convert('RGB') #We don't want alpha
        return img

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def dataset_transform(phase, norm_family = 'imagenet'):
    '''
        Inputs
            phase: String controlling which set of transforms to use
            norm_family: String controlling which normaliztion values to use
        
        Returns
            transform: A list of pytorch transforms
    '''
    mean, std = get_norm_values(norm_family=norm_family)

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif phase == 'val' or phase == 'test':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'all':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('Invalid transform')

    return transform

# Dataset class now

class CompositionDataset(Dataset):
    '''
    Inputs
        root: String of base dir of dataset
        phase: String train, val, test
        split: String dataset split
        subset: Boolean if true uses a subset of train at each epoch
        num_negs: Int, numbers of negative pairs per batch
        pair_dropout: Percentage of pairs to leave in current epoch
    '''
    def __init__(
        self,
        root,
        phase,
        split = 'compositional-split',
        model = 'resnet18',
        norm_family = 'imagenet',
        subset = False,
        num_negs = 1,
        pair_dropout = 0.0,
        update_features = False,
        return_images = False,
        train_only = False,
        open_world=False
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.model = model
        self.num_negs = num_negs
        self.pair_dropout = pair_dropout
        self.norm_family = norm_family
        self.return_images = return_images
        self.update_features = update_features
        self.feat_dim = 512 if 'resnet18' in model else 2048 # todo, unify this  with models
        self.feat_dim_l = self.feat_dim_h = 512
        self.open_world = open_world

        self.attrs, self.objs, self.pairs, self.train_pairs, \
            self.val_pairs, self.test_pairs = self.parse_split()
        self.train_data, self.val_data, self.test_data = self.get_split_info()
        self.full_pairs = list(product(self.attrs,self.objs))
        
        # Clean only was here
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr : idx for idx, attr in enumerate(self.attrs)}
        if self.open_world:
            self.pairs = self.full_pairs

        self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        if train_only and self.phase == 'train':
            print('Using only train pairs')
            self.pair2idx = {pair : idx for idx, pair in enumerate(self.train_pairs)}
        else:
            print('Using all pairs')
            self.pair2idx = {pair : idx for idx, pair in enumerate(self.pairs)}
        
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        elif self.phase == 'test':
            self.data = self.test_data
        elif self.phase == 'all':
            print('Using all data')
            self.data = self.train_data + self.val_data + self.test_data
        else:
            raise ValueError('Invalid training phase')
        
        self.all_data = self.train_data + self.val_data + self.test_data
        print('Dataset loaded')
        print('Train pairs: {}, Validation pairs: {}, Test Pairs: {}'.format(
            len(self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('Train images: {}, Validation images: {}, Test images: {}'.format(
            len(self.train_data), len(self.val_data), len(self.test_data)))

        if subset:
            ind = np.arange(len(self.data))
            ind = ind[::len(ind) // 1000]
            self.data = [self.data[i] for i in ind]


        # Keeping a list of all pairs that occur with each object
        self.obj_affordance = {}
        self.train_obj_affordance = {}
        for _obj in self.objs:
            candidates = [attr for (_, attr, obj) in self.train_data+self.test_data if obj==_obj]
            self.obj_affordance[_obj] = list(set(candidates))

            candidates = [attr for (_, attr, obj) in self.train_data if obj==_obj]
            self.train_obj_affordance[_obj] = list(set(candidates))

        if 'ut' in self.root:
            fo=open('utils/partial_utzappos_split.pkl','rb')
            sample_mask = pickle.load(fo)
        elif 'mit' in self.root:
            fo=open('utils/partial_mitstates_split.pkl','rb')
            sample_mask = pickle.load(fo)
        elif 'cgqa' in self.root:
            fo = open('utils/partial_cgqa_split.pkl','rb')
            sample_mask = pickle.load(fo)
        else:
            sample_mask = None

        self.sample_mask = sample_mask
        self.sample_indices = list(range(len(self.data)))
        #self.sample_mask = [random.randint(0,1) for i in range(len(self.data))]
        self.sample_pairs = self.train_pairs

        # Load based on what to output
        self.transform = dataset_transform(self.phase, self.norm_family)
        self.loader = ImageLoader(ospj(self.root, 'images'))
        if not self.update_features:
            feat_file = ospj(root, model+'_featurers.t7')
            print(f'Using {model} and feature file {feat_file}')
            if not os.path.exists(feat_file):
                with torch.no_grad():
                    self.generate_features(feat_file, model)
            self.phase = phase
            activation_data = torch.load(feat_file)

            self.activations_l = dict(
                zip(activation_data['files'], activation_data['features_l']))
            self.activations_h = dict(
                zip(activation_data['files'], activation_data['features_h']))
            self.feat_dim_l = activation_data['features_l'].size(1)
            self.feat_dim_h = activation_data['features_h'].size(1)
            print('{} activations loaded'.format(len(self.activations_l)))



    def parse_split(self):
        '''
        Helper function to read splits of object atrribute pair
        Returns
            all_attrs: List of all attributes
            all_objs: List of all objects
            all_pairs: List of all combination of attrs and objs
            tr_pairs: List of train pairs of attrs and objs
            vl_pairs: List of validation pairs of attrs and objs
            ts_pairs: List of test pairs of attrs and objs
        '''
        def parse_pairs(pair_list):
            '''
            Helper function to parse each phase to object attrribute vectors
            Inputs
                pair_list: path to textfile
            '''

            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                if "vaw-czsl" in self.root:
                    pairs = [t.split('+') for t in pairs]
                else:
                    pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            ospj(self.root, self.split, 'train_pairs.txt')
        )
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            ospj(self.root, self.split, 'val_pairs.txt')
        )
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            ospj(self.root, self.split, 'test_pairs.txt')
        )
        
        #now we compose all objs, attrs and pairs
        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def get_split_info(self):
        '''
        Helper method to read image, attrs, objs samples

        Returns
            train_data, val_data, test_data: List of tuple of image, attrs, obj
        '''
        data = torch.load(ospj(self.root, 'metadata_{}.t7'.format(self.split)))

        train_data, val_data, test_data = [], [], []

        for instance in data:
            image, attr, obj, settype = instance['image'], instance['attr'], \
                instance['obj'], instance['set']
            intab = r'[?*|:><]'
            image = re.sub(intab, "", image)
            if 'cgqa' in self.root:
                image = image
            else:
                image_ = image.split('/')
                image = os.path.join(image_[0], image_[1])
            curr_data = [image, attr, obj]

            if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
                # Skip incomplete pairs, unknown pairs and unknown set
                continue

            if settype == 'train':
                train_data.append(curr_data)
            elif settype == 'val':
                val_data.append(curr_data)
            else:
                test_data.append(curr_data)
        return train_data, val_data, test_data

    def get_dict_data(self, data, pairs):
        data_dict = {}
        for current in pairs:
            data_dict[current] = []

        for current in data:
            image, attr, obj = current
            data_dict[(attr, obj)].append(image)
        
        return data_dict


    def reset_dropout(self):
        ''' 
        Helper function to sample new subset of data containing a subset of pairs of objs and attrs
        '''
        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        # Using sampling from random instead of 2 step numpy
        n_pairs = int((1 - self.pair_dropout) * len(self.train_pairs))

        self.sample_pairs = random.sample(self.train_pairs, n_pairs)
        print('Sampled new subset')
        print('Using {} pairs out of {} pairs right now'.format(
            n_pairs, len(self.train_pairs)))

        self.sample_indices = [ i for i in range(len(self.data))
            if (self.data[i][1], self.data[i][2]) in self.sample_pairs
        ]
        print('Using {} images out of {} images right now'.format(
            len(self.sample_indices), len(self.data)))

    def sample_negative(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Returns
            Tuple of a different attribute, object indexes
        '''
        new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]

        while new_attr == attr and new_obj == obj:
            new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]
        
        return (self.attr2idx[new_attr], self.obj2idx[new_obj])

    def sample_affordance(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object
        '''
        new_attr = np.random.choice(self.obj_affordance[obj])
        
        while new_attr == attr:
            new_attr = np.random.choice(self.obj_affordance[obj])
        
        return self.attr2idx[new_attr]

    def sample_train_affordance(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object from the training pairs
        '''
        new_attr = np.random.choice(self.train_obj_affordance[obj])
        
        while new_attr == attr:
            new_attr = np.random.choice(self.train_obj_affordance[obj])
        
        return self.attr2idx[new_attr]

    def generate_features(self, out_file, model):
        '''
        Inputs
            out_file: Path to save features
            model: String of extraction model
        '''
        # data = self.all_data
        data = ospj(self.root, 'images')
        files_before = glob(ospj(data, '**', '*.jpg'), recursive=True)
        files_all = []
        for current in files_before:
            parts = current.split('\\')
            if "cgqa" in self.root:
                intab = r'[?*|:><]'
                img_root = re.sub(intab, "", parts[-1])
                files_all.append(img_root)
            else:
                intab = r'[?*|:><]'
                img_root = re.sub(intab, "", os.path.join(parts[-2],parts[-1]))
                files_all.append(img_root)
        transform = dataset_transform('test', self.norm_family)
        feat_extractor = get_image_extractor(arch = model).eval()
        feat_extractor = feat_extractor.to(device)

        image_feats_l = []
        image_feats_h = []
        image_files = []
        for chunk in tqdm(
                chunks(files_all, 512), total=len(files_all) // 512, desc=f'Extracting features {model}'):

            files = chunk
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            feats_l, feats_h = feat_extractor(torch.stack(imgs, 0).to(device))

            image_feats_l.append(feats_l.data.cpu())
            image_feats_h.append(feats_h.data.cpu())
            image_files += files

        image_feats_l = torch.cat(image_feats_l, 0)
        image_feats_h = torch.cat(image_feats_h, 0)

        print('features for %d images generated' % (len(image_files)))

        torch.save({'features_l': image_feats_l, 'features_h': image_feats_h, 'files': image_files}, out_file)


    def __getitem__(self, index):
        '''
        Call for getting samples
        '''
        index = self.sample_indices[index]

        image, attr, obj = self.data[index]

        # Decide what to output
        if not self.update_features:

            img_l = self.activations_l[image]
            img_h = self.activations_h[image]

        else:
            img = self.loader(image)
            img = self.transform(img)

        if not self.update_features:
            data = [img_l, img_h, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)], self.sample_mask[index]]
        else:
            data = [img, img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)], self.sample_mask[index]]
        
        if self.phase == 'train':
            all_neg_attrs = []
            all_neg_objs = []

            for curr in range(self.num_negs):
                neg_attr, neg_obj = self.sample_negative(attr, obj) # negative for triplet lose,
                all_neg_attrs.append(neg_attr)
                all_neg_objs.append(neg_obj)

            neg_attr, neg_obj = torch.LongTensor(all_neg_attrs), torch.LongTensor(all_neg_objs)
            
            #note here
            if len(self.train_obj_affordance[obj])>1:
                  inv_attr = self.sample_train_affordance(attr, obj) # attribute for inverse regularizer
            else:
                  inv_attr = (all_neg_attrs[0]) 

            comm_attr = self.sample_affordance(inv_attr, obj) # attribute for commutative regularizer
            

            data += [neg_attr, neg_obj, inv_attr, comm_attr]

        # Return image paths if requested as the last element of the list
        if self.return_images and self.phase != 'train':
            data.append(image)

        return data
    
    def __len__(self):
        '''
        Call for length
        '''
        return len(self.sample_indices)
