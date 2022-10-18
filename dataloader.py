#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode,args):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)


        self.args = args

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        positive_sample = torch.LongTensor(positive_sample)


        if self.args.sample_method=='uni':
            negative_sample_list = []
            negative_sample_size = 0
            while negative_sample_size < self.negative_sample_size:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
                if self.mode == 'head-batch':
                    mask = np.in1d(
                        negative_sample, 
                        self.true_head[(relation, tail)], 
                        assume_unique=True, 
                        invert=True
                    )
                elif self.mode == 'tail-batch':
                    mask = np.in1d(
                        negative_sample, 
                        self.true_tail[(head, relation)], 
                        assume_unique=True, 
                        invert=True
                    )
                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size

            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

            negative_sample = torch.LongTensor(negative_sample)

        elif self.args.sample_method=='gumbel':
            mask = torch.ones([self.nentity], dtype=torch.float32,requires_grad=False) 
            if self.mode == 'head-batch':
                label = self.true_head[(relation, tail)] 
                mask[label] = 1e-38
            elif self.mode == 'tail-batch':
                label = self.true_tail[(head, relation)]
                mask[label] = 1e-38
            negative_sample = None


            ##################
        return positive_sample, negative_sample, subsampling_weight, mask,self.mode,self.args.sample_method
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        if data[0][5]=='uni':
            negative_sample = torch.stack([_[1] for _ in data], dim=0)
            mask = None

        elif data[0][5]=='gumbel':
            mask = torch.stack([_[3] for _ in data], dim=0)
            negative_sample = None
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        
        mode = data[0][4]
        return positive_sample, negative_sample, subsample_weight,mask, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))


        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data



def Emb_MKG_WY(args,ent2id,device):
    import h5py
    import numpy as np
    ent_text_emb = torch.zeros([len(ent2id), 4, 384], device=device)
    ent_img_emb = torch.zeros([len(ent2id),24, 383], device=device)
    if args.data_path=='data/MKG-W' :
        text_path =  "data/MKG_W_description_sentences.h5"
        image_path = "data/MKG_W_img_BEIT_16-224.h5"
    elif args.data_path=='data/MKG-Y':
        text_path =  "data/MKG_Y_description_sentences.h5"
        image_path = "data/MKG_Y_img_BEIT_16-224.h5"

    ent_link = {}
    if args.data_path=='data/MKG-W':
        ent_link_path = 'data/MKG-W/ent_links'
    elif args.data_path=='data/MKG-Y':
        ent_link_path = 'data/MKG-Y/ent_links'

    with open(ent_link_path) as fin:
        for line in fin:
            D,W = line.strip().split('\t')
            ent_link[D] = W

    text_count = 0
    with h5py.File(text_path, 'r') as f:
        for k in f.keys():
            v = np.array(f[k])
            sentence_num = v.shape[0]
            try:
                if sentence_num >=4:
                    ent_text_emb[ent2id[ent_link['http://dbpedia.org/resource/'+k]]] = torch.from_numpy(v[:4])
                else:
                    ent_text_emb[ent2id[ent_link['http://dbpedia.org/resource/'+k]]][:sentence_num] = torch.from_numpy(v)
            except KeyError:
                text_count += 1

    image_count = 0
    with h5py.File(image_path, 'r') as f:
        for k in f.keys():
            v = np.array(f[k])
            try:
                ent_img_emb[ent2id[ent_link['http://dbpedia.org/resource/'+k]]] = torch.from_numpy(v)
            except KeyError:
                image_count += 1

    print('Multimodal data loaded')
    return ent_text_emb, ent_img_emb


def Emb_MMKB_DB15K(args,ent2id,device):
    import h5py
    import numpy as np
    ent_text_emb = torch.zeros([len(ent2id), 4, 384], device=device)
    ent_img_emb = torch.zeros([len(ent2id),24, 383], device=device)

    text_path =  "data/MMKB_description_sentences.h5"
    image_path = "data/MMKB_img_BEIT_16-224.h5"

    link_FB_DB = {}
    ent_link_path_DB = 'data/MMKB-DB15K/DB15K_SameAsLink.txt'

    with open(ent_link_path_DB) as fin:
        for line in fin:
            F,r,D,_ = line.strip().split(' ')
            link_FB_DB[F] = D[1:-1]


    text_count = 0
    with h5py.File(text_path, 'r') as f:
        for k in f.keys():
            v = np.array(f[k])
            sentence_num = v.shape[0]
            try:
                name = 'http://dbpedia.org/resource/'+k

                if sentence_num >=4:
                    ent_text_emb[ent2id[name]] = torch.from_numpy(v[:4])
                else:
                    ent_text_emb[ent2id[name]][:sentence_num] = torch.from_numpy(v)            
            except KeyError:
                text_count += 1

    image_count = 0
    with h5py.File(image_path, 'r') as f:
        for k in f.keys():
            v = np.array(f[k])
            try:
                name = link_FB_DB['/m/'+k[2:]]
                ent_img_emb[ent2id[name]] = torch.from_numpy(v)

            except KeyError:
                image_count += 1

    print('Multimodal data loaded')

    return ent_text_emb, ent_img_emb


