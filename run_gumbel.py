#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import DataLoader
import KCA_model

from util import override_config,save_model,read_triple,set_logger,log_metrics,plot_config
from dataloader import TrainDataset, TestDataset, BidirectionalOneShotIterator,Emb_MKG_WY,Emb_MMKB_DB15K




def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--perf_file', type=str, default=None)

    parser.add_argument('--data_path', type=str, default='data/MKG-W')
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default='./models', type=str)

    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=3, type=int)
    parser.add_argument('-d', '--hidden_dim', default=200, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-kca_lr', '--kca_learning_rate', default=0.001, type=float)
    parser.add_argument('--sample_method', default='gumbel',choices=['uni','gumbel'],type=str)
    parser.add_argument('--pre_sample_num', default=1500,type=int)
    parser.add_argument('--loss_rate', default=100,type=int)
    parser.add_argument('--exploration_temp', default=10,type=int)

    parser.add_argument('-b', '--batch_size', default=256, type=int)
    parser.add_argument('--test_batch_size', default=2, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', default=False,type=bool, 
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-cpu', '--cpu_num', default=3, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    

    return parser.parse_args(args)


def train_step(kge_model, optimizer, positive_sample, negative_sample, subsampling_weight,
         mode, args,device,simil_img, simil_text,simil_t,pre_sample,pos_neg_mask,neg_for_adv):
    kge_model.train()
    optimizer.zero_grad()

    positive_sample = positive_sample.to(device)
    negative_sample = negative_sample.to(device)
    subsampling_weight = subsampling_weight.to(device)

    negative_score = kge_model((positive_sample, negative_sample), mode,'train')  # batch * neg

    if args.sample_method=='uni':
        if args.model in ['RotatE','PairRE']:
            self_adversarial_weight = F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()  # batch * neg
            negative_score = (self_adversarial_weight * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)
    elif args.sample_method=='gumbel':
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
        self_adversarial_weight = F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()  # batch * neg
        negative_score = (self_adversarial_weight * F.logsigmoid(-negative_score)).sum(dim=1)
        
        ####  contrastive loss self_adversarial 
        self_adversarial_weight_for_presample = torch.ones_like(simil_img,device=device,requires_grad=False) /simil_img.shape[-1]  # batchsize x pre_sample
        batch_index = torch.arange(args.batch_size)
        for n in range(neg_for_adv.shape[-1]):
            self_adversarial_weight_for_presample[batch_index,neg_for_adv[:,n]] *= self_adversarial_weight[:,n]

        simil_img = torch.mul( torch.exp(simil_img), self_adversarial_weight_for_presample )  # batchsize x pre_sample
        simil_text = torch.mul( torch.exp(simil_text) , self_adversarial_weight_for_presample )  # batchsize x pre_sample
        simil_t = torch.mul( torch.exp(simil_t) ,self_adversarial_weight_for_presample)   # batchsize x pre_sample

        # pre_sample 
        contra_loss_img = - torch.log( torch.sum(simil_img[pos_neg_mask[:,pre_sample]])  / torch.sum(simil_img[~pos_neg_mask[:,pre_sample]])  )  
        contra_loss_text = - torch.log( torch.sum(simil_text[pos_neg_mask[:,pre_sample]])  / torch.sum(simil_text[~pos_neg_mask[:,pre_sample]])  )  
        contra_loss_t = - torch.log( torch.sum(simil_t[pos_neg_mask[:,pre_sample]])  / torch.sum(simil_t[~pos_neg_mask[:,pre_sample]]))  
            


    positive_score = kge_model(positive_sample,'single','train')
    positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

    if args.uni_weight:
        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
    else:
        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

    if args.sample_method=='uni':
        loss = (positive_sample_loss + negative_sample_loss) / 2 
    else:
        loss = (positive_sample_loss + negative_sample_loss) / 2  + (contra_loss_img+ contra_loss_text+contra_loss_t)  / (3 * args.loss_rate)  

    if args.regularization != 0.0:
        # Use L3 regularization for ComplEx and DistMult
        regularization = args.regularization * (
            kge_model.entity_embedding.norm(p=3) ** 3 +
            kge_model.relation_embedding.norm(p=3).norm(p=3) ** 3
        )
        loss = loss + regularization 
        regularization_log = {'regularization': regularization.item()}
    else:
        regularization_log = {}

    loss.backward()
    optimizer.step()
    log = {
        **regularization_log,
        'positive_sample_loss': positive_sample_loss.item(),
        'negative_sample_loss': negative_sample_loss.item(),
        'loss': loss.item()
    }

    return log

def test_step(kge_model, test_triples, all_true_triples, args,device):
    kge_model.eval()

    test_dataloader_head = DataLoader(
        TestDataset(
            test_triples,
            all_true_triples,
            args.nentity,
            args.nrelation,
            'head-batch'
        ),
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TestDataset.collate_fn
    )

    test_dataloader_tail = DataLoader(
        TestDataset(
            test_triples,
            all_true_triples,
            args.nentity,
            args.nrelation,
            'tail-batch'
        ),
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TestDataset.collate_fn
    )

    test_dataset_list = [test_dataloader_head, test_dataloader_tail]

    logs = []

    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])

    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                
                positive_sample = positive_sample.to(device)
                negative_sample = negative_sample.to(device)
                filter_bias = filter_bias.to(device)

                batch_size = positive_sample.size(0)

                score = kge_model((positive_sample, negative_sample), mode,'test')
                score += filter_bias

                # Explicitly sort all the entities to ensure that there is no test exposure bias
                argsort = torch.argsort(score, dim=1, descending=True)

                if mode == 'head-batch':
                    positive_arg = positive_sample[:, 0]
                elif mode == 'tail-batch':
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError('mode %s not supported' % mode)

                for i in range(batch_size):
                    # Notice that argsort is not ranking
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1

                    # ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0 / ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

    return metrics


def main(args):
    torch.set_num_threads(args.cpu_num)
    device = 'cuda:'+args.gpu if torch.cuda.is_available() else 'cpu'

    if args.perf_file == None:
        args.perf_file = 'results/'+ args.data_path.split('/')[1]  +'-' + args.model + '-' + args.sample_method  + '.txt'
    plot_config(args)
    from KGC_model import KGEModel
        
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    set_logger(args)
    
    ent_set, rel_set = OrderedSet(), OrderedSet()
    for split in ['train', 'test', 'valid']:
        for line in open('{}/{}.txt'.format(args.data_path, split), encoding='utf-8'):
            sub, rel, obj = line.strip().split('\t')
            ent_set.add(sub)
            rel_set.add(rel)
            ent_set.add(obj)

    entity2id = {ent: idx for idx, ent in enumerate(ent_set)}
    relation2id = {rel: idx for idx, rel in enumerate(rel_set)}

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    
    kge_model = KGEModel(
        sample_method = args.sample_method,
        device=device,
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    kge_model = kge_model.to(device)


    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch',args), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn,
            drop_last=True
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch',args), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn,
            drop_last=True
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        if args.sample_method=='uni':
            optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, kge_model.parameters()), lr=current_learning_rate )
        elif args.sample_method=='gumbel':
            if args.data_path == 'data/MMKB-DB15K':
                ent_text_emb, ent_img_emb = Emb_MMKB_DB15K(args,entity2id,device)
            else:
                ent_text_emb, ent_img_emb = Emb_MKG_WY(args,entity2id,device)
            KCA = KCA_model.KCA(args,None, args.nentity,ent_text_emb, ent_img_emb)
            KCA = KCA.to(device)
            optimizer = torch.optim.Adam([{'params':kge_model.parameters(),'lr':current_learning_rate}, 
                                            {'params':KCA.parameters(),'lr':args.kca_learning_rate},
                                            ])
        

        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
        

    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)


        
        training_logs = []


        positive_sample_loss,negative_sample_loss,loss = [],[],[]
        # pre_sample_pro = torch.ones([args.batch_size,args.nentity],device=device)
        for step in range(init_step, args.max_steps):
            
            positive_sample, negative_uniform_sample, subsampling_weight,mask, mode = next(train_iterator)
            
            if args.sample_method=='uni':
                negative_sample = negative_uniform_sample
                log = train_step(kge_model, optimizer, positive_sample, negative_sample, subsampling_weight, mode, args,device,0)

            elif args.sample_method=='gumbel':
                temperature = args.exploration_temp / (1+ torch.log( torch.tensor([step +1],device=device)))

                # pre_sample = torch.multinomial(pre_sample_pro, args.pre_sample_num, replacement=False,device=device)  # B x pre_sample 
                pre_sample = torch.randperm(args.nentity)[:args.pre_sample_num].to(device) # pre_sample_num

                pos_neg_mask = torch.le(mask,0.5) # B x num_entity

                neg_distribution,simil_img, simil_text,simil_t = KCA(kge_model,positive_sample,mode,temperature,pre_sample,pos_neg_mask) # batch * num_entity

                ## mask for filter
                if args.pre_sample_num:
                    neg_distribution = torch.log(neg_distribution) + torch.log(mask.to(device)[:,pre_sample]) 
                else:
                    neg_distribution = torch.log(neg_distribution) + torch.log(mask.to(device))

                neg_all = []
                neg_for_adv = []
                for it in range(args.negative_sample_size):
                    neg_onehot = F.gumbel_softmax(neg_distribution,tau=1,hard=True,dim=1)  # batch * presample_entity（one hot）
                    neg_all.append(neg_onehot)
                    neg_for_adv.append(torch.argmax(neg_onehot, dim=1,keepdim=True))
                    # sampling without replacement
                    neg_distribution[neg_onehot.bool()] += torch.log(torch.tensor([1e-38],device=device))
                
                neg_all = torch.stack(neg_all,dim=1) # batch * neg_num * presample_entity (one-hot)                
                neg_for_adv = torch.cat(neg_for_adv,dim=1) # batch * neg_num 

                if args.pre_sample_num:
                    neg_emb = torch.matmul(neg_all,kge_model.entity_embedding[pre_sample]) # batch * neg_num * ent_dim 
                else:
                    neg_emb = torch.matmul(neg_all,kge_model.entity_embedding) # batch * neg_num * ent_dim 

                log = train_step(kge_model, optimizer, positive_sample, neg_emb, subsampling_weight, mode, args,device,simil_img, simil_text,simil_t,pre_sample,pos_neg_mask,neg_for_adv)


            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                # optimizer = torch.optim.Adam(
                #     #filter(lambda p: p.requires_grad, kge_model.parameters()), 
                #     lr=current_learning_rate
                # )
                warm_up_steps = warm_up_steps * 3
            
            if step!= 0 and step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
                
            if step!= 0 and step % 1000 == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                # log_metrics('Training average', step, metrics)
                for metric in metrics:
                    logging.info('Training average %s at step %d: %f' % (metric, step, metrics[metric]))
                training_logs = []
                positive_sample_loss.append(metrics['positive_sample_loss'])
                negative_sample_loss.append(metrics['negative_sample_loss'])
                loss.append(metrics['loss'])

            if step!= 0 and args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = test_step(kge_model, valid_triples, all_true_triples, args,device)
                a = log_metrics('Valid', step, metrics)
                with open(args.perf_file, 'a') as f:
                    f.write(' Valid at step %d: %.4f|%.2f|%.4f|%.4f|%.4f|\n' %( step, a[0]*100,a[1],a[2]*100,a[3]*100,a[4]*100))
                
                logging.info('Evaluating on Test Dataset...')
                metrics = test_step(kge_model, test_triples, all_true_triples, args,device)
                a = log_metrics('Test', step, metrics)
                with open(args.perf_file, 'a') as f:
                    f.write('Test at step %d: %.4f|%.2f|%.4f|%.4f|%.4f|\n' %( step, a[0]*100,a[1],a[2]*100,a[3]*100,a[4]*100))
            

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = test_step(kge_model, valid_triples, all_true_triples, args,device)
        a = log_metrics('Valid', step, metrics)
        with open(args.perf_file, 'a') as f:
            f.write(' Valid at step %d: %.4f|%.2f|%.4f|%.4f|%.4f|\n' %( step, a[0]*100,a[1],a[2]*100,a[3]*100,a[4]*100))
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = test_step(kge_model, test_triples, all_true_triples, args,device)
        a = log_metrics('Test', step, metrics)
        with open(args.perf_file, 'a') as f:
            f.write('Test at step %d: %.4f|%.2f|%.4f|%.4f|%.4f|\n' %( step, a[0]*100,a[1],a[2]*100,a[3]*100,a[4]*100))

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = test_step(kge_model, train_triples, all_true_triples, args,device)
        log_metrics('Train', step, metrics)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    setup_seed(42)
    main(parse_args())

