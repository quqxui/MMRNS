import json
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader




def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join('case_model', 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    a = []
    for metric in metrics:
        a.append(metrics[metric])
    logging.info('%s at step %d: %.4f|%.2f|%.4f|%.4f|%.4f|' %(mode, step, a[0]*100,a[1],a[2]*100,a[3]*100,a[4]*100))
    return a

def plot_config(args):
    out_str = "\n data_path:{} model:{} n:{} d:{} g:{} a:{} r:{} lr:{} policy_lr:{} sample:{} pre_sample_num:{} loss_rate:{} exploration_temp:{} batch:{}\n".format(
            args.data_path,args.model,args.negative_sample_size,args.hidden_dim,args.gamma,args.adversarial_temperature,
            args.regularization,args.learning_rate,args.kca_learning_rate,args.sample_method,args.pre_sample_num,args.loss_rate,args.exploration_temp,args.batch_size)
    with open(args.perf_file, 'a') as f:
        f.write(out_str)
