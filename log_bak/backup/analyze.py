import json
import os
import random
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from scipy import sparse
from transformers import BertTokenizer, get_linear_schedule_with_warmup, BertConfig, BertForMaskedLM
from dataset import *
from tqdm import tqdm
import logging
import argparse
from utils import TimeIt, random_walk_restart

def get_rel2desc(filename):
    rel2desc = json.load(open('./rel2desc.json'))
    rel2desc = rel2desc[filename.split('/')[-1]]
    return rel2desc


#given single file, construct corresponding graph of terms and its dictionary and query set
def simple_load_data(filename='../data/datasets/cl.obo', use_text_preprocesser = False, return_triples=False):
    """
    args:
        use text preprocesser: decide whether we process the data wtih lowercasing and removing punctuations
    
    returns:
        name_array: array of all the terms' names. no repeated element, in the manner of lexicographic order

        query_id_array: array of (query, id), later we split the query_set into train and test dataset;sorted by ids

        mention2id: map all mentions(names and synonyms of all terms) to ids, the name and synonyms with same term have the same id
         
        graph

    
    some basic process rules:
    1.To avoid overlapping, we just abandon the synonyms which are totally same as their names
    2. Considering that some names appear twice or more, We abandon correspoding synonyms
    3. Some synonyms have more than one corresponding term, we just take the first time counts
    """
    text_processer = TextPreprocess() 
    name_list = [] #record of all terms, rememeber some elements are repeated
    name_array = []
    query_id_array = []
    mention2id = {}
    
    edges=[] 
    triples = []

    with open(file=filename, mode='r', encoding='utf-8') as f:
        check_new_term = False
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line[:6]=='[Term]':#starts with a [Term] 
                check_new_term = True
                continue
            if line[:1]=='\n':#ends with a '\n'
                check_new_term = False
                continue
            if check_new_term == True:
                if line[:5]=='name:':
                    name_list.append(text_processer.run(line[6:-1])if use_text_preprocesser else line[6:-1])
        
        name_count = {}
        
        #record the count of names in raw file
        for i, name in enumerate(name_list):
            name_count[name] = name_list.count(name)
        
        #build a mapping function of name2id, considering that some names appear twice or more, we remove the duplication and sort them
        name_array = sorted(list(set(name_list)))

        for i, name in enumerate(name_array):
            mention2id[name] = i
        
        #temporary variables for every term
        #construct a scipy csr matrix of edges and collect synonym pairs
        check_new_term = False
        check_new_name = False#remember that not every term has a name and we just take the terms with name count. Good news: names' locations are relatively above
        name = ""
        iter_name = iter(name_list)

        for i, line in enumerate(lines):
            if line[:6]=='[Term]':#starts with a [Term] and ends with an '\n'
                check_new_term = True
                continue
            if line[:5]=='name:':
                check_new_name = True
                if check_new_term == True:
                    name = next(iter_name)
                continue
            if line[:1]=='\n':# signal the end of current term
                check_new_term = False
                check_new_name = False
                continue

            if check_new_term == True and check_new_name == True:
                #construct term graph
                if line[:5]=='is_a:':
                    entry = line.split(" ")
                    if '!' in entry:# some father_nodes are not divided by '!' and we abandon them
                        father_node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if father_node in name_array:#some father_nodes are not in concepts_list, and we abandon them.
                            edges.append((mention2id[father_node], mention2id[name]))
                if line[:16]=='intersection_of:':
                    entry = line.split(" ")
                    if '!' in entry:# some father_nodes are not divided by '!' and we abandon them
                        brother_node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if brother_node in name_array:#some father_nodes are not in concepts_list, and we abandon them.
                            edges.append((mention2id[brother_node], mention2id[name]))
                
                if line[:13]=='relationship:':
                    entry = line.split(" ")
                    if '!' in entry:# some father_nodes are not divided by '!' and we abandon them
                        brother_node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if brother_node in name_array:#some father_nodes are not in concepts_list, and we abandon them.
                            edges.append((mention2id[brother_node], mention2id[name]))
                
                # collect synonyms and to dictionary set and query set
                if line[:8]=='synonym:' and name_count[name] == 1: #anandon the situations that name appears more than once
                    start_pos = line.index("\"") + 1
                    end_pos = line[start_pos:].index("\"") + start_pos
                    synonym = text_processer.run(line[start_pos:end_pos]) if use_text_preprocesser else line[start_pos:end_pos]
                    if synonym==name:continue#filter these mentions that are literally equal to the node's name, make sure there is no verlap
                    if synonym in mention2id.keys():continue# only take the first time synonyms appears counts
                    id = mention2id[name]
                    mention2id[synonym] = id
                    query_id_array.append((synonym, id))

                rel2desc = get_rel2desc(filename)
                for r in rel2desc:
                    if re.match('^[^:]+: {} '.format(r), line):
                        if '!' in entry:
                            node = " ".join(entry[entry.index('!') + 1:])[:-1]
                            if node in mention2id:
                                triples.append((mention2id[name], r, mention2id[node]))
                if re.match('^is_a: ', line):
                    if '!' in entry:
                        node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if node in mention2id:
                            triples.append((mention2id[name], 'is_a', mention2id[node]))
        
        query_id_array = sorted(query_id_array, key = lambda x:x[1])
        triples = sorted(list(set(triples)))
        
        print('mention num', len(mention2id.items()))
        print('names num', len(name_array))
        print('query num', len(query_id_array))
        print('edge num', len(list(set(edges))))
        print('triple num', len(triples))
        
        values=[1]*(2*len(edges))
        rows = [i for (i, j) in edges] + [j for (i, j) in edges]# construct undirected graph
        cols = [j for (i, j) in edges] + [i for (i, j) in edges]
        edge_index = torch.LongTensor([rows, cols])# undirected graph edge index
        # graph = sparse.coo_matrix((values, (rows, cols)), shape = (len(name_array), len(name_array)))
        # n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        # #print(n_components)

        ret = np.array(name_array), np.array(query_id_array), mention2id, edge_index, triples
        if return_triples:
            return ret
        else:
            return ret[:-1]

class SimpleModel(nn.Module):
    def __init__(self, model_path, ent_num, args):
        super().__init__()
        self.args = args
        self.bert_lm = BertForMaskedLM.from_pretrained(model_path)
        config = self.bert_lm.config
        self.ent_vocab = nn.Embedding(ent_num, config.hidden_size)
        
        self._cls_bn = nn.BatchNorm1d(num_features=1)
    
    def cls_bn(self, x):
        return self._cls_bn(x.unsqueeze(1)).squeeze(1)
    
    def get_ent_logits(self, last_hidden_state, ent_emb=None):
        if ent_emb is None:
            ent_emb = self.ent_vocab.weight

        hidden_state = self.bert_lm.cls.predictions.transform(last_hidden_state)
        return self.cls_bn(hidden_state.matmul(ent_emb.T))


class SimpleTrainer():
    def __init__(self, args):
        self.args = args
        self.filename = self.args['filename']
        self.use_text_preprocesser = self.args['use_text_preprocesser']
        self.name_array, query_id_array, self.mention2id, self.edge_index, self.triples = simple_load_data(self.filename, self.use_text_preprocesser, return_triples=True)#load data
        self.queries_train, self.queries_valid, self.queries_test = data_split(query_id_array=query_id_array, is_unseen=self.args['is_unseen'], test_size=0.33)# data split
        self.tokenizer = BertTokenizer.from_pretrained(self.args['pretrained_model'])
        self.model = SimpleModel(self.args['pretrained_model'], len(self.name_array), args)
        self.model.cuda()

        torch.save(self.triples, os.path.join(self.args['exp_path'], 'triples.bin'))

        self.train_entity_set = set([self.mention2id[i] for i in self.queries_train])
        from collections import defaultdict
        id2mention = defaultdict(list)
        for k, v in self.mention2id.items():
            id2mention[v].append(k)
        self.id2mention = dict(id2mention)

        self.args['logger'].info('get dataset')
        if 'DEBUG_LOAD_DATASET' in os.environ:
            self.train_dataset = torch.load('/tmp/train_dataset_anal')
            self.valid_dataset = torch.load('/tmp/valid_dataset_anal')
            self.test_dataset = torch.load('/tmp/test_dataset_anal')
        else:
            self.train_dataset = self.get_dataset(query_array=self.queries_train, triples=self.triples)
            self.valid_dataset = self.get_dataset(query_array=self.queries_valid, triples=[])
            self.test_dataset = self.get_dataset(query_array=self.queries_test, triples=[])
            torch.save(self.train_dataset, '/tmp/train_dataset_anal')
            torch.save(self.valid_dataset, '/tmp/valid_dataset_anal')
            torch.save(self.test_dataset, '/tmp/test_dataset_anal')
        self.args['logger'].info('dataset done')

        self.model.ent_vocab.weight.data = self.get_ent_embeddings()

        # if self.args['use_rwr']:
        #     N = len(self.name_array)
        #     A = np.zeros((N, N))
        #     for h, r, t in self.triples:
        #         if r == 'is_a':
        #             A[h, t] = A[t, h] = 1
        #     self.get_random_walk_prob(A, self.args['rwr_reset_prob'])
        #     self.get_label2triples(self.train_dataset[2])

    def get_dataset(self, query_array, triples):
        entity_set = set([self.mention2id[i] for i in query_array])
        depth = self.args['path_depth']
        
        from collections import defaultdict
        id2mention = defaultdict(list)
        for k, v in self.mention2id.items():
            id2mention[v].append(k)
        id2mention = dict(id2mention)

        id2is_a = defaultdict(list)
        for h,r,t in triples:
            if r == 'is_a':
                id2is_a[h].append(t)
        id2is_a = dict(id2is_a)
        def lookup_is_a(id_):
            if id_ in id2is_a:
                return np.random.choice(id2is_a[id_])
            else:
                return -100
        def get_is_a_seq(id_, depth=2):
            x = (id_, )
            for i in range(depth-1):
                x += (lookup_is_a(x[-1]), )
            return x

        pack = []

        path_template = ', which is a kind of {}'

        # synonyms
        inputs = []
        labels = []
        template = '{} is identical with {}'
        for q in query_array:
            inputs.append(template.format('[MASK]', q) + path_template.format('[MASK]') * (depth - 1) )
            labels.append(get_is_a_seq(self.mention2id[q], depth))
        pack.append((inputs, labels))

        # entity name
        inputs = []
        labels = []
        template = '{} is identical with {}'
        for e in self.name_array:
            inputs.append(template.format('[MASK]', e) + path_template.format('[MASK]') * (depth - 1) )
            labels.append(get_is_a_seq(self.mention2id[e], depth))
        pack.append((inputs, labels))
        
        # triples
        rel2desc = get_rel2desc(self.args['filename'])
        inputs = []
        labels = []
        template = '{} {} {}'
        def get_name(idx):
            if idx in entity_set:
                return np.random.choice(id2mention[idx])
            else:
                return self.name_array[idx]
        for h, r, t in triples:
            inputs.append(template.format(get_name(h), rel2desc[r], '[MASK]')  + path_template.format('[MASK]') * (depth - 1) )
            labels.append(get_is_a_seq(t, depth))
            inputs.append(template.format('[MASK]', rel2desc[r], get_name(t))  + path_template.format('[MASK]') * (depth - 1) )
            labels.append((h, ) + get_is_a_seq(t, depth)[1:])
        pack.append((inputs, labels))

        new_pack = []
        for inputs, labels in pack:
            if len(inputs) == len(labels) == 0:
                new_pack.append(None)
                continue
            input_ids, attention_mask = self.tokenize(inputs, self.args['max_seq_len'])
            labels = torch.LongTensor(labels)
            new_pack.append((input_ids, attention_mask, labels))

        synonym_dataset = TensorDataset(*new_pack[0])
        if new_pack[1] is None:
            entity_dataset = None
        else:
            entity_dataset = TensorDataset(*new_pack[1])
        if new_pack[2] is None:
            triple_dataset = None
        else:
            triple_dataset = TensorDataset(*new_pack[2])
        return synonym_dataset, entity_dataset, triple_dataset

    def tokenize(self, str_list, max_length):
        ret = self.tokenizer(str_list, add_special_tokens=True, 
            max_length = max_length, padding='max_length',
            truncation=True, return_attention_mask = True,
            return_tensors='pt')
        return ret.input_ids, ret.attention_mask

    def save(self, path=None):
        if path is None:
            path = self.args['exp_path']
        model_path = os.path.join(path, 'pytorch_model.bin')
        torch.save(self.model.state_dict(), model_path)
        self.args['logger'].info('model saved at %s'%path)
        
    def load(self, path=None):
        if path is None:
            path = self.args['exp_path']
        model_path = os.path.join(path, 'pytorch_model.bin')
        self.model.load_state_dict(torch.load(model_path))
        self.args['logger'].info('model loaded from %s'%path)

    def train_step(self, input_ids, attention_mask, labels, optimizer, scheduler, criterion, ent_emb=None):
        self.model.train()
        batch_size, seq_len = input_ids.shape
        
        optimizer.zero_grad()
        outputs = self.model.bert_lm.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            ).last_hidden_state

        interest = outputs[input_ids == self.tokenizer.mask_token_id]
        #interest = interest.reshape(batch_size, -1)
        scores = self.model.get_ent_logits(interest, ent_emb)
        
        # labels = labels[(input_ids == self.tokenizer.mask_token_id).any(dim=1)] # avoid trunc out
        # avoid trunc out:
        labels, labels_bak = [], labels
        for i in range(batch_size):
            n_mask = (input_ids[i] == self.tokenizer.mask_token_id).long().sum()
            labels.append(labels_bak[i][:n_mask])
        labels = torch.cat(labels, dim=0)
    
        loss = criterion(scores, labels)

        loss.backward()
        if self.args['max_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        return outputs, scores, loss

    def train(self):
        self.args['logger'].info('train')
        
        train_loader_syn = DataLoader(dataset=self.train_dataset[0], batch_size=self.args['batch_size'], shuffle=True)
        train_loader_ent = DataLoader(dataset=self.train_dataset[1], batch_size=self.args['batch_size'], shuffle=True)
        train_loader_hrt = DataLoader(dataset=self.train_dataset[2], batch_size=self.args['batch_size'], shuffle=True)
        syn_size = len(train_loader_syn)
        
        def make_infinite_dataloader(dataloader):
            while True:
                for i in dataloader:
                    yield i
        train_loader_ent = make_infinite_dataloader(train_loader_ent)
        train_loader_hrt = make_infinite_dataloader(train_loader_hrt)
        # train_loader_sib = make_infinite_dataloader(train_loader_sib)
        # train_loader_grand = make_infinite_dataloader(train_loader_grand)

        print('syn dataset: ', len(self.train_dataset[0]))
        print('ent dataset: ', len(self.train_dataset[1]))
        print('hrt dataset: ', len(self.train_dataset[2]))
        import sys; sys.stdout.flush()

        criterion = nn.CrossEntropyLoss(reduction='mean')        
        optimizer = torch.optim.Adam(
                # self.model.parameters(),
                [{'params': self.model.bert_lm.parameters()},
                {'params': self.model.ent_vocab.parameters(),'lr':1e-3,'weight_decay':0},
                {'params': self.model._cls_bn.parameters(),'lr':1e-3,'weight_decay':0}],
                lr=self.args['lr'], weight_decay=self.args['weight_decay']
            )

        loader_selector = ([0] * syn_size
                          +[1] * int(syn_size / self.args['syn_ratio'] * self.args['ent_ratio'])
                          +[2] * int(syn_size / self.args['syn_ratio'] * self.args['hrt_ratio'])
                           )
        loaders = [train_loader_syn, train_loader_ent, train_loader_hrt, 
                   ]
        
        pbar = tqdm(range(self.args['pretrain_emb_iter']), desc='pretrain emb')
        for iteration in pbar:
            if self.args['use_get_ent_emb'] and iteration % 100 == 0:
                ent_emb = self.get_ent_embeddings()
            else:
                ent_emb = None
            task = 1
            loader = loaders[task]
            batch = next(loader)

            batch = (i.cuda() for i in batch)
            input_ids, attention_mask, labels = batch
            outputs, interest, loss = self.train_step(input_ids, attention_mask, labels, optimizer, None, criterion, ent_emb)
            m, M, mean, std = interest.min(), interest.max(), interest.mean(), interest.std()
            pbar.set_postfix({("loss%d"%task): float(loss),
                               "[min, max, mean, std]": ['%.2e'%i for i in [m, M, mean, std]],
                               "lr":['%.2e'%group["lr"] for group in optimizer.param_groups]})
           
            if 'DEBUG_DECODE_OUTPUT' in os.environ and iteration % 100 == 0:
                    print(self.tokenizer.batch_decode(input_ids[:10], skip_special_tokens=True))
                    print([self.name_array[i] for i in interest[:10].argmax(dim=-1).tolist()])
                    print([self.name_array[i] for i in labels[:10].tolist()])
                    import sys; sys.stdout.flush()

        t_total = self.args['epoch_num'] * len(loader_selector)
        if self.args['use_scheduler']:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
        else:
            scheduler = None

        for epoch in range(1, self.args['epoch_num'] + 1):
            torch.cuda.empty_cache()
            self.model.train()
            np.random.shuffle(loader_selector)
            selector_idx = 0
            loss_sum = 0
            pbar = tqdm(enumerate(train_loader_syn), total=len(train_loader_syn))
            for iteration, syn_batch in pbar:
                if self.args['use_get_ent_emb'] and iteration % 100 == 0:
                    ent_emb = self.get_ent_embeddings()
                else:
                    ent_emb = None
                while loader_selector[selector_idx] != 0:
                    task = loader_selector[selector_idx]
                    loader = loaders[task]
                    batch = next(loader)

                    batch = (i.cuda() for i in batch)
                    input_ids, attention_mask, labels = batch
                    outputs, interest, loss = self.train_step(input_ids, attention_mask, labels, optimizer, scheduler, criterion, ent_emb)
                    loss_sum += loss.item() * len(input_ids)
                    m, M, mean, std = interest.min(), interest.max(), interest.mean(), interest.std()
                    pbar.set_postfix({("loss%d"%task): float(loss), 
                                        "[min, max, mean, std]": ['%.2e'%i for i in [m, M, mean, std]], 
                                        "lr":['%.2e'%group["lr"] for group in optimizer.param_groups]})
                    if iteration % 100 == 0:
                        print(self.tokenizer.batch_decode(input_ids[:2], skip_special_tokens=True))
                        print([self.name_array[i] for i in interest[:4].argmax(dim=-1).tolist()])
                        print([self.name_array[i] for i in labels[:2].tolist()])
                        import sys; sys.stdout.flush()
                    selector_idx += 1

                task = loader_selector[selector_idx]
                batch = syn_batch

                batch = (i.cuda() for i in batch)
                input_ids, attention_mask, labels = batch
                outputs, interest, loss = self.train_step(input_ids, attention_mask, labels, optimizer, scheduler, criterion, ent_emb)
                loss_sum += loss.item() * len(input_ids)
                m, M, mean, std = interest.min(), interest.max(), interest.mean(), interest.std()
                pbar.set_postfix({("loss%d"%task): float(loss), 
                                "[min, max, mean, std]": ['%.2e'%i for i in [m, M, mean, std]], 
                                "lr":['%.2e'%group["lr"] for group in optimizer.param_groups]})
                selector_idx += 1

                if 'DEBUG_DECODE_OUTPUT' in os.environ and iteration % 100 == 0:
                    print(self.tokenizer.batch_decode(input_ids[:10], skip_special_tokens=True))
                    print([self.name_array[i] for i in interest[:10].argmax(dim=-1).tolist()])
                    print([self.name_array[i] for i in labels[:10].tolist()])
                    import sys; sys.stdout.flush()

            # print('train')
            # accu_1, accu_k = self.eval(self.train_dataset, epoch=epoch)
            # print('valid')
            accu_1, accu_k = self.eval(self.valid_dataset, epoch=epoch)
            loss_sum/=len(loader_selector)
            print('loss_sum:', float(loss_sum))
            import sys; sys.stdout.flush()
    
    @torch.no_grad()
    def get_ent_embeddings(self, ):
        ent_embeddings = torch.zeros_like(self.model.ent_vocab.weight)
        data_loader = DataLoader(dataset=self.train_dataset[1],batch_size=64)

        for i, batch in enumerate(data_loader):
            (input_ids, attention_mask, labels) = (i.cuda() for i in batch)
            outputs = self.model.bert_lm.bert(
                input_ids = input_ids,
                attention_mask = attention_mask,
                ).last_hidden_state
            assert (input_ids[:, 1] == self.tokenizer.mask_token_id).all()
            emb = self.model.bert_lm.cls.predictions.transform(outputs[:, 1])
            ent_embeddings[labels[:, 0]] = emb
            
        return ent_embeddings
        

    @torch.no_grad()
    def eval(self, eval_dataset, epoch):
        self.model.eval()
        assert len(eval_dataset) == 3
        eval_dataset = eval_dataset[0]
        eval_loader = DataLoader(dataset=eval_dataset, batch_size = 64, shuffle=False)

        pack = [[], [], []]

        accu_1 = torch.FloatTensor([0]).cuda()
        accu_k = torch.FloatTensor([0]).cuda()

        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader))
        if self.args['use_get_ent_emb']:
                ent_emb = self.get_ent_embeddings()
        else:
            ent_emb = None
        for iteration, batch in pbar:
            batch = (i.cuda() for i in batch)
            input_ids, attention_mask, labels = batch
            batch_size, seq_len = input_ids.shape
            
            outputs = self.model.bert_lm.bert(
                input_ids = input_ids,
                attention_mask = attention_mask,
                ).last_hidden_state

            assert (input_ids[:, 1] == self.tokenizer.mask_token_id).all()
            interest = outputs[:, 1]
            # interest = outputs[input_ids == self.tokenizer.mask_token_id]
            #interest = interest.reshape(batch_size, -1)
            scores = self.model.get_ent_logits(interest, ent_emb)
            
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            pack[0].append(sorted_scores.clone().detach()[:, :100].cpu())
            pack[1].append(sorted_indices.clone().detach()[:,:100].cpu())
            pack[2].append(labels.clone().detach().cpu())
            # labels = labels[(input_ids == self.tokenizer.mask_token_id).any(dim=1)] # avoid trunc out
            labels = labels[(input_ids == self.tokenizer.mask_token_id).any(dim=1)][:, 0] # avoid trunc out
            accu_1 += (sorted_indices[:, 0]==labels).sum() / len(eval_dataset)
            accu_k += (sorted_indices[:, :self.args['eval_k']]== labels.unsqueeze(dim=1)).sum() / len(eval_dataset)

            if 'DEBUG_DECODE_EVAL' in os.environ and iteration % 100 == 0:
                print(self.tokenizer.decode(input_ids[0]))
                print(sorted_scores[:10])
                print(sorted_indices[:10])
                import sys; sys.stdout.flush()

        self.args['logger'].info("epoch %d done, accu_1 = %f, accu_%d = %f"%(epoch, float(accu_1), self.args['eval_k'], float(accu_k)))
        if epoch == -1:
            pack = [torch.cat(i, dim=0) for i in pack]
            pack = {'labels':pack[-1], 'scores':pack[0], 'idx':pack[1]}
            torch.save(pack, os.path.join(self.args['exp_path'], 'pack.bin'))
        if accu_1 < 1e-4 and epoch != 0:
            print('bad args!')
            exit(-1)
        return accu_1, accu_k


#set up seed         
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def get_nei(triples, max_length, ent_total):
    from collections import defaultdict
    import copy
    neis = [{} for i in range(max_length+1)] # neis[i] stores i-hop neighbors

    for i in range(ent_total):
        neis[1][i] = set()
    for h, r, t in triples:
        neis[1][h].add(t)
        neis[1][t].add(h)
    
    for length in range(2, max_length+1):
        nei_1 = neis[1]
        nei_last = neis[length-1]
        nei = neis[length]
        for center in range(ent_total):
            nei[center] = copy.deepcopy(nei_last[center])
            for i in nei_last[center]:
                nei[center] = nei[center].union(nei_1[i])
    return neis
        

# set up logger
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    if not os.path.exists(name):
        os.makedirs(name)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='../data/datasets/cl.obo')
    parser.add_argument('--use_text_preprocesser', action='store_true', default=False)
    parser.add_argument('--is_unseen', action='store_true', default=False)
    parser.add_argument('--pretrained_model', type=str, default='dmis-lab/biobert-base-cased-v1.2')
    parser.add_argument('--exp_path', type=str, default='../exp/simple/')

    parser.add_argument('--max_seq_len', type=int, default=60)
    
    parser.add_argument('--eval_k', type=int, default=10)

    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--syn_ratio', type=float, default=0.5)
    parser.add_argument('--ent_ratio', type=float, default=0.25)
    parser.add_argument('--hrt_ratio', type=float, default=0.25)
    parser.add_argument('--path_depth', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_scheduler', action='store_true', default=False)
    parser.add_argument('--emb_init_std', type=float, default=1e-3)
    parser.add_argument('--use_get_ent_emb', action='store_true', default=False)
    parser.add_argument('--pretrain_emb_iter', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=0) # default: no max grad norm
    
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    args = args.__dict__

    logger = setup_logger(name=args['exp_path'], log_file=os.path.join(args['exp_path'], 'log.log'))
    args['logger'] = logger
    print(args)
    import sys; sys.stdout.flush()

    assert args['hrt_ratio'] > 0 or args['path_depth'] == 1 and args['hrt_ratio'] == 0

    setup_seed(args['seed'])

    if True:
        b=SimpleTrainer(args)
        b.load()
        # b.eval(b.valid_dataset, epoch=0)
        b.eval(b.test_dataset, epoch=-1)

        neis = get_nei(b.triples, 5, len(b.name_array))
        
        path = args['exp_path']
        nei_path = os.path.join(path, 'neis.bin')
        name_path = os.path.join(path, 'name.bin')
        torch.save(neis, nei_path)
        torch.save(b.name_array, name_path)
        
        exit(0)

        with torch.no_grad():
            b.model.eval()
            if args['use_get_ent_emb']:
                ent_emb = b.get_ent_embeddings().cpu()
            else:
                ent_emb = b.model.ent_vocab.weight.clone().detach().cpu()
        path = args['exp_path']
        ent_path = os.path.join(path, 'ent_emb.bin')
        torch.save(ent_emb, ent_path)
    else:
        path = args['exp_path']
        ent_path = os.path.join(path, 'ent_emb.bin')
        ent_emb = torch.load(ent_path).cuda()

    if False:
        with torch.no_grad():
            sim_mat = []
            for i in ent_emb:
                i = i.unsqueeze(0)
                sim = torch.cosine_similarity(i, ent_emb)
                # sim = i.matmul(ent_emb.T).softmax(dim=-1)
                # sim = ((i - ent_emb)**2).sum(-1)
                sim_mat.append(sim)

            sim_mat = torch.stack(sim_mat, dim=0).squeeze().cpu()

            path = args['exp_path']
            sim_path = os.path.join(path, 'sim_mat.bin')
            torch.save(sim_mat, sim_path)
    else:
        path = args['exp_path']
        sim_path = os.path.join(path, 'sim_mat.bin')
        sim_mat = torch.load(sim_path)

    if True:
        values = []
        for length in range(1, 6):
            idx0, idx1 = [], []
            for i in neis[length]:
                tmp = list(neis[length][i])
                idx0 += [i]*len(tmp)
                idx1 += tmp
            v = sim_mat[idx0, idx1]
            values.append(v)
        torch.save(values, 'values.bin')
    else:
        values = torch.load('values.bin')

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.title(args['exp_path'])
    labels = '1','2','3','4','5'
    plt.boxplot(values, labels = labels)
    plt.savefig(os.path.join(args['exp_path'], 'box.png'))
