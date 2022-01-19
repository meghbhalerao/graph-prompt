#%%
import torch
import matplotlib.pyplot as plt
#%%
from transformers import BertTokenizer, get_linear_schedule_with_warmup, BertConfig, BertForMaskedLM
model_path = 'dmis-lab/biobert-base-cased-v1.2'
tokenizer = BertTokenizer.from_pretrained(model_path)
bert_lm = BertForMaskedLM.from_pretrained(model_path)
#%%
# test_data_set = torch.load('/tmp/test_dataset_anal')
a = torch.load('/tmp/pack.bin')
neis = torch.load('../exp/0hp_unseen_path2/neis.bin')
ent_total = len(neis[1])
neis[0] = {e:{e} for e in range(ent_total)}
for i in range(5):
    for j in range(i+1, 6):
        for e in range(ent_total):
            neis[-i-1][e] -= neis[-j-1][e]
name_array = torch.load('../exp/0hp_unseen_path2/name.bin')
triples = torch.load('../exp/0hp_unseen_path2/triples.bin')
# %%
# labels = a['labels'][:, 0]
# N = len(labels)


# results = []
# for depth in range(6):
#     result = 0
#     for i in range(N):
#         topk = a['idx'][i][:10].tolist()
#         result += len(neis[depth][int(labels[i])].intersection(topk)) / 10 # top 10
#     result /= N
#     print(f'{depth}-hop neighbor in top{10} = {result}')
#     results.append(result)

# print('sum = ', sum(results))
#%%
labels = a['labels'][:, 0]
N = len(labels)


results = []
for depth in range(6):
    result = 0
    for i in range(N):
        topk = a['idx'][i][:1].tolist()
        result += len(neis[depth][int(labels[i])].intersection(topk)) / 1 # top 1
    result /= N
    print(f'{depth}-hop neighbor in top{1} = {result}')
    results.append(result)

print('sum = ', sum(results))
#%%

from collections import defaultdict
child2parent = defaultdict(set)
for h, r, t in triples:
    if r == 'is_a':
        child2parent[h].add(t)

E = len(name_array)
siblings = set()
for aa in range(E):
 for c in range(aa+1, E):
  if len(child2parent[aa].intersection(child2parent[c])) > 0:
   siblings.add((aa,c))
   siblings.add((c,aa))
#%%

grandpas = set()
grandsons = set()
for aa in range(E):
 for b in child2parent[aa]:
  for c in child2parent[b]:
   grandpas.add((aa,c))
   grandsons.add((c,aa))

# %%
results = []
depth = 2

results = {'grandpa':0, 'grandson':0, 'sibling':0, 'other':0}
cnt = 0

for i in range(N):
    top1 = int(a['idx'][i][0])
    if top1 in neis[depth][int(labels[i])]:
        cnt += 1
        if (int(labels[i]), top1) in grandpas:
            results['grandpa'] += 1
        elif (int(labels[i]), top1) in grandsons:
            results['grandson'] += 1
        elif (int(labels[i]), top1) in siblings:
            results['sibling'] += 1
        else:
            results['other'] += 1
            print(i, int(labels[i]), top1)
result = {k:v/cnt for k, v in results.items()}
print(f'{depth}-hop: pred is label\'s {result}')


# %%
from tqdm import tqdm
entity_set = {}
for aa, c in siblings:
    entity_set[aa] = None
    entity_set[c] = None
for aa, c in grandsons:
    entity_set[aa] = None
    entity_set[c] = None
entity_ids = list(entity_set.keys())
names = [name_array[i] for i in entity_ids]
inputs = tokenizer(names, return_tensors='pt', max_length=60, padding='max_length')
#%%
from torch.utils.data import Dataset, TensorDataset, DataLoader
input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
dataloader = DataLoader(dataset=TensorDataset(input_ids, attention_mask), batch_size=32, shuffle=False)
name_emb = []

bert_lm.eval()
bert_lm.cuda()
with torch.no_grad():
    for i, (input_ids, attention_mask) in enumerate(dataloader):
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        tmp = bert_lm.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, :].mean(dim=1)
        name_emb.append(tmp.cpu())
name_emb = torch.cat(name_emb, dim=0)

# %%
siblings_cos = torch.cosine_similarity(name_emb[[aa for (aa, c) in siblings]], name_emb[[c for (aa, c) in siblings]])
grandpas_cos = torch.cosine_similarity(name_emb[[aa for (aa, c) in grandpas]], name_emb[[c for (aa, c) in grandpas]])
grandsons_cos = torch.cosine_similarity(name_emb[[aa for (aa, c) in grandsons]], name_emb[[c for (aa, c) in grandsons]])
child_parent = [(h,t) for h,r,t in triples if r == 'is_a']
parent_cos = torch.cosine_similarity(name_emb[[aa for (aa, c) in child_parent]], name_emb[[c for (aa, c) in child_parent]])
assert grandsons_cos.mean() == grandpas_cos.mean()

print('similarity:', 'sib=',siblings_cos.mean(), 'grand=', grandsons_cos.mean(), 'parent=', parent_cos.mean())
#%%
def mat_mul(x, y):
    return x.matmul(y.T)
siblings_matmul = mat_mul(name_emb[[aa for (aa, c) in siblings]], name_emb[[c for (aa, c) in siblings]])
grandpas_matmul = mat_mul(name_emb[[aa for (aa, c) in grandpas]], name_emb[[c for (aa, c) in grandpas]])
grandsons_matmul = mat_mul(name_emb[[aa for (aa, c) in grandsons]], name_emb[[c for (aa, c) in grandsons]])
child_parent = [(h,t) for h,r,t in triples if r == 'is_a']
parent_matmul = mat_mul(name_emb[[aa for (aa, c) in child_parent]], name_emb[[c for (aa, c) in child_parent]])
assert grandsons_matmul.mean() == grandpas_matmul.mean()

print('similarity:', 'sib=',siblings_matmul.mean(), 'grand=', grandsons_cos.mean(), 'parent=', parent_matmul.mean())


#%%
import Levenshtein
Levenshtein.distance

# %%

# def counter(x):
#     from collections import defaultdict
#     cnt = defaultdict(int)
#     for i in x:
#         cnt[i] += 1
#     return dict(cnt)

# cnt = counter(a['idx'][:, 10].reshape(-1).tolist())
# %%
# top10 = sorted([(v, k) for k,v in cnt.items()])[-10:][::-1]
# for v, k in top10:
#     print(name_array[k], v)
#%%






#%%






#%%







# %%
test_id = 530
print('input={}\nlabel={}\npred={}'.format(tokenizer.decode(test_data_set[0].tensors[0][test_id], skip_special_tokens=True), name_array[int(labels[test_id])], [name_array[i] for i in a['idx'][test_id][:10].tolist()]))
# %%
neis[1][8640]
# %%
name_array[479], name_array[3623]
# %%
neis[2][8640]
# %%
list(name_array[i] for i in neis[2][8640])
# %%
for i in neis[2][8640]:
    print(i, i in labels)
# %%




#%%
labels = a['labels'][:, 0]
N = len(labels)

result1 = [[] for i in range(6)]
for depth in range(6):
    for i in range(N):
        top1 = a['idx'][i, 0].tolist()
        if top1 in neis[depth][int(labels[i])]:
            result1[depth].append(i)

# %%
test_id = 11
print('input={}\nlabel={}\npred={}'.format(tokenizer.decode(test_data_set[0].tensors[0][test_id], skip_special_tokens=True), name_array[int(labels[test_id])], [name_array[i] for i in a['idx'][test_id][:10].tolist()]))

# %%
test_id = 11
print('input={}\nlabel={}'.format(tokenizer.decode(test_data_set[0].tensors[0][test_id], skip_special_tokens=True), name_array[int(labels[test_id])]))
for i in a['idx'][test_id][:10].tolist():
    flag = False
    for k in range(6):
        if i in neis[k][int(labels[test_id])]:
            print(f'{name_array[i]} is {k}-hop neighbor of label')
            flag = True
            break
    if not flag:
        print(f'{name_array[i]} is >5-hop neighbor of label')

# %%

test_id = 42
print('input={}\nlabel={}'.format(tokenizer.decode(test_data_set[0].tensors[0][test_id], skip_special_tokens=True), name_array[int(labels[test_id])]))
for i in a['idx'][test_id][:10].tolist():
    flag = False
    for k in range(6):
        if i in neis[k][int(labels[test_id])]:
            print(f'{name_array[i]} is {k}-hop neighbor of label')
            flag = True
            break
    if not flag:
        print(f'{name_array[i]} is >5-hop neighbor of label')
# %%
test_id = 8
print('input={}\nlabel={}'.format(tokenizer.decode(test_data_set[0].tensors[0][test_id], skip_special_tokens=True), name_array[int(labels[test_id])]))
for i in a['idx'][test_id][:10].tolist():
    flag = False
    for k in range(6):
        if i in neis[k][int(labels[test_id])]:
            print(f'{name_array[i]} is {k}-hop neighbor of label')
            flag = True
            break
    if not flag:
        print(f'{name_array[i]} is >5-hop neighbor of label')
# %%