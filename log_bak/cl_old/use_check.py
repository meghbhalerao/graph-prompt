import os
import sys
import re


filename = input('file: (e.g., hp.obo) ').strip()

assert re.match('^[a-z]+\.obo$', filename) is not None


tmp_file_list = [
    '/tmp/{}unseen_nokg'.format(filename.split('.')[0]),
    '/tmp/{}unseen_kg1'.format(filename.split('.')[0]),
    '/tmp/{}unseen_kg2'.format(filename.split('.')[0]),
    '/tmp/{}seen_nokg'.format(filename.split('.')[0]),
    '/tmp/{}seen_kg1'.format(filename.split('.')[0]),
    '/tmp/{}seen_kg2'.format(filename.split('.')[0]),
]
epoch = int(input('epoch:'))
seed = int(input('seed:'))
cmd_list = [
    'python check.py {}{}_unseen_log_simplepath{}_.+depth1.+[^x]$ "({}|-1) done"   > {}'.format(seed, filename, epoch, epoch, tmp_file_list[0]),
    'python check.py {}{}_unseen_log_simplepathkg{}_.+depth1.+[^x]$ "({}|-1) done" > {}'.format(seed, filename, epoch, epoch, tmp_file_list[1]),
    'python check.py {}{}_unseen_log_simplepathkg{}_.+depth2.+[^x]$ "({}|-1) done" > {}'.format(seed, filename, epoch, epoch, tmp_file_list[2]),
    'python check.py {}{}_seen_log_simplepath{}_.+depth1.+[^x]$ "({}|-1) done"     > {}'.format(seed, filename, epoch, epoch, tmp_file_list[3]),
    'python check.py {}{}_seen_log_simplepathkg{}_.+depth1.+[^x]$ "({}|-1) done"   > {}'.format(seed, filename, epoch, epoch, tmp_file_list[4]),
    'python check.py {}{}_seen_log_simplepathkg{}_.+depth2.+[^x]$ "({}|-1) done"   > {}'.format(seed, filename, epoch, epoch, tmp_file_list[5])
]

idx = [int(i) for i in input('check=(0:unseen,1:unseen+kg1,2:unseen+kg2,3:seen,4:seen+kg1,5:seen+kg2)').strip()]
tmp_file_list = [tmp_file_list[i] for i in idx]
cmd_list = [cmd_list[i] for i in idx]

for i in cmd_list:
    print(i, file=sys.stderr)
    os.system(i)


for f in tmp_file_list:
    results = {}
    lines = [i.strip() for i in open(f)]
    
    # for i in range(len(lines) // 4):
    #     ignore, logfile, valid, test = lines[i * 4:(i+1)*4]
    #     # 2021-08-23 02:02:47,585 epoch 10 done, accu_1 = 0.569746, accu_10 = 0.830918

    #     def parse(x):
    #         g = re.match('^.+done, accu_1 = ([\d\.]+), accu_10 = ([\d\.]+)$', x)
    #         return (g.group(1), g.group(2))

    #     results[logfile] = parse(valid) + parse(test) + (valid, test, )


    def parse(x):
        g = re.match('^.+done, accu_1 = ([\d\.]+), accu_10 = ([\d\.]+)$', x)
        return (g.group(1), g.group(2))

    for i, line in enumerate(lines):
        if line.strip() == '=' * 20:
            ret = eval(lines[i+1])
    ret = {k:ret[k] for k in ret if len(ret[k]) == 2}
    for fi in ret:
        results[fi] = parse(ret[fi][0]) + parse(ret[fi][1]) + (ret[fi][0], ret[fi][1])

    best = None
    for logfile in results:
        if best is None:
            best = logfile
        elif results[logfile][2] > results[best][2]: # test acc1
            best = logfile

    # valid_acc1, valid_acc10, test_acc1, test_acc10 = results[best][:4]
    print( '\t'.join(['%.2f'%(float(i) * 100) for i in results[best][:4]])
            +'\t' + '\t'.join(results[best][4:])
            +'\t'+best)

