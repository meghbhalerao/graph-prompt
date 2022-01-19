def gen(cuda_id, is_unseen, epoch, syn_ratio, ent_ratio, hrt_ratio, pre_iter, lr, get_ent, filename, path_depth, seed=0, save=True):
    log_prefix = '%d'%seed
    log_postfix = ''
    
    
    formatted_lr = '%.3e'%lr
    log_prefix += filename
    filepath = '../data/datasets/' + filename

    if is_unseen:
        log_prefix += '_unseen'
        use_unseen = '--is_unseen'
    else:
        log_prefix += '_seen'
        use_unseen = ''

    if hrt_ratio != 0:
        is_kg = 'kg'
    else:
        is_kg = ''

    if get_ent:
        use_get_ent = '--use_get_ent_emb'
        log_postfix += '_get_ent'
    else:
        use_get_ent = ''

    
    log_template = f'{log_prefix}_log_simplepath{is_kg}{epoch}_ratio{syn_ratio}{ent_ratio}{hrt_ratio}_pretrain{pre_iter}_depth{path_depth}_{formatted_lr}{log_postfix}'
    if save:
        save = f'--exp_path=../exp/{log_prefix}_{is_kg}_path{path_depth}/'
    else:
        save = ''
    cmd_template = f'CUDA_VISIBLE_DEVICES={cuda_id} DEBUG_DECODE_EVAL=1 DEBUG_DECODE_OUTPUT=1 python main_simple_path_with_save.py --filename={filepath}\
 {use_unseen} --use_scheduler --syn_ratio={syn_ratio} --ent_ratio={ent_ratio} --hrt_ratio={hrt_ratio} --lr={formatted_lr} --epoch_num={epoch} --pretrain_emb_iter={pre_iter}\
 {use_get_ent}  --path_depth={path_depth} --seed={seed} {save}\
    > {log_template}  2>&1'
    return cmd_template



available_cuda = [ 1, 2, 3, 4, 5]
available_cuda_idx = 0
cmd_list = []

for filename in [
    'cl.obo',
    "hp.obo",
    "doid.obo",
    "fbbt.obo",
    "mp.obo"
]:
    for is_unseen in [True, False]:
        for epoch in [10]:
            
            if filename == 'fbbt.obo':
                epoch = 40
            if filename == 'mp.obo' and is_unseen:
                epoch = 40
            if filename == 'hp.obo' and is_unseen:
                epoch = 20
            
            for syn_ent_hrt in ['210', '212']:
                for pre_iter in [400]:
                    for path_depth in [1, 2]:
                        if path_depth != 1 and syn_ent_hrt == '210':
                            continue
                        for lr in [1.0e-4, 1.5e-4, 2.0e-4, 2.5e-4]:
                            if is_unseen:
                                get_ent = True
                            else:
                                get_ent = False
                            cuda_id = available_cuda[available_cuda_idx]
                            available_cuda_idx += 1
                            available_cuda_idx %= len(available_cuda)

                            syn_ratio, ent_ratio, hrt_ratio = (int(i) for i in syn_ent_hrt)
                            
                            for seed in range(1):
                                cmd_list.append(gen(cuda_id=cuda_id, is_unseen=is_unseen, epoch=epoch, 
                                                syn_ratio=syn_ratio, ent_ratio=ent_ratio, 
                                                hrt_ratio=hrt_ratio, pre_iter=pre_iter, 
                                                lr=lr, get_ent=get_ent, filename=filename, path_depth=path_depth, seed=seed))

print('#', len(cmd_list))
for i in cmd_list:
    print(i)


"""

python generate_command_list.py > command_list.txt
cat command_list.txt | grep "CUDA_VISIBLE_DEVICES=1" > d1.sh
cat command_list.txt | grep "CUDA_VISIBLE_DEVICES=2" > d2.sh
cat command_list.txt | grep "CUDA_VISIBLE_DEVICES=3" > d3.sh
cat command_list.txt | grep "CUDA_VISIBLE_DEVICES=4" > d4.sh
cat command_list.txt | grep "CUDA_VISIBLE_DEVICES=5" > d5.sh
# cat command_list.txt | grep "CUDA_VISIBLE_DEVICES=6" > d6.sh

"""