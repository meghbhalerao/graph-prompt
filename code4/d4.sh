CUDA_VISIBLE_DEVICES=4 DEBUG_DECODE_EVAL=1 DEBUG_DECODE_OUTPUT=1 python main_simple_path_with_save.py --filename=../data/datasets/cl.obo --is_unseen --use_scheduler --syn_ratio=2 --ent_ratio=1 --hrt_ratio=0 --lr=2.500e-04 --epoch_num=10 --pretrain_emb_iter=400 --use_get_ent_emb  --path_depth=1 --seed=0 --exp_path=../exp/0cl.obo_unseen__path1/    > 0cl.obo_unseen_log_simplepath10_ratio210_pretrain400_depth1_2.500e-04_get_ent  2>&1
CUDA_VISIBLE_DEVICES=4 DEBUG_DECODE_EVAL=1 DEBUG_DECODE_OUTPUT=1 python main_simple_path_with_save.py --filename=../data/datasets/cl.obo  --use_scheduler --syn_ratio=2 --ent_ratio=1 --hrt_ratio=0 --lr=1.500e-04 --epoch_num=10 --pretrain_emb_iter=400   --path_depth=1 --seed=0 --exp_path=../exp/0cl.obo_seen__path1/    > 0cl.obo_seen_log_simplepath10_ratio210_pretrain400_depth1_1.500e-04  2>&1
CUDA_VISIBLE_DEVICES=4 DEBUG_DECODE_EVAL=1 DEBUG_DECODE_OUTPUT=1 python main_simple_path_with_save.py --filename=../data/datasets/cl.obo  --use_scheduler --syn_ratio=2 --ent_ratio=1 --hrt_ratio=2 --lr=2.000e-04 --epoch_num=10 --pretrain_emb_iter=400   --path_depth=1 --seed=0 --exp_path=../exp/0cl.obo_seen_kg_path1/    > 0cl.obo_seen_log_simplepathkg10_ratio212_pretrain400_depth1_2.000e-04  2>&1
CUDA_VISIBLE_DEVICES=4 DEBUG_DECODE_EVAL=1 DEBUG_DECODE_OUTPUT=1 python main_simple_path_with_save.py --filename=../data/datasets/cl.obo  --use_scheduler --syn_ratio=2 --ent_ratio=1 --hrt_ratio=2 --lr=2.500e-04 --epoch_num=10 --pretrain_emb_iter=400   --path_depth=2 --seed=0 --exp_path=../exp/0cl.obo_seen_kg_path2/    > 0cl.obo_seen_log_simplepathkg10_ratio212_pretrain400_depth2_2.500e-04  2>&1
CUDA_VISIBLE_DEVICES=4 DEBUG_DECODE_EVAL=1 DEBUG_DECODE_OUTPUT=1 python main_simple_path_with_save.py --filename=../data/datasets/fbbt.obo --is_unseen --use_scheduler --syn_ratio=2 --ent_ratio=0 --hrt_ratio=2 --lr=1.000e-04 --epoch_num=40 --pretrain_emb_iter=0 --use_get_ent_emb  --path_depth=2 --seed=0 --exp_path=../exp/0fbbt.obo_unseen_kg_path2xx/    > 0fbbt.obo_unseen_log_simplepathkg40_ratio212_pretrain400_depth2_1.000e-04_get_entxx  2>&1
CUDA_VISIBLE_DEVICES=4 DEBUG_DECODE_EVAL=1 DEBUG_DECODE_OUTPUT=1 python main_simple_path_with_save.py --filename=../data/datasets/hp.obo  --use_scheduler --syn_ratio=2 --ent_ratio=0 --hrt_ratio=2 --lr=1.500e-04 --epoch_num=20 --pretrain_emb_iter=0   --path_depth=2 --seed=0 --exp_path=../exp/0hp.obo_seen_kg_path2dxx/    > 0hp.obo_seen_log_simplepathkg10_ratio212_pretrain400_depth2_1.500e-04dxx  2>&1
CUDA_VISIBLE_DEVICES=4 DEBUG_DECODE_EVAL=1 DEBUG_DECODE_OUTPUT=1 python main_simple_path_with_save.py --filename=../data/datasets/mp.obo --is_unseen --use_scheduler --syn_ratio=2 --ent_ratio=0 --hrt_ratio=2 --lr=1.500e-04 --epoch_num=80 --pretrain_emb_iter=0 --use_get_ent_emb  --path_depth=2 --seed=0 --exp_path=../exp/0mp.obo_unseen_kg_path2dxx/    > 0mp.obo_unseen_log_simplepathkg40_ratio212_pretrain400_depth2_1.500e-04_get_entdxx  2>&1
