CUDA_VISIBLE_DEVICES=1 DEBUG_DECODE_EVAL=1 DEBUG_DECODE_OUTPUT=1 python main_simple_path_with_save.py --filename=../data/datasets/hp.obo  --use_scheduler --syn_ratio=2 --ent_ratio=1 --hrt_ratio=2 --lr=1.500e-04 --epoch_num=10 --pretrain_emb_iter=400   --path_depth=2 --seed=0 --exp_path=../exp/0hp.obo_seen_kg_path2/
