#CUDA_VISIBLE_DEVICES=0 nohup bash run_unseen.sh 0 2 >../out_file/0.out&
#CUDA_VISIBLE_DEVICES=1 nohup bash run_unseen.sh 2 5 >../out_file/1.out&
#CUDA_VISIBLE_DEVICES=2 nohup bash run_unseen.sh 5 7  >../out_file/2.out&
#CUDA_VISIBLE_DEVICES=3 nohup bash run_unseen.sh 7 -1 >../out_file/3.out&
#CUDA_VISIBLE_DEVICES=4 nohup bash run_seen.sh 0 2 >../out_file/4.out&
#CUDA_VISIBLE_DEVICES=5 nohup bash run_seen.sh 2 5 >../out_file/5.out&
CUDA_VISIBLE_DEVICES=6 nohup bash run_seen.sh 3 5>../out_6.out&
CUDA_VISIBLE_DEVICES=7 nohup bash run_seen.sh 5 -1 >../out_7.out&

#CUDA_VISIBLE_DEVICES=5 nohup bash run_normco.sh cl >run_out/cl-unseen.out&
#CUDA_VISIBLE_DEVICES=6 nohup bash run_normco.sh hp >run_out/hp-unseen.out&
#CUDA_VISIBLE_DEVICES=7 nohup bash run_normco.sh doid >run_out/doid-unseen.out&
#CUDA_VISIBLE_DEVICES=0 nohup bash run_normco.sh fbbt >run_out/fbbt-unseen.out&
#CUDA_VISIBLE_DEVICES=1 nohup bash run_normco.sh mp >run_out/mp-unseen.out&
