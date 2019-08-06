# MLE: 5.151166
python new_main.py --data mmb_hard --mmb_clusters 6 --K 1 --method mle --lr 0.001 --max_iter 50 --init_theta random

# DMHP K=1 5.151166
python new_main.py --data mmb_hard --mmb_clusters 6 --K 1 --method mdhp --lr 0.001 --max_iter 50 --init_theta random
# DMHP K=3 5.18048
python new_main.py --data mmb_hard --mmb_clusters 6 --K 3 --method mdhp --lr 0.003 --max_iter 50 --init_theta random
# DMHP K=6 5.169166
python new_main.py --data mmb_hard --mmb_clusters 6 --K 6 --method mdhp --lr 0.0065 --max_iter 50 --init_theta random
# DMHP K=10 5.1760737063491
python new_main.py --data mmb_hard --mmb_clusters 6 --K 10 --method mdhp --lr 0.007 --max_iter 50 --init_theta random

# MLE SEP 5.0563031
python test_baseline1.py --data mmb_hard --mmb_clusters 6 --K 1 --method mle --lr 0.0005 --max_iter 500 --init_theta random

# MTL 5.16148
python test_baseline_MTL.py --data mmb_hard --mmb_clusters 6 --K 1 --method mle --lr 0.001 --max_iter 50 --init_theta random

# 2step K=1 5.151166
python new_main.py --data mmb_hard --mmb_clusters 6 --K 1 --method 2step --lr 0.001 --max_iter 50 --init_theta random
# 2step K=3 5.03476967
python new_main.py --data mmb_hard --mmb_clusters 6 --K 3 --method 2step --lr 0.01 --max_iter 50 --init_theta random
# 2step K=6 5.1403
python new_main.py --data mmb_hard --mmb_clusters 6 --K 6 --method 2step --lr 0.017 --max_iter 50 --init_theta random
# 2step K=10 5.066497
python new_main.py --data mmb_hard --mmb_clusters 6 --K 10 --method 2step --lr 0.01 --max_iter 50 --init_theta random


# maml K=1 5.183809
python new_main.py --data mmb_hard --mmb_clusters 6 --K 1  --method maml --lr 0.001 --inner_lr 5e-5 --max_iter 50 --init_theta random
# maml K=3 5.24883328
python new_main.py --data mmb_hard --mmb_clusters 6 --K 3  --method maml --lr 0.007 --inner_lr 5e-6 --max_iter 50 --init_theta random
# maml K=6 5.184980451
python new_main.py --data mmb_hard --mmb_clusters 6 --K 6  --method maml --lr 0.008 --inner_lr 2e-4 --max_iter 50 --init_theta random --pretrain_iter 0
# maml K=10 5.18565
python new_main.py --data mmb_hard --mmb_clusters 6 --K 10  --method maml --lr 0.011 --inner_lr 7e-5 --max_iter 50 --init_theta random --pretrain_iter 0



# fomaml K=1 5.186447091102
python new_main.py --data mmb_hard --mmb_clusters 6 --K 1  --method fomaml --lr 0.0006 --inner_lr 5e-4 --max_iter 50 --init_theta random
# fomaml K=3 5.210234
python new_main.py --data mmb_hard --mmb_clusters 6 --K 3  --method fomaml --lr 0.0002 --inner_lr 1e-5 --max_iter 50 --init_theta random
# fomaml K=6 5.22537235869839
python new_main.py --data mmb_hard --mmb_clusters 6 --K 6  --method fomaml --lr 0.00006 --inner_lr 3e-5 --max_iter 50 --init_theta random
# fomaml K=10 5.24032040
python new_main.py --data mmb_hard --mmb_clusters 6 --K 10  --method fomaml --lr 4.5e-6 --inner_lr 1.5e-6 --max_iter 50 --init_theta random




# reptile K=1
python new_main.py --data mmb_hard --mmb_clusters 6 --K 1  --method reptile --lr 0.002 --inner_lr 0.001--max_iter 50 --init_theta random
# reptile K=3
# reptile K=6
# reptile K=10
