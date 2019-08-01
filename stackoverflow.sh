nohup python main.py  --method mle --lr 0.0004 --inner_lr 0.0004 --K 2 --max_iter 1000 --init_theta random &
nohup python main.py  --method mle --lr 0.001 --inner_lr 0.001 --K 2 --max_iter 1000 --init_theta random &

nohup python main.py  --method maml --lr 0.0003 --inner_lr 0.0003 --K 2 --max_iter 1000 --init_theta random &
nohup python main.py  --method maml --lr 0.0004 --inner_lr 0.0004 --K 2 --max_iter 1000 --init_theta random &
nohup python main.py  --method maml --lr 0.001 --inner_lr 0.001 --K 2 --max_iter 1000 --init_theta random &

nohup python main.py  --method reptile --lr 0.05 --inner_lr 0.05 --K 2 --max_iter 1000 --init_theta random &
nohup python main.py  --method reptile --lr 0.1 --inner_lr 0.1 --K 2 --max_iter 1000 --init_theta random &

nohup python main.py  --method fomaml --lr 0.0004 --inner_lr 0.0004 --K 2 --max_iter 1000 --init_theta uniform &
nohup python main.py  --method fomaml --lr 0.001 --inner_lr 0.001 --K 2 --max_iter 1000 --init_theta uniform &
nohup python main.py  --method fomaml --lr 0.002 --inner_lr 0.002 --K 2 --max_iter 1000 --init_theta uniform &



# python3.6 main.py  --method mle --lr 0.0004 --inner_lr 0.0004 --K 2 --max_iter 1300 &
# python3.6 main.py --method maml --lr 0.004 --inner_lr 0.004 --K 2 --max_iter 1300 &
# python3.6 main.py --method maml --lr 0.004 --inner_lr 0.004 --K 3 --max_iter 1300 &
# python3.6 main.py --method reptile --lr 0.03 --inner_lr 0.03 --K 2 --max_iter 1500 &
# python3.6 main.py --method reptile --lr 0.03 --inner_lr 0.03 --K 3 --max_iter 1500 &
# python3.6 main.py --method fomaml --lr 0.0004 --inner_lr 0.0004 --K 2 --max_iter 1300 &

#python3.6 main.py --method reptile --lr 0.04 --inner_lr 0.04 --K 2 --max_iter 1500 &
#python3.6 main.py --method fomaml --lr 0.0002 --inner_lr 0.0002 --K 2 --max_iter 1300 &