import os
import pickle
import argparse
from loader import log_dataloader
from solver import Solver

def main(args):
    # load data
    with open(os.path.join(args.data_path, 'log_input.pkl'), 'rb') as f:
        log_data = pickle.load(f)
    with open(os.path.join(args.data_path, 'log_vocab.pkl'), 'rb') as f:
        log_vocab = pickle.load(f)

    data_loader = log_dataloader(log_data, log_vocab,
                                 args.min_day, args.max_day,
                                 args.batch_size, args.window_size,
                                 args.target_hour)
    solver = Solver(args, train_loader=data_loader)
    solver.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='/home/datamininglab/Downloads/sinyu/')
    parser.add_argument('--save_path', type=str, default='/home/datamininglab/Downloads/sinyu/save_hanlda')

    parser.add_argument('--min_day', type=int, default=6)
    parser.add_argument('--max_day', type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=64)
    # batch_size = n_user * per_user

    parser.add_argument('--n_topics', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--target_hour', type=int, default=3)

    parser.add_argument('--embed_size', type=int, default=100)
    parser.add_argument('--pre_embed', type=bool, default=True)
    parser.add_argument('--embed_tuning', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip', type=float, default=5)

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--print_iters', type=int, default=30)
    parser.add_argument('--eval_iters', type=int, default=2000)
    parser.add_argument('--decay_iters', type=int, default=10000)
    parser.add_argument('--save_iters', type=int, default=10000)
    parser.add_argument('--test_iters', type=int, default=5000)

    parser.add_argument('--lambda_', type=float, default=200.0)
    parser.add_argument('--alpha_', type=float, default=1)

    args = parser.parse_args()
    main(args)
    
