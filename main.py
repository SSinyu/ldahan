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
                                 args.batch_size, args.target_hour)

    solver = Solver(args, data_loader=data_loader)
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='.Downloads/')
    parser.add_argument('--save_path', type=str, default='.Downloads/save')

    parser.add_argument('--min_day', type=int, default=6)
    parser.add_argument('--max_day', type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--n_topics', type=int, default=10)
    parser.add_argument('--target_hour', type=int, default=3)

    parser.add_argument('--embed_size', type=int, default=100)
    parser.add_argument('--word_init', type=bool, default=True)
    parser.add_argument('--word_tuning', type=bool, default=True)
    parser.add_argument('--doc_init', type=bool, default=True)
    parser.add_argument('--doc_tuning', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--clip', type=float, default=5)

    parser.add_argument('--num_epochs', type=int, default=37)
    parser.add_argument('--print_iters', type=int, default=10)
    parser.add_argument('--decay_iters', type=int, default=10000)
    parser.add_argument('--eval_iters', type=int, default=100)
    parser.add_argument('--save_iters', type=int, default=10000)
    parser.add_argument('--test_iters', type=int, default=100000)

    parser.add_argument('--lambda_', type=float, default=0.00005)
    parser.add_argument('--alpha_', type=float, default=1)

    parser.add_argument('--m', type=str, default='ldahan', help='ldahan | han')
    parser.add_argument('--c', type=str, default='add', help='add | concat | wsum')
    parser.add_argument('--mode', type=str, default='train', help='train | test')

    args = parser.parse_args()
    print(args)
    main(args)
