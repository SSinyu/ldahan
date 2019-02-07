import os
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver

def main(config):
    cudnn.benchmark = True

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
        print('Create path : {}'.format(config.save_path))

    if config.mode == 'train':
        train_loader = get_loader(input_path=config.input_path,
                                  vocab_path=config.vocab_path,
                                  min_day_length=config.min_day_length,
                                  max_day_length=config.max_day_length,
                                  per_user=config.per_user,
                                  batch_size=config.n_user)

        solver = Solver(config, train_loader=train_loader)
        solver.train()

    elif config.mode == 'test':
        test_loader = get_loader(input_path=config.input_path,
                                 vocab_path=config.vocab_path,
                                 min_day_length=14,
                                 max_day_length=14,
                                 per_user=1,
                                 batch_size=50,
                                 mode=config.mode)

        solver = Solver(config, train_loader=None, test_loader=test_loader)
        solver.test()

    else:
        raise ValueError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default="/media/datamininglab/새 볼륨/USERLOG/select_user2/dic/each_user/355k_10m_input.pkl")
    parser.add_argument('--vocab_path', type=str, default="/media/datamininglab/새 볼륨/USERLOG/select_user2/dic/each_user/355k_10m_vocab.pkl")
    parser.add_argument('--save_path', type=str, default="/home/datamininglab/PycharmProjects/lda2vec_pytorch/HAN/model_ver1/save/2_add_lambda_and_alpha/")

    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--min_day_length', type=int, default=5)
    parser.add_argument('--max_day_length', type=int, default=10)
    parser.add_argument('--per_user', type=int, default=5)
    parser.add_argument('--n_user', type=int, default=30)
    # batch_size = n_user * per_user

    parser.add_argument('--n_topics', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--num_epochs', type=int, default=10)

    parser.add_argument('--decay_iters', type=int, default=5000)
    parser.add_argument('--print_iters', type=int, default=100)
    parser.add_argument('--save_iters', type=int, default=5000)
    parser.add_argument('--test_iters', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=.5)
    parser.add_argument('--beta2', type=float, default=.999)

    parser.add_argument('--lambda_', type=float, default=200.0)
    parser.add_argument('--alpha_w', type=float, default=1)

    config = parser.parse_args()
    main(config)
