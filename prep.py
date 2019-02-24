import pickle
import numpy as np
import os
import argparse
import logging
from gensim.models import word2vec


def main(args):
    with open(os.path.join(args.data_path, 'log_input_cat.pkl'), 'rb') as f:
        log_data = pickle.load(f)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    log_vec = word2vec.Word2Vec(log_data, iter=args.iter, min_count=0, sg=1, size=args.embed_size)

    log_embedding = np.zeros((len(log_vec.wv.vocab)+1, args.embed_size), dtype='float32')
    for i in range(0, len(log_vec.wv.vocab)):
        embed_v = log_vec.wv[log_vec.wv.index2word[i]]
        log_embedding[i+1] = embed_v
    print('shape: ', log_embedding.shape, type(log_embedding))

    np.save(os.path.join(args.save_path, 'embed.npy'), log_embedding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='/home/datamininglab/Downloads/sinyu/')
    parser.add_argument('--save_path', type=str, default='/home/datamininglab/Downloads/sinyu/')

    parser.add_argument('--embed_size', type=int, default=100)
    parser.add_argument('--iter', type=int, default=10)

    args = parser.parse_args()
    print(args)
    main(args)
