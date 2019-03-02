import numpy as np
from operator import itemgetter


class log_dataloader(object):
    def __init__(self, log_seq, log_vocab, min_day, max_day, batch_size, window_size, target_hour):
        self.log_seq = log_seq
        self.log_vocab = log_vocab
        self.min_day = min_day
        self.max_day = max_day
        self.batch_size = batch_size
        self.window_size = window_size
        self.target_len = target_hour * 6
        self.day_log = 24 * 6

    def get_batch_ver2(self, shuffle=True):
        # remove the cross entropy part from get_batch()
        if shuffle:
            indices = np.arange(len(self.log_seq))
            np.random.shuffle(indices)
            log_seq = itemgetter(*indices)(self.log_seq)
        else:
            log_seq = self.log_seq

        for user_i in range(0, len(log_seq)-self.batch_size, self.batch_size):
            batch_user = log_seq[user_i:(user_i+self.batch_size)]

            batch_hx, batch_y, user_ids = [], [], []
            for user_id, user_log in batch_user:
                day_length = np.random.randint(self.min_day, self.max_day+1)
                seq_size = day_length * self.day_log
                x_start = np.random.randint(0, len(user_log)-seq_size-self.target_len)
                x_end = x_start + seq_size

                subseq_x = user_log[x_start:x_end]
                subseq_x = subseq_x + [0] * (self.max_day * self.day_log - len(subseq_x))
                subseq_hx = [subseq_x[sub_i:(sub_i+self.day_log)] for sub_i in range(0, len(subseq_x), self.day_log)]
                batch_hx.append(subseq_hx)

                y_end = x_end + self.target_len
                subseq_y = user_log[x_end: y_end]
                batch_y.append(self.seq_to_ratio(subseq_y))

                user_ids.append(user_id)

            batch_hx = np.array(batch_hx)
            batch_y = np.array(batch_y)
            user_ids = np.array(user_ids)
            yield batch_hx, batch_y, user_ids

    def get_batch(self, shuffle=True):
        if shuffle:
            indices = np.arange(len(self.log_seq))
            np.random.shuffle(indices)
            log_seq = itemgetter(*indices)(self.log_seq)
        else:
            log_seq = self.log_seq

        for user_i in range(0, len(log_seq)-self.batch_size, self.batch_size):
            batch_user = log_seq[user_i:(user_i+self.batch_size)]

            batch_x, batch_hx, batch_y = [], [], []
            user_ids, pivot_logs, target_logs = [], [], []

            for user_id, user_log in batch_user:

                day_length = np.random.randint(self.min_day, self.max_day + 1)
                seq_size = day_length * self.day_log
                x_start = np.random.randint(0, len(user_log) - seq_size - self.target_len)
                x_end = x_start + seq_size

                subseq_x = user_log[x_start:x_end]
                # zero padding
                subseq_x = subseq_x + [0] * (self.max_day * self.day_log - len(subseq_x))
                # convert to be hierarchical
                subseq_hx = [subseq_x[sub_i:(sub_i + self.day_log)] for sub_i in range(0, len(subseq_x), self.day_log)]
                batch_hx.append(subseq_hx)

                y_end = x_end + self.target_len
                subseq_y = user_log[x_end: y_end]  # x_end == y_start
                batch_y.append(self.seq_to_ratio(subseq_y))

                subseq_x = np.array(subseq_x)
                subseq_x = subseq_x[np.where(subseq_x > 0)]
                for pivot_i in range(self.window_size, len(subseq_x) - self.window_size):
                    for wi in range(1, self.window_size+1):
                        user_ids.append(user_id)
                        pivot_logs.append(user_log[pivot_i])
                        target_logs.append(user_log[pivot_i - wi])
                        user_ids.append(user_id)
                        pivot_logs.append(user_log[pivot_i])
                        target_logs.append(user_log[pivot_i + wi])

            # sampling by batch size
            total_ind = np.arange(len(user_ids))
            np.random.shuffle(total_ind)
            sltd_user_ids = itemgetter(*total_ind)(user_ids)[:self.batch_size]
            sltd_pivot_logs = itemgetter(*total_ind)(pivot_logs)[:self.batch_size]
            sltd_target_logs = itemgetter(*total_ind)(target_logs)[:self.batch_size]

            batch_hx = np.array(batch_hx)
            batch_y = np.array(batch_y)
            user_ids = np.array(sltd_user_ids)
            pivot_logs = np.array(sltd_pivot_logs)
            target_logs = np.array(sltd_target_logs)

            yield batch_hx, batch_y, pivot_logs, target_logs, user_ids

    def seq_to_ratio(self, y):
        count_ = np.unique(y, return_counts=True)
        ratio_ = [0] * len(self.log_vocab)
        for ch, cnt in zip(count_[0], count_[1]):
            ratio_[ch] = cnt/self.target_len
        return ratio_


