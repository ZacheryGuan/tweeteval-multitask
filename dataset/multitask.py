from torch.nn.utils.rnn import pad_sequence, pack_sequence
from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler, SequentialSampler
from itertools import cycle
import random

class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        task_id_2_data_set_dic = {}
        for task_id, dataset in enumerate(datasets):
            task_id_2_data_set_dic[task_id] = dataset
        self.task_id_2_data_set_dic = task_id_2_data_set_dic


    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, idx):
        ans = []
        for sample in idx:
            task_id, sample_id = sample
            ans.append(self.task_id_2_data_set_dic[task_id][sample_id])

        return ans

class MultiTaskBatchSampler(BatchSampler):
    def __init__(
            self,
            datasets,
            batch_size,
    ):
        self._datasets = datasets
        self._batch_size = batch_size
        train_data_list = []
        for task_id, dataset in enumerate(datasets):
            train_data_list.append(
                self._get_shuffled_index_batches(len(dataset), batch_size)
            )
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [
            list(range(i, min(i + batch_size, dataset_len)))
            for i in range(0, dataset_len, batch_size)
        ]
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        task_ids = []

        for i in range(0, len(self._train_data_list)):
            task_ids += [i] * len(self._train_data_list[i])
        random.shuffle(task_ids)
        for task_id in task_ids:
            batch = next(all_iters[task_id])
            yield [(task_id, sample_id) for sample_id in batch]