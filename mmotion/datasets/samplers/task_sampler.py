import random
from collections import defaultdict
from torch.utils.data import BatchSampler, Sampler
from typing import Iterator, List, Union, Dict
from mmengine.registry import DATA_SAMPLERS  # 假设这是正确的导入路径

@DATA_SAMPLERS.register_module()
class TaskBatchSampler(BatchSampler):

    def __init__(self,
                 sampler: Union[Sampler[int], Iterator[int]],
                 batch_size: int,
                 drop_last: bool,
                 shuffle_tasks: bool = True,
                 shuffle_within_task: bool = True) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.shuffle_tasks = shuffle_tasks
        self.shuffle_within_task = shuffle_within_task
        self.task_to_indices = self._group_indices_by_task()

    def _group_indices_by_task(self) -> Dict[str, List[int]]:
        from collections import defaultdict
        from torch.utils.data import ConcatDataset

        task_groups = defaultdict(list)
        dataset = self.sampler.dataset

        for idx in self.sampler:
            if isinstance(dataset, ConcatDataset):
                dataset_idx, sample_idx = dataset._get_ori_dataset_idx(idx)
                task = dataset.datasets[dataset_idx].get_task_by_idx(sample_idx).abbr
            else:
                task = dataset.get_task_by_idx(idx).abbr
            task_groups[task].append(idx)
        return task_groups

    def __iter__(self) -> Iterator[List[int]]:
        task_to_indices = {task: indices.copy() for task, indices in self.task_to_indices.items()}

        while task_to_indices:
            # 计算剩余样本总量
            total_remaining_samples = sum(len(indices) for indices in task_to_indices.values())

            # 计算每个任务的选择概率
            tasks = list(task_to_indices.keys())
            probabilities = [len(task_to_indices[task]) / total_remaining_samples for task in tasks]

            if self.shuffle_tasks:
                # 随机打乱任务顺序
                selected_task = random.choices(tasks, weights=probabilities)[0]
            else:
                selected_task = tasks[0]

            indices = task_to_indices[selected_task]

            if self.shuffle_within_task:
                random.shuffle(indices)

            batch = indices[:self.batch_size]
            task_to_indices[selected_task] = indices[self.batch_size:]

            if len(task_to_indices[selected_task]) == 0:
                del task_to_indices[selected_task]
            yield batch

    def __len__(self) -> int:
        total_batches = 0
        for indices in self.task_to_indices.values():
            if self.drop_last:
                total_batches += len(indices) // self.batch_size
            else:
                total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_batches
