
from collections import defaultdict
from os.path import exists
from typing import List, Dict, Optional, Union, Sequence, Callable, Mapping, Type, Tuple

from mmengine import load, Config, join_path

from mmotion.datasets.motion_dataset import MotionDataset
from mmotion.registry import DATASETS
from mmotion.utils.logger import print_colored_log
from mmotion.utils.task import Modality, abbr_list_to_task_list, Task, all_tasks


@DATASETS.register_module(force=True)
class MultiModalLlamaDataset(MotionDataset):
    def __init__(self, ann_file: Optional[str] = '', metainfo: Union[Mapping, Config, None] = None,
                 data_root: Optional[str] = '', data_prefix: dict = dict(motion_path=''),
                 filter_cfg: Optional[dict] = None, indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False, max_refetch: int = 1000,
                 min_duration: float = 3.2,
                 task_mode: str = 'auto',
                 tasks=None, task_key: str = None,
                 task_repeat=dict(),
                 modal_keys: Dict = dict(),
                 verbose: bool = False
                 ):
        """
        :param task_mode: how to determine the tasks. Choices:  ['auto', 'key', 'preset']
        auto: Automatically determine tasks w.r.t modalities exist in the data item
        key: Keys in the annotation files
        preset: determine in the dataset param [tasks]
        :param min_duration: min duration of the motion file(in second)
        :param instruct_mode: 1.instruct mode 2.pretrain mode
        """
        assert task_mode in ['auto', 'key', 'preset'], f"task_mode must be 'auto', 'key', 'preset', but got {task_mode}"
        if task_mode == 'auto':
            assert tasks is None and task_key is None, "When task_mode is 'auto', please leave task_key and tasks to be None"
        self.task_mode = task_mode
        self.tasks = tasks
        self.task_key = task_key
        self.modal_keys = modal_keys
        self.task_repeat = defaultdict(lambda: 1)
        if task_repeat is not None:
            self.task_repeat.update(task_repeat)
        self.verbose = verbose
        self.task_counter = defaultdict(lambda: 0)
        super().__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline,
                         test_mode, lazy_init, max_refetch, min_duration)

    def check_path(self, data_info: Dict, required_keys: List[str]) -> bool:
        """ check if task-required data path all exist
        :param data_info: data info dict
        :param required_keys: the keys which the task requires.
         If the path doesn't exist, the sample will be ignored for current task
        :return: True for exists, False for not
        """
        for key in required_keys:
            assert key in data_info
            if key.endswith('_path') and not exists(data_info.get(key)):
                return False
        return True

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        This method should return dict or list of dict. Each dict or list
        contains the data information of a training sample. If the protocol of
        the sample annotations is changed, this function can be overridden to
        update the parsing logic while keeping compatibility.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            list or list[dict]: Parsed annotation.
        """
        for key, value in raw_data_info.items():
            if key.endswith('path'):
                prefix = self.data_prefix.get(key, self.data_root)
                raw_data_info[key] = join_path(prefix, value)
        for prefix_key, prefix in self.data_prefix.items():
            if prefix_key not in raw_data_info:
                continue
            raw_data_info[prefix_key] = join_path(prefix, raw_data_info[prefix_key])
        return raw_data_info

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`METAINFO`
        and ``metainfo`` argument of constructor.

        Returns:
            list[dict]: A list of annotation.
        """
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        metainfo = annotations['meta_info']
        raw_data_list: Dict = annotations['data_list']
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        # load and parse data_infos.
        data_list = []
        self.skip_samples = []

        for key, raw_data_info in raw_data_list.items():
            if raw_data_info.get('invalid') is True:
                self.skip_samples.append(key)
                continue
            # load interactor information to current sample
            raw_data_info = self.merge_inter(raw_data_info, raw_data_list)

            tasks, task_required_keys = self.determine_task(raw_data_info)

            duration = raw_data_info['duration']
            if duration < self.min_duration:
                self.skip_samples.append(key)
                continue

            for task, required_keys in zip(tasks, task_required_keys):
                task_data_info = raw_data_info.copy()
                task_data_info = self.parse_data_info(task_data_info)

                task_data_info = self.filter_keys(task_data_info, required_keys)

                task_data_info['index'] = f'{key}_{task.abbr}'
                task_data_info['task'] = task
                if_exists = self.check_path(task_data_info, required_keys)

                if not if_exists:
                    if self.verbose:
                        print_colored_log(f"{task_data_info['index']} not exists")
                    self.skip_samples.append(task_data_info)
                    continue
                repeat_times = self.task_repeat[task.abbr]
                for repeat in range(repeat_times):
                    task_data_info['index'] = f'{key}_{task.abbr}_{repeat}'
                    data_list.append(task_data_info)

                self.task_counter[task.abbr] += repeat_times

        self.data_list = data_list
        self.log_dataset_info()
        return data_list

    def log_dataset_info(self):
        print_colored_log(
            f'{len(self.data_list)} samples in total,'
            f' {len(self.skip_samples)} skipped because of absence or short length')
        for task, num_samples in self.task_counter.items():
            if num_samples == 0:
                continue
            print_colored_log(f'task {task}: {num_samples} samples')

    def merge_inter(self, cur_sample: Dict, all_samples: Dict):
        """ Load the interactor's information from data list and save to current sample's dict
        :param cur_sample:
        :param all_samples:
        :return:
        """
        interactor_key = cur_sample.get('interactor_key', None)
        if interactor_key is None:
            return cur_sample
        interactor_sample = all_samples[interactor_key]
        if interactor_sample.get('invalid'):
            return cur_sample
        for k, v in interactor_sample.items():
            cur_sample[f'interactor_{k}'] = v
        return cur_sample

    def determine_task(self, data_info: Dict):
        if self.task_mode == 'preset':
            candidate_tasks = abbr_list_to_task_list(self.tasks)
            tasks, task_required_keys = self.auto_det_tasks(data_info, candidate_tasks)
        elif self.task_mode == 'key':
            candidate_tasks = abbr_list_to_task_list(data_info[self.task_key])
            tasks, task_required_keys = self.auto_det_tasks(data_info, candidate_tasks)
        else:
            tasks, task_required_keys = self.auto_det_tasks(data_info)
        return tasks, task_required_keys

    def auto_det_tasks(self, data_info: Dict, candidate_tasks=None) -> Tuple[List[Type[Task]], List[List[str]]]:
        """ Determine what tasks can this data sample be used to train the MotionLlama
        :param data_info: information of the data sample
        :param candidate_tasks: Only det tasks within candidate tasks
        :return: tasks and the keys needed for each task to train
        """
        tasks = []
        task_required_keys = []
        task_lib = candidate_tasks or all_tasks

        for task in task_lib:
            # involved modalities
            all_required_modals: List[Type[Modality]] = task.all_modality()
            task_key = []
            for modal in all_required_modals:
                modal_exist = False
                candidate_keys = modal.load_keys + self.modal_keys.get(modal.name, [])
                for key in candidate_keys:
                    if key not in data_info:
                        key = f'{key}_path'

                    if key in data_info:
                        task_key.append(key)
                        modal_exist = True
                        break

                if not modal_exist:
                    break

            if len(task_key) == len(all_required_modals):
                task_key = list(set(task_key))
                tasks.append(task)
                task_required_keys.append(task_key)

        assert len(tasks) == len(task_required_keys)
        return tasks, task_required_keys

    @staticmethod
    def filter_keys(data_info: Dict, required_keys: List) -> Dict:
        for key in list(data_info.keys()):
            if key.endswith('_path') and key not in required_keys:
                data_info.pop(key)
        return data_info

    def get_task_by_idx(self, idx):
        """ Used for TaskBatchSampler to gather the samples with the same task into a batch
        :param idx: index
        :return: task
        """
        return self.data_list[idx]['task']
