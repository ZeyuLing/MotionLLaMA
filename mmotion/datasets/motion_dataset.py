import os.path
from typing import Optional, Union, Mapping, Sequence, Callable, List, Dict

from mmengine import Config, load
from mmengine.dataset import BaseDataset

from mmotion.registry import DATASETS
from mmotion.utils.logger import print_colored_log


@DATASETS.register_module(force=True)
class MotionDataset(BaseDataset):

    def __init__(self, ann_file: Optional[str] = '', metainfo: Union[Mapping, Config, None] = None,
                 data_root: Optional[str] = '', data_prefix: dict = dict(motion_path=''),
                 filter_cfg: Optional[dict] = None, indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True, pipeline: List[Union[dict, Callable]] = [], test_mode: bool = False,
                 lazy_init: bool = False, max_refetch: int = 1000, min_duration: float = 3.2):
        """
        :param min_duration: min duration of the motion file(in second)
        """

        self.min_duration = min_duration
        super().__init__(ann_file, metainfo, data_root, data_prefix, filter_cfg, indices, serialize_data, pipeline,
                         test_mode, lazy_init, max_refetch)

    def check_path(self, data_info) -> bool:
        """ check if the file path in data_info exists
        :param data_info: data info dict
        :return: True for exists, False for not
        """
        for key in self.data_prefix.keys():
            if not os.path.exists(data_info[key]):
                return False
        return True

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
        skip_samples = []
        for key, raw_data_info in raw_data_list.items():
            if raw_data_info.get('invalid') is True:
                skip_samples.append(key)
                continue
            data_info = self.parse_data_info(raw_data_info)

            data_info['index'] = key
            if_exist = self.check_path(data_info)
            if not if_exist:
                print_colored_log(f'{key} does not exist!')
                skip_samples.append(key)
                continue
            if self.min_duration > 0:
                if data_info['duration'] < self.min_duration:
                    skip_samples.append(key)
                    continue

            if isinstance(data_info, dict):

                data_list.append(data_info)
            elif isinstance(data_info, list):

                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')
        print_colored_log(f'{len(data_list)} samples in total, {len(skip_samples)} skipped because of absence or short length')
        return data_list

