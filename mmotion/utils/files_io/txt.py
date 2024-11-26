from typing import List


def read_txt(txt_path: str, encoding='utf8') -> str:
    with open(txt_path, 'r', encoding=encoding) as fp:
        return fp.read()

def write_txt(save_path,data,encoding='utf8'):
    with open(save_path, 'w', encoding=encoding) as fp:
        fp.write(data)


def read_list_txt(txt_path: str, encoding='utf8') -> List[str]:
    with open(txt_path, 'r', encoding=encoding) as fp:
        lines = fp.read()
        lines = lines.strip().split('\n')
        return [line.strip() for line in lines]



def write_list_txt(data: List, save_path, encoding='utf8'):
    with open(save_path, 'w', encoding=encoding) as fp:
        fp.write('\n'.join(data))


def read_list_txt(txt_path: str, encoding='utf8') -> List[str]:
    with open(txt_path, 'r', encoding=encoding) as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]
        return lines
