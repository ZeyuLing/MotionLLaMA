import pickle


def load_pickle(pickle_path: str, encoding='utf8'):
    with open(pickle_path, 'rb') as fp:
        result = pickle.load(fp, encoding=encoding)
    return result


def save_pickle(result, pickle_path: str):
    with open(pickle_path, 'wb') as fp:
        pickle.dump(result, fp)
