from utils.constants import task_to_task_id
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm

def preprocess(all_data, tasks=(0, 1, 2, 3, 4, 5, 6)):
    '''
    preprocess raw data text of tasks, store in all_data dict (inplace),
    only tasks specified will be preprocess
    :param all_data: data dict include ["train", "val", "test"] keys
    :return: None
    '''
    tokenizer = TweetTokenizer()

    for task_name, task_id in tqdm(task_to_task_id.items(), desc="preprocess tweet"):
        if task_id not in tasks:  # skip tasks not specified
            continue
        for split in ["train", "val", "test"]:
            with open(f"tweeteval/datasets/{task_name}/{split}_text.txt") as xs, \
                    open(f"tweeteval/datasets/{task_name}/{split}_labels.txt") as ys:
                for y, x in zip(ys.readlines(), xs.readlines()):
                    if task_id == 6:
                        all_data[split]["stance"].append((task_id, int(y.strip()), tokenizer.tokenize(x.lower())))
                    else:
                        all_data[split][task_name].append((task_id, int(y.strip()), tokenizer.tokenize(x.lower())))
            # can also access data by task id
            all_data[split][task_id] = all_data[split][task_name]

def get_vocabulary(all_data):
    '''
    get vocabulary of training data
    :param all_data: data dict including ["train", "val", "test"] keys
    :return: vocabulary built by train data
    '''
    vocabulary = set()
    for task_name, task_data in all_data["train"].items():
        for a, b, sentence in task_data:
            vocabulary |= set(sentence)

    for word in list(vocabulary):
        if word.startswith("#"):
            vocabulary.remove(word)
    if "@user" in vocabulary:
        vocabulary.remove("@user")
    return vocabulary
