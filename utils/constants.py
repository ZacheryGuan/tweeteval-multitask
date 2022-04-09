task_to_task_id = {
    "emoji": 0,  # 40k
    "emotion": 1,  # 0
    "hate": 2,  # 10k
    "irony": 3,
    "offensive": 4,
    "sentiment": 5,  # 25k

    # stance vocab = 0
    "stance/abortion": 6,  # use '/' for indexing data file
    "stance/atheism": 6,
    "stance/climate": 6,
    "stance/feminist": 6,
    "stance/hillary": 6
}

tid_2_name = dict(zip(range(7), [
    "emoji",
    "emotion",
    "hate",
    "irony",
    "offensive",
    "sentiment",
    "stance"
]))