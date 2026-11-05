from enum import StrEnum


class DocileSplit(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TRAINVAL = "trainval"
    TEST = "test"
