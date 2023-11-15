from data_utils.dataset import *

a = [1, 2, 3, 4, 5, 6, 6, 12, 23]

if __name__ == "__main__":
    train_analyses, valid_analyses = create_train_and_valid_analyses(truncate_length=6)

    train_set0, valid_set0 = create_n2key_datatsets(train_analyses, valid_analyses)
    train_set1, valid_set1 = create_key2red_datasets(train_analyses, valid_analyses)
    train_set2, valid_set2 = create_red2mel_datasets(train_analyses, valid_analyses)
    train_set3, valid_set3 = create_mel2acc_datasets(train_analyses, valid_analyses)

    train_set0.show(0)
    train_set1.show(450)
    train_set2.show(233)
    train_set3.show(233)
