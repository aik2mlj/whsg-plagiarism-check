from data_utils.dataset import create_train_and_valid_analyses

if __name__ == "__main__":
    train_analyses, valid_analyses = create_train_and_valid_analyses()
    print(len(train_analyses))

    

