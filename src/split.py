import os
import shutil
import random

def main():
    split_ratio = 0.2
    test_dir = "data/test"
    train_dir = "data/train"
    data_dir = "data/Sara_dataset"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    file_list = os.listdir(data_dir)

    test_size = int(split_ratio * len(file_list))

    random.shuffle(file_list)

    for file_name in file_list[:test_size]:
        source_path = os.path.join(data_dir, file_name)
        target_path = os.path.join(test_dir, file_name)
        shutil.move(source_path, target_path)

    for file_name in file_list[test_size:]:
        source_path = os.path.join(data_dir, file_name)
        target_path = os.path.join(train_dir, file_name)
        shutil.move(source_path, target_path)

    shutil.rmtree(data_dir)


if __name__ == '__main__':
    main()
