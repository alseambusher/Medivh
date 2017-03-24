import os
import pandas as pd
import numpy as np
import config


def caltech256():
    if not os.path.exists(config.trainset_path):
        if not os.path.exists(config.caltech_path):
            os.makedirs(config.caltech_path)
        image_dir_list = os.listdir(config.dataset_path)

        label_pairs = map(lambda x: x.split('.'), image_dir_list)
        labels, label_names = zip(*label_pairs)
        labels = map(lambda x: int(x), labels)

        label_dict = pd.Series( labels, index=label_names )
        label_dict -= 1
        n_labels = len( label_dict )

        image_paths_per_label = map(lambda one_dir: map(lambda one_file: os.path.join(config.dataset_path, one_dir, one_file ), os.listdir( os.path.join(config.dataset_path, one_dir))), image_dir_list)
        image_paths_train = np.hstack(map(lambda one_class: one_class[:-10], image_paths_per_label))
        image_paths_test = np.hstack(map(lambda one_class: one_class[-10:], image_paths_per_label))

        trainset = pd.DataFrame({'image_path': image_paths_train})
        testset  = pd.DataFrame({'image_path': image_paths_test })

        trainset = trainset[ trainset['image_path'].map( lambda x: x.endswith('.jpg'))]
        trainset['label'] = trainset['image_path'].map(lambda x: int(x.split('/')[-2].split('.')[0]) - 1)
        trainset['label_name'] = trainset['image_path'].map(lambda x: x.split('/')[-2].split('.')[1])

        testset = testset[ testset['image_path'].map( lambda x: x.endswith('.jpg'))]
        testset['label'] = testset['image_path'].map(lambda x: int(x.split('/')[-2].split('.')[0]) - 1)
        testset['label_name'] = testset['image_path'].map(lambda x: x.split('/')[-2].split('.')[1])

        label_dict.to_pickle(config.label_dict_path)
        trainset.to_pickle(config.trainset_path)
        testset.to_pickle(config.testset_path)
    else:
        trainset = pd.read_pickle(config.trainset_path)
        testset  = pd.read_pickle(config.testset_path)
        label_dict = pd.read_pickle(config.label_dict_path)
        n_labels = len(label_dict)