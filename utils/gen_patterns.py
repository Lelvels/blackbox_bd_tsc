import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder

def generative_pattern(noise_generator, x, y, y_target, poison_rate, clean_label, one_hot=False, exclude_target=False):
    x, y_backdoor = process_instances(x, y, y_target, poison_rate, clean_label, one_hot, exclude_target)
    #noise_generator.model.load_weights('./results/fcn_generator/mts_archive/ECG/generator_final.hdf5')

    pattern = noise_generator.model(x)
    pattern = (pattern - pattern.numpy().mean()) / pattern.numpy().std()
    data_std = np.resize(x.std(axis=1), (x.shape[0], 1, x.shape[2])).repeat(x.shape[1], axis=1)
    data_mean = np.resize(x.mean(axis=1), (x.shape[0], 1, x.shape[2])).repeat(x.shape[1], axis=1)
    x_backdoor = x.copy() + pattern * data_std + data_mean
    print(f'Generative rate: {poison_rate}')
    return x_backdoor, y_backdoor

def process_instances(x, y, y_target, poison_rate, clean_label, one_hot=False, exclude_target=False, only_target=False):
    y_classlabel = np.argmax(y, axis=1)
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(y_classlabel.reshape(-1, 1))

    if exclude_target:
        index_exclude = np.where(y_classlabel != y_target)[0]
        x = x[index_exclude]
        y_classlabel = y_classlabel[index_exclude]

    if clean_label:
        index = np.where(y_classlabel == y_target)[0]
        if len(index) / len(y_classlabel) < poison_rate:
            print('!!!ACTUAL POISON RATE:', len(index) / len(y_classlabel))

    else:
        index = np.where(y_classlabel != y_target)[0]
        if poison_rate < 1.0:
            index = np.random.choice(index, size=int(len(y_classlabel) * poison_rate), replace=False)

    y_backdoor = y_classlabel.copy()
    y_backdoor[index] = y_target

    if only_target:
        index_target = np.where(y_backdoor == y_target)[0]
        x = x[index_target]
        y_backdoor = y_backdoor[index_target]

    if one_hot:
        y_backdoor = enc.transform(y_backdoor.reshape(-1, 1)).toarray()

    return x, y_backdoor

def poison_random_data(x, y, y_target, poison_rate, one_hot=True):
    y_classlabel = np.argmax(y, axis=1)
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(y_classlabel.reshape(-1, 1))

    # Select random instances to poison
    if poison_rate < 1.0:
        index = np.random.choice(np.arange(x.shape[0]), size=int(x.shape[0] * poison_rate), replace=False)
        x_backdoor = x[index]
        y_backdoor = np.array([y_target]*len(index))
    else:
        x_backdoor = x
        y_backdoor = np.array([y_target]*x.shape[0])
    if one_hot:
        y_backdoor = enc.transform(y_backdoor.reshape(-1, 1)).toarray()
    return x_backdoor, y_backdoor

def gen_vanilla_pattern(x, y, y_target, poison_rate, clean_label, one_hot=False, exclude_target=False):
    # num of instance in target class < poison_rate * total num of instances
    INTENSITY = 0.02
    x, y_backdoor = process_instances(x, y, y_target, poison_rate, clean_label, one_hot, exclude_target)

    pattern_max = np.max(x, axis=1)
    pattern_max = pattern_max.reshape(pattern_max.shape[0], 1, pattern_max.shape[1])
    pattern_min = np.min(x, axis=1)
    pattern_min = pattern_min.reshape(pattern_min.shape[0], 1, pattern_min.shape[1])

    pattern = np.concatenate((pattern_max, pattern_min), axis=1)
    #pattern[:, 1, :] = -pattern[:, 1, :]
    pattern = np.tile(pattern, (int(INTENSITY * x.shape[1] / 2), 1))
    x_backdoor = x.copy()
    x_backdoor[:, 0:int(INTENSITY * x.shape[1] / 2) * 2, :] = pattern

    return x_backdoor, y_backdoor

def gen_powerline_noise(x, y, y_target, poison_rate, clean_label, one_hot=False, exclude_target=False):
    PATTERN_FILE = './powerline_pattern.npy'
    x, y_backdoor = process_instances(x, y, y_target, poison_rate, clean_label, one_hot, exclude_target)
    pattern = np.load(PATTERN_FILE)
    pattern = (pattern - np.mean(pattern)) / np.std(pattern)

    if x.shape[1] < pattern.shape[0] * 5:
        pattern = pattern[::pattern.shape[0] // x.shape[1] * 5, 0]
    pattern = np.resize(pattern, (1, x.shape[1], 1)).repeat(x.shape[2], axis=2).repeat(x.shape[0], axis=0)
    normal_mul = (np.max(x, axis=1) - np.min(x, axis=1)).reshape(x.shape[0], 1, x.shape[2]).repeat(pattern.shape[1],
                                                                                                   axis=1) / 10

    pattern *= normal_mul
    x_backdoor = x.copy() + pattern

    return x_backdoor, y_backdoor
