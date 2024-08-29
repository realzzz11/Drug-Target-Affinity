import os
import tarfile
import pandas as pd


def extract_data(measure, file):
    data = pd.read_csv(file, sep='\t', usecols=list(range(49)), dtype=str) #以制表符（\t）分隔的，并且只读取前49列的数据，将其作为字符串（str）加载
    data = data[['Ligand SMILES', 'BindingDB Target Chain  Sequence', 'pKi_[M]', 'pIC50_[M]', 'pKd_[M]', 'pEC50_[M]']]
    data.columns = ['SMILES', 'Sequence', 'Ki', 'IC50', 'Kd', 'EC50']
    data = data[['SMILES', 'Sequence', measure]] # 选择SMILES、Sequence和传入参数measure所指定的一列。
    data.to_csv(file + '.txt', index=None, header=None)


def affinity_data_prepare(dataset):
    data_dir = '../data/affinity/' + dataset
    if not os.path.isdir(data_dir):
        tar = tarfile.open(data_dir + '.tar.xz')
        os.mkdir(data_dir)
        for name in tar.getnames():
            tar.extract(name, '../data/affinity/')
        tar.close()

    extract_data(dataset, data_dir + '/train')
    extract_data(dataset, data_dir + '/test')


def interaction_data_prepare(dataset):
    pass


def training_data_prepare(task, dataset):
    if task == 'affinity':
        affinity_data_prepare(dataset)
    else:
        interaction_data_prepare(dataset)


if __name__ == '__main__':
    for dataset in ['IC50', 'EC50', 'Ki', 'Kd']: #依次调用affinity_data_prepare函数
        affinity_data_prepare(dataset)

    for dataset in ['human', 'celegans']:
        interaction_data_prepare(dataset)
