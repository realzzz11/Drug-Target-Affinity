import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
from data_prepare import training_data_prepare

# 定义五个字典
# 它们都是defaultdict对象。默认情况下，这些字典在访问不存在的键时，会将该键的值设置为当前字典的长度，即键值对的数量。
atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
word_dict = defaultdict(lambda: len(word_dict))

# 接受一个分子对象mol作为输入
def create_atoms(mol):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()] # 从分子对象中获取所有原子的符号（如C、O、N等），并将它们存储在atoms列表中
    """相当于下面的简单写法
    atoms = []  # 初始化一个空的列表，用于存储原子符号
    for a in mol.GetAtoms():  # 遍历 mol.GetAtoms() 返回的每一个原子对象
        symbol = a.GetSymbol()  # 获取当前原子的符号，例如 'C', 'O', 'N' 等
        atoms.append(symbol)  # 将符号添加到 atoms 列表中
    """
    for a in mol.GetAromaticAtoms(): # 遍历分子中所有的芳香性原子
        i = a.GetIdx() # 获取当前芳香性原子的索引位置
        atoms[i] = (atoms[i], 'aromatic') # 将该原子的符号与'aromatic'字符串组合，存回到原位置
    atoms = [atom_dict[a] for a in atoms] # atoms 列表中的每个元素（原子符号）都会被替换为 atom_dict 字典中的对应值
    # atoms = ['C', 'O', 'N', 'C'] --> atoms = [0, 1, 2, 0]
    return np.array(atoms)


def create_ijbonddict(mol):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))

    atoms_set = set(range(mol.GetNumAtoms())) # mol.GetNumAtoms()-->分子中的原子总数, set() 用于创建一个集合(一种无序且不重复的元素集合)
    # 找到分子中所有孤立的原子为它们添加一个虚拟的键（'nan'）
    isolate_atoms = atoms_set - set(i_jbond_dict.keys()) 
    bond = bond_dict['nan']
    for a in isolate_atoms:
        i_jbond_dict[a].append((a, bond))

    return i_jbond_dict


def atom_features(atoms, i_jbond_dict, radius):
    # 如果分子只有一个原子或半径为0，则直接为每个原子分配一个指纹（即特征）
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]
    # 否则，根据指定的半径递归地计算原子的指纹。每个原子的指纹由其自身和邻居原子的特征组合而成，并存储在fingerprints列表中
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        # 逐步增加指纹计算时考虑的邻域范围
        for _ in range(radius):
            fingerprints = [] # 存储当前半径下每个原子的指纹
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            # 更新节点和边信息，以便在下一个半径计算中使用。
            nodes = fingerprints
            _i_jedge_dict = defaultdict(lambda: []) # 重置
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints) # 是通过计算得到的指纹

# 生成并返回分子的邻接矩阵（表示原子之间的连接关系），并在对角线位置上添加1，以表示每个原子与自身的连接。
def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    adjacency = np.array(adjacency)
    adjacency += np.eye(adjacency.shape[0], dtype=int)
    return adjacency

# 计算并返回分子的摩根指纹（以位向量形式表示），这个指纹用于表示分子的结构特征
def get_fingerprints(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=True)
    return fp.ToBitString()

# 将蛋白质序列两端加上特殊字符'-'和'='，然后将其分割为ngram大小的片段，并用word_dict字典将这些片段转换为索引数组返回
def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


def extract_input_data(input_path, output_path, radius, ngram):
    data = pd.read_csv(input_path + '.txt', header=None)
    compounds, adjacencies, fps, proteins, interactions = [], [], [], [], []

    for index in range(len(data)):
        smiles, sequence, interaction = data.iloc[index, :] # 将每行数据分为三个部分：化合物的 SMILES 字符串（smiles），蛋白质的序列（sequence），以及它们之间的交互（interaction）

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        compounds.append(atom_features(atoms, i_jbond_dict, radius))
        adjacencies.append(create_adjacency(mol))
        fps.append(get_fingerprints(mol))
        proteins.append(split_sequence(sequence, ngram))
        interactions.append(np.array([float(interaction)]))

    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, 'compounds'), compounds)
    np.save(os.path.join(output_path, 'adjacencies'), adjacencies)
    np.save(os.path.join(output_path, 'fingerprint'), fps)
    np.save(os.path.join(output_path, 'proteins'), proteins)
    np.save(os.path.join(output_path, 'interactions'), interactions)


def training_data_process(task, dataset):
    radius, ngram = 2, 3

    if not os.path.isdir(os.path.join('../data', task, dataset)):
        training_data_prepare(task, dataset)

    for name in ['train', 'test']:
        input_path = os.path.join('../data', task, dataset, name)
        output_path = os.path.join('../datasets', task, dataset, name)
        extract_input_data(input_path, output_path, radius, ngram)

    dump_dictionary(fingerprint_dict, os.path.join('../datasets', task, dataset, 'atom_dict'))
    dump_dictionary(word_dict, os.path.join('../datasets', task, dataset, 'amino_dict'))
