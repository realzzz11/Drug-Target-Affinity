import optuna
import os
import sys
import math
import pickle
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(code_dir)

from models.model_protBERT import BACPI
from utils import *
from data_processing.data_process import training_data_process
from torch.utils.tensorboard import SummaryWriter
import time
import types

args = argparse.ArgumentParser(description='Argparse for compound-protein interactions prediction')
args.add_argument('-task', type=str, default='affinity', help='affinity/interaction')
args.add_argument('-dataset', type=str, default='Kd', help='choose a dataset')
args.add_argument('-mode', type=str, default='gpu', help='gpu/cpu')
args.add_argument('-cuda', type=str, default='0', help='visible cuda devices')
args.add_argument('-verbose', type=int, default=1, help='0: do not output log in stdout, 1: output log')

# Hyper-parameter
args.add_argument('-lr', type=float, default=0.0005, help='init learning rate')
args.add_argument('-step_size', type=int, default=10, help='step size of lr_scheduler')
args.add_argument('-gamma', type=float, default=0.5, help='lr decay rate')
args.add_argument('-batch_size', type=int, default=16, help='batch size')
args.add_argument('-num_epochs', type=int, default=20, help='number of epochs')

# graph attention layer
args.add_argument('-gat_dim', type=int, default=50, help='dimension of node feature in graph attention layer')
args.add_argument('-num_head', type=int, default=3, help='number of graph attention layer head')
args.add_argument('-dropout', type=float, default=0.1)
args.add_argument('-alpha', type=float, default=0.1, help='LeakyReLU alpha')

args.add_argument('-comp_dim', type=int, default=80, help='dimension of compound atoms feature')
args.add_argument('-prot_dim', type=int, default=80, help='dimension of protein amino feature')
args.add_argument('-latent_dim', type=int, default=80, help='dimension of compound and protein feature')

args.add_argument('-window', type=int, default=5, help='window size of cnn model')
args.add_argument('-layer_cnn', type=int, default=3, help='number of layer in cnn model')
args.add_argument('-layer_out', type=int, default=3, help='number of output layer in prediction model')

params, _ = args.parse_known_args()

# 目标函数，用于Optuna的调参
def objective(trial):
    # 使用Optuna建议的超参数，动态替换
    params.lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    params.step_size = trial.suggest_int('step_size', 5, 20)
    params.gamma = trial.suggest_float('gamma', 0.1, 0.9)
    params.batch_size = trial.suggest_int('batch_size', 8, 64)
    params.num_epochs = trial.suggest_int('num_epochs', 30, 100)
    
    # graph attention layer
    params.gat_dim = trial.suggest_int('gat_dim', 16, 128)
    params.num_head = trial.suggest_int('num_head', 1, 8)
    params.dropout = trial.suggest_float('dropout', 0.1, 0.6)
    params.alpha = trial.suggest_float('alpha', 0.01, 0.2)

    # compound and protein features
    params.comp_dim = trial.suggest_int('comp_dim', 32, 128)
    params.prot_dim = trial.suggest_int('prot_dim', 32, 128)
    params.latent_dim = trial.suggest_int('latent_dim', 32, 128)

    # cnn window size and layers
    params.window = trial.suggest_int('window', 3, 10)
    params.layer_cnn = trial.suggest_int('layer_cnn', 1, 5)
    params.layer_out = trial.suggest_int('layer_out', 1, 5)

    # 打印当前试验的超参数组合到txt文件
    with open("./protBERT/optuna_protBERT_log.txt", "a") as f:
        f.write(f"Trial {trial.number}:\n")
        f.write(f"  lr: {params.lr}\n")
        f.write(f"  step_size: {params.step_size}\n")
        f.write(f"  gamma: {params.gamma}\n")
        f.write(f"  batch_size: {params.batch_size}\n")
        f.write(f"  num_epochs: {params.num_epochs}\n")
        f.write(f"  gat_dim: {params.gat_dim}\n")
        f.write(f"  num_head: {params.num_head}\n")
        f.write(f"  dropout: {params.dropout}\n")
        f.write(f"  alpha: {params.alpha}\n")
        f.write(f"  comp_dim: {params.comp_dim}\n")
        f.write(f"  prot_dim: {params.prot_dim}\n")
        f.write(f"  latent_dim: {params.latent_dim}\n")
        f.write(f"  window: {params.window}\n")
        f.write(f"  layer_cnn: {params.layer_cnn}\n")
        f.write(f"  layer_out: {params.layer_out}\n")

    # 定义data_dir，确保在 objective 函数中加载数据时使用
    data_dir = os.path.join('./datasets_ProtBERT', params.task, params.dataset) 
    # 加载数据
    print('Load data...')
    train_data = load_data(data_dir, 'train')
    train_data, dev_data = split_data(train_data, 0.1)
    print("protein:",train_data[3].shape)
    atom_dict = pickle.load(open(data_dir + '/atom_dict', 'rb'))

    # 创建并训练模型
    model = BACPI(params.task, len(atom_dict), params)
    model.to(device)
    
    # 调用 train_eval 传递 trial
    res = train_eval(model, params.task, train_data, dev_data, None, device, params, trial)

    # 根据任务类型返回评价指标
    if params.task == 'affinity':
        score = res[0]  # rmse为优化目标
    elif params.task == 'interaction':
        score = res[0]  # auc为优化目标

    # 打印结果到txt文件
    with open("./protBERT/optuna_protBERT_log.txt", "a") as f:
        f.write(f"  Result: {score}\n")
        f.write("\n")  # 每次试验结束后换行，便于区分

    # 返回用于优化的评价指标
    return score

    
def train_eval(model, task, data_train, data_dev=None, data_test=None, device=None, params=None, trial=None):
    """
    训练和评估模型。根据传递的验证集或测试集动态调整。
    
    Args:
        model: 需要训练的模型
        task: 当前任务 ('affinity' or 'interaction')
        data_train: 训练数据集
        data_dev: 验证数据集 (默认为None，可能不传递)
        data_test: 测试数据集 (默认为None，可能不传递)
        device: 使用的设备 ('cuda' or 'cpu')
        params: 模型超参数
        trial: Optuna中的trial对象，用于动态设置训练参数
    """
    writer = SummaryWriter(log_dir=os.path.join('protBERT/optuna_runs', f'experiment_{trial.number}'))
    
    if task == 'affinity':
        criterion = F.mse_loss
        best_res = float('inf')  # 初始化为无穷大，表示损失最小化
    elif task == 'interaction':
        criterion = F.cross_entropy
        best_res = 0  # 初始化为0，表示AUC最大化
    else:
        print("Please choose a correct task!!!")
        return 
    
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0.01, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)
    
    idx = np.arange(len(data_train[0]))
    batch_size = params.batch_size
    
    # 训练循环
    for epoch in range(params.num_epochs):
        np.random.shuffle(idx)
        model.train()
        pred_labels, predictions, labels = [], [], []
        epoch_loss = 0
        start_time = time.time()

        # 批次训练
        for i in range(math.ceil(len(data_train[0]) / batch_size)):
            batch_data = [data_train[di][idx[i * batch_size: (i + 1) * batch_size]] for di in range(len(data_train))]
            atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad, amino_mask, label = batch2tensor(batch_data, device)
            pred = model(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps)
            if task == 'affinity':
                loss = criterion(pred.float(), label.float())
                predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
                labels += label.cpu().numpy().reshape(-1).tolist()
            elif task == 'interaction':
                loss = criterion(pred.float(), label.view(label.shape[0]).long())
                ys = F.softmax(pred, 1).to('cpu').data.numpy()
                pred_labels += list(map(lambda x: np.argmax(x), ys))
                predictions += list(map(lambda x: x[1], ys))
                labels += label.cpu().numpy().reshape(-1).tolist()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if params.verbose:
                sys.stdout.write(f'\rEpoch {epoch}, Batch {i}/{math.ceil(len(data_train[0]) / batch_size) - 1}, Loss: {loss.item()}')
                sys.stdout.flush()

        # 记录每个epoch的运行时间    
        epoch_time = time.time() - start_time
        writer.add_scalar('Time/epoch', epoch_time, epoch)
        # 记录每个epoch的平均损失
        writer.add_scalar('Loss/train', epoch_loss / len(data_train[0]), epoch)

        # 如果有验证集，进行评估
        if data_dev is not None:
            if task == 'affinity':
                predictions, labels = np.array(predictions), np.array(labels)
                rmse_train, pearson_train, spearman_train = regression_scores(labels, predictions)
                print(f'Train RMSE: {rmse_train}, Pearson: {pearson_train}, Spearman: {spearman_train}')
                writer.add_scalar('RMSE/train', rmse_train, epoch)
                writer.add_scalar('Pearson/train', pearson_train, epoch)
                writer.add_scalar('Spearman/train', spearman_train, epoch)

                # 在验证集上进行评估
                rmse_dev, pearson_dev, spearman_dev = test(model, task, data_dev, batch_size, device)
                print(f'Dev RMSE: {rmse_dev}, Pearson: {pearson_dev}, Spearman: {spearman_dev}')
                writer.add_scalar('RMSE/dev', rmse_dev, epoch)
                writer.add_scalar('Pearson/dev', pearson_dev, epoch)
                writer.add_scalar('Spearman/dev', spearman_dev, epoch)
                
                if rmse_dev < best_res:
                    best_res = rmse_dev
                    res = [rmse_dev, pearson_dev, spearman_dev]

            else:  # 交互任务
                pred_labels, predictions, labels = np.array(pred_labels), np.array(predictions), np.array(labels)
                auc_train, acc_train, aupr_train = classification_scores(labels, predictions, pred_labels)
                print(f'Train AUC: {auc_train}, Accuracy: {acc_train}, AUPR: {aupr_train}')

                # 在验证集上进行评估
                auc_dev, acc_dev, aupr_dev = test(model, task, data_dev, batch_size, device)
                print(f'Dev AUC: {auc_dev}, Accuracy: {acc_dev}, AUPR: {aupr_dev}')

                # 选择最佳验证集模型
                if auc_dev > best_res:
                    best_res = auc_dev
                    torch.save(model, os.path.join('.protBERT/optuna_checkpoint', f'best_model_interaction_{trial.number}.pth'))
                    res = [auc_dev, acc_dev, aupr_dev]

        # 更新学习率调度器
        scheduler.step()

    # 如果有测试集，则在最后一轮训练后进行测试集评估
    if data_test is not None:
        print("Evaluating on test data...")
        if task == 'affinity':
            rmse_test, pearson_test, spearman_test = test(model, task, data_test, batch_size, device)
            print(f'Test RMSE: {rmse_test}, Pearson: {pearson_test}, Spearman: {spearman_test}')
            res = [rmse_test, pearson_test, spearman_test]
        else:
            auc_test, acc_test, aupr_test = test(model, task, data_test, batch_size, device)
            print(f'Test AUC: {auc_test}, Accuracy: {acc_test}, AUPR: {aupr_test}')
            res = [auc_test, acc_test, aupr_test]

    writer.close()
    return res

def test(model, task, data_test, batch_size, device):
    model.eval()
    predictions, pred_labels, labels = [], [], []
    for i in range(math.ceil(len(data_test[0]) / batch_size)):
        batch_data = [data_test[di][i * batch_size: (i + 1) * batch_size] for di in range(len(data_test))]
        atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad, amino_mask, label = batch2tensor(batch_data, device)
        with torch.no_grad():
            pred = model(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps)
        if task == 'affinity':
            predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += label.cpu().numpy().reshape(-1).tolist()
        else:
            ys = F.softmax(pred, 1).to('cpu').data.numpy()
            pred_labels += list(map(lambda x: np.argmax(x), ys))
            predictions += list(map(lambda x: x[1], ys))
            labels += label.cpu().numpy().reshape(-1).tolist()

    # 转换为 NumPy 数组
    predictions = np.array(predictions)
    labels = np.array(labels)

    if task == 'affinity':
        rmse_value, pearson_value, spearman_value = regression_scores(labels, predictions)
        return rmse_value, pearson_value, spearman_value
    else:
        auc_value, acc_value, aupr_value = classification_scores(labels, predictions, pred_labels)
        return auc_value, acc_value, aupr_value

if __name__ == '__main__':
    task = params.task
    dataset = params.dataset
    data_dir = os.path.join('./datasets_ProtBERT', task, dataset)
    if not os.path.isdir(data_dir):
        training_data_process(task, dataset)

    if params.mode == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = params.cuda
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("cuda is not available!!!")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    
    print(f'The code runs on the {device}')

    # 创建并优化调参任务
    study = optuna.create_study(direction='minimize' if task == 'affinity' else 'maximize')
    study.optimize(objective, n_trials=50)

    # 评估最佳trial的结果在test_data上
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 使用最佳模型评估test_data
    train_data = load_data(data_dir, 'train')  # 重新加载训练数据
    test_data = load_data(data_dir, 'test')  # 加载测试集
    atom_dict = pickle.load(open(data_dir + '/atom_dict', 'rb'))

    # 获取最佳 trial 的超参数，并重新训练模型
    best_params = trial.params
    best_params['verbose'] = 1
    # 使用最佳超参数创建模型
    best_model = BACPI(
        task=params.task,
        n_atom=len(atom_dict),
        params=types.SimpleNamespace(**best_params)  # 将最佳超参数打包在 params 中传递
    )

    # 优化器相关参数在训练过程中传递
    optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=best_params['step_size'], gamma=best_params['gamma'])

    # 将模型移到设备上
    best_model.to(device)

    # 使用最佳超参数重新训练模型，并在测试集上评估
    res = train_eval(
        model=best_model,
        task=params.task,
        data_train=train_data,
        data_dev=None,
        data_test=test_data,
        device=device,
        params=types.SimpleNamespace(**best_params),
        trial=trial
    )

    # 根据任务类型输出测试结果
    if params.task == 'affinity':
        print(f"Test RMSE: {res[0]}")
    elif params.task == 'interaction':
        print(f"Test AUC: {res[0]}")

    # 保存最终的最佳模型
    torch.save(best_model.state_dict(), './protBERT/optuna_checkpoint/best_model.pth')