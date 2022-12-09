'''
# File: run.py
# Project: APFL
# Created Date: 2022-02-18 11:54
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2022-06-07 2:43
# Modified By: Yutong Dai yutongdai95@gmail.com
# 
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''


import os
import torch
import torch.nn as nn
import yaml
import sys
import argparse

sys.path.append("../../")
sys.path.append("../")
from src.flbase.utils import setup_clients, resume_training
from src.utils import setup_seed, mkdirs, get_datasets

global wandb_installed
try:
    import wandb
    wandb_installed = True
except ModuleNotFoundError:
    wandb_installed = False


def run(partition, beta, num_classes_per_client, num_shards_per_client, yamlfile, use_wandb, keep_clients_model, device):
    from src.flbase.strategies.APFL import APFLClient, APFLServer
    use_wandb = wandb_installed and use_wandb
    setup_seed(2022)

    with open(yamlfile, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    server_config = config['server_config']
    client_config = config['client_config']
    num_clients = server_config['num_clients']
    server_config['strategy'] = 'APFL'
    server_config['partition'] = partition
    server_config['beta'] = beta
    server_config['num_classes_per_client'] = num_classes_per_client
    server_config['num_shards_per_client'] = num_shards_per_client
    client_config['normalize'] = False
    if server_config['partition'] == 'noniid-label-distribution':
        partition_arg = f'beta:{beta}'
    elif server_config['partition'] == 'noniid-label-quantity':
        partition_arg = f'num_classes_per_client:{num_classes_per_client}'
    elif server_config['partition'] == 'shards':
        partition_arg = f'num_shards_per_client:{num_shards_per_client}'
    else:
        raise ValueError('not implemented')

    run_tag = f"{server_config['strategy']}_{server_config['dataset']}_{client_config['model']}_{server_config['partition']}_{partition_arg}_num-clients:{server_config['num_clients']}"
    run_id = "debug"
    if use_wandb:
        run = wandb.init(config=config, name=run_tag, project='fl-benchmarks')
        run_id = run.id

    client_config_lst = [client_config for i in range(num_clients)]
    criterion = nn.CrossEntropyLoss()

    trainset, testset, _ = get_datasets(server_config['dataset'])

    clients_dict = setup_clients(APFLClient, trainset, None, criterion,
                                 client_config_lst, device,
                                 server_config=server_config,
                                 beta=server_config['beta'],
                                 num_classes_per_client=server_config['num_classes_per_client'],
                                 num_shards_per_client=server_config['num_shards_per_client'],
                                 )
    #  weight_init=torch.nn.init.normal_,  init_params={'mean':0.0, 'std':1.0}
    server = APFLServer(server_config, clients_dict, exclude=server_config['exclude'],
                          server_side_criterion=criterion, global_testset=testset, global_trainset=trainset,
                          client_cstr=APFLClient, server_side_client_config=client_config, server_side_client_device=device)
    directory = f"./experiments_{server_config['strategy']}/"
    mkdirs(directory)
    path = directory + run_tag
    print('results are saved in: ', path)
    server.run(filename=path + '_best_global_model.pkl', use_wandb=use_wandb, run_id=run_id)
    server.save(filename=path + '_final_server_obj.pkl', keep_clients_model=keep_clients_model)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test APFL.')
    parser.add_argument('--partition', default='noniid-label-distribution', type=str, help='strategy for partition the dataset')
    parser.add_argument('--beta', default=None, type=str, help='Dirichlet Distribution parameter')
    parser.add_argument('--num_classes_per_client', default=None, type=int, help='pathological non-iid parameter')
    parser.add_argument('--num_shards_per_client', default=None, type=int, help='pathological non-iid parameter')
    parser.add_argument('--yamlfile', default='./FashionMnsit.yaml', type=str, help='Configuration file.')
    parser.add_argument('--use_wandb', default=True, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='Use wandb pkg')
    parser.add_argument('--keep_clients_model', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='Keep APFL local model')
    parser.add_argument('--device', default='cuda:1', type=str, help='cuda device')
    args = parser.parse_args()
    run(args.partition, args.beta, args.num_classes_per_client,
        args.num_shards_per_client, args.yamlfile, args.use_wandb,
        args.keep_clients_model, args.device)
