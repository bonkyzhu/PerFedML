import os
import yaml


def gen_command(
        strategy,
        yamlfile,
        filedir='../test',
        use_wandb=True,
        device='cuda:1',
        beta=None,
        num_classes_per_client=None,
        num_shards_per_client=None,
        noniid_type=None):
    is_existed = check_existence(strategy, yamlfile, beta, num_classes_per_client, num_shards_per_client)
    if beta is not None:
        command = f"python {filedir}/{strategy}/run.py --partition noniid-label-distribution --beta {beta} --yamlfile {yamlfile} --use_wandb {use_wandb} --device {device} &"
    elif num_classes_per_client is not None:
        command = f"python {filedir}/{strategy}/run.py --partition noniid-label-quantity --num_classes_per_client {num_classes_per_client} --yamlfile {yamlfile} --use_wandb {use_wandb} --device {device} &"
    elif num_shards_per_client is not None:
        command = f"python {filedir}/{strategy}/run.py --partition shards --num_shards_per_client {num_shards_per_client} --yamlfile {yamlfile} --use_wandb {use_wandb} --device {device} &"
    else:
        raise ValueError
    if is_existed:
        print(f"Task was already completed. Did not genertaed the task. Detailed task:\n {command}!")
        return None
    else:
        return command


def check_existence(strategy, yamlfile, beta, num_classes_per_client, num_shards_per_client):
    with open(yamlfile, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    server_config = config['server_config']
    client_config = config['client_config']
    num_clients = server_config['num_clients']
    if beta is not None:
        partition = 'noniid-label-distribution'
    if num_classes_per_client is not None:
        partition = 'noniid-label-quantity'
    if num_shards_per_client is not None:
        partition = 'shards'
    server_config['partition'] = partition
    server_config['beta'] = beta
    server_config['num_classes_per_client'] = num_classes_per_client
    server_config['num_shards_per_client'] = num_shards_per_client
    server_config['strategy'] = strategy

    if server_config['partition'] == 'noniid-label-distribution':
        partition_arg = f'beta:{beta}'
    elif server_config['partition'] == 'noniid-label-quantity':
        partition_arg = f'num_classes_per_client:{num_classes_per_client}'
    elif server_config['partition'] == 'shards':
        partition_arg = f'num_shards_per_client:{num_shards_per_client}'
    else:
        raise ValueError('not implemented')

    run_tag = f"./experiments_{server_config['strategy']}/{server_config['strategy']}_{server_config['dataset']}_{client_config['model']}_{server_config['partition']}_{partition_arg}_num-clients:{server_config['num_clients']}"
    file1 = run_tag + '_best_global_model.pkl'
    file2 = run_tag + '_final_server_obj.pkl'
    if os.path.exists(file1) or os.path.exists(file2):
        print(f'Find:{file1}')
        return True
    else:
        return False


if __name__ == "__main__":
    import os
    os.system('rm *.sh')
    # print(gen_command(strategy='FedAvg', yamlfile='./Cifar10_Conv2Cifar.yaml', beta='0.3b', num_classes_per_client=None))
    # yamlfile = './Cifar100_MobilenetV1Cifar.yaml'
    # yamlfile = './Cifar100_Conv2Cifar.yaml'
    planned, actual = 0, 0
    yamlfile_lst = ['./Cifar100_Conv2Cifar_verify.yaml']
    # yamlfile_lst = ['./Cifar10_Conv2Cifar.yaml', './Cifar100_Conv2Cifar.yaml']
    for run_type in ['beta']:
        for yamlfile in yamlfile_lst:
            # for (strategy, cuda) in [('FedAvg', 'cuda:0'), ('pFedUH', 'cuda:1'), ('FedNH', 'cuda:1')]:
            # for (strategy, cuda) in [('FedAvg', 'cuda:0'), ('FedPer', 'cuda:1'), ('Ditto', 'cuda:2')]:
            for strategy in ['FedAvg', 'FedPer', 'Ditto', 'pFedMe', 'FedROD', 'FedUH', 'FedNH']:
                if run_type == 'beta':
                    # for beta in ['0.3b', '0.5b', '1.0b']:
                    for beta in ['0.1', '0.3']:
                        cuda = f'cuda:{planned % 4}'
                        command = gen_command(strategy=strategy, yamlfile=yamlfile, device=cuda, beta=beta,
                                              num_classes_per_client=None, num_shards_per_client=None)

                        planned += 1
                        if command is not None:
                            actual += 1
                            if len(beta.split('b')) == 2:
                                filename = f'{strategy}_dir_b.sh'
                            else:
                                filename = f'{strategy}_dir.sh'
                            with open(filename, 'a') as f:
                                f.write(command + "\n")
                if run_type == 'shards':
                    if yamlfile.split('_')[0].split('/')[1] == 'Cifar10':
                        ss = [2, 5]
                    else:
                        ss = [10, 50]
                    for shards in ss:
                        cuda = f'cuda:{planned % 4}'
                        command = gen_command(strategy=strategy, yamlfile=yamlfile, device=cuda, beta=None,
                                              num_classes_per_client=None, num_shards_per_client=shards)
                        planned += 1
                        if command is not None:
                            actual += 1
                            with open(f'{strategy}_shards.sh', 'a') as f:
                                f.write(command + "\n")

    print(f"actual/planned:{actual}/{planned}")
