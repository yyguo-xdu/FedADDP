import copy
import math
import os
import random
import sys

import torch.optim as optim
from tqdm import tqdm
from tqdm.auto import trange, tqdm

from data import *
from net import *
from utils import compute_fisher_diag, get_sigma0

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
num_clients = args.num_clients
global_epoch = args.global_epoch
batch_size = args.batch_size
target_epsilon = args.target_epsilon
target_delta = args.target_delta
dataset = args.dataset
user_sample_rate = args.user_sample_rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.store == True:
    saved_stdout = sys.stdout
    file = open(
        f'./txt/{args.dirStr}/'
        f'dataset {dataset} '
        f'--num_clients {num_clients} '
        f'--global_epoch {global_epoch} '
        f'--batch_size {batch_size} '
        f'--target_epsilon {target_epsilon} '
        f'--target_delta {target_delta} '
        f'--fisher_threshold {args.fisher_threshold} '
        f'--lambda_1 {args.lambda_1} '
        f'--lambda_2 {args.lambda_2} '
        f'--beta {args.beta} '
        f'--lr {args.lr} '
        f'--alpha {args.dir_alpha}'
        f'.txt'
        , 'a'
    )
    sys.stdout = file


def local_update(model, dataloader, global_model, sigma0, clipping_bound):
    fisher_threshold = args.fisher_threshold
    model = model.to(device)
    global_model = global_model.to(device)
    fisher_diag = compute_fisher_diag(model, dataloader)
    w_glob = [param.clone().detach() for param in global_model.parameters()]

    u_loc, v_loc = [], []
    for param, fisher_value in zip(model.parameters(), fisher_diag):
        u_param = (param * (fisher_value > fisher_threshold)).clone().detach()
        v_param = (param * (fisher_value <= fisher_threshold)).clone().detach()
        u_loc.append(u_param)
        v_loc.append(v_param)

    u_glob, v_glob = [], []
    for global_param, fisher_value in zip(global_model.parameters(), fisher_diag):
        u_param = (global_param * (fisher_value > fisher_threshold)).clone().detach()
        v_param = (global_param * (fisher_value <= fisher_threshold)).clone().detach()
        u_glob.append(u_param)
        v_glob.append(v_param)

    for u_param, v_param, model_param in zip(u_loc, v_glob, model.parameters()):
        model_param.data = u_param + v_param

    w_0 = [param.clone().detach() for param in model.parameters()]

    def custom_loss(outputs, labels, param_diffs, reg_type, v_clipping_bound):
        ce_loss = F.cross_entropy(outputs, labels)
        if reg_type == "R1":
            reg_loss = (args.lambda_1 / 2) * torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))

        elif reg_type == "R2":
            norm_diff = torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))
            reg_loss = (args.lambda_2 / 2) * torch.norm(norm_diff - v_clipping_bound)

        else:
            raise ValueError("Invalid regularization type")

        return ce_loss + reg_loss

    last_client_model = None
    last_client_model_update = [torch.zeros_like(param) for param in model.parameters()]
    optimizer1 = optim.Adam(model.parameters(), lr=args.lr)
    optimizer2 = optim.Adam(model.parameters(), lr=args.lr)
    u_gradients = [torch.zeros_like(param) for param in model.parameters()]  # List to store gradients for u
    v_gradients = [torch.zeros_like(param) for param in model.parameters()]  # List to store gradients for v

    for batch_id, (data, labels) in enumerate(dataloader):
        # 记录下上一个model
        last_client_model = copy.deepcopy(model)
        data, labels = data.to(device), labels.to(device)
        optimizer1.zero_grad()
        outputs = model(data)
        param_diffs = [u_new - u_old for u_new, u_old in zip(model.parameters(), w_glob)]
        loss = custom_loss(outputs, labels, param_diffs, "R1", 0)
        loss.backward()
        # 记录下u的grad
        with torch.no_grad():
            for grad, model_param, u_param in zip(u_gradients, model.parameters(), u_loc):
                model_param.grad *= (u_param != 0)
                grad.copy_(model_param.grad)
        optimizer1.step()

        optimizer2.zero_grad()
        outputs = model(data)
        param_diffs = [model_param - w_old for model_param, w_old in zip(model.parameters(), w_glob)]
        loss = custom_loss(outputs, labels, param_diffs, "R2", clipping_bound)
        loss.backward()
        # 记录下v的grad
        with torch.no_grad():
            for grad, model_param, v_param in zip(v_gradients, model.parameters(), v_glob):
                model_param.grad *= (v_param != 0)
                grad.copy_(model_param.grad)
        # 记录下上一个model的grad
        last_client_model_update = [args.lr * (u_grad + v_grad) for u_grad, v_grad in zip(u_gradients, v_gradients)]

        with torch.no_grad():
            for model_param, v_param in zip(model.parameters(), v_glob):
                model_param.grad *= (v_param != 0)
        optimizer2.step()

    # 返回加噪后的new_model,但本地保存的model是不加噪的
    new_model = copy.deepcopy(model)

    if args.no_clip:
        return new_model, 0
    beta = args.beta
    M = len(list(new_model.parameters()))
    new_clipping_bound = 0
    for (client_model_param, last_client_model_param, last_client_model_update_param, w_o_param) in zip(
            new_model.parameters(), last_client_model.parameters(), last_client_model_update,
            w_0):
        q = (beta / 2) * torch.abs(last_client_model_param - last_client_model_update_param)
        client_model_param.data = client_model_param.data.to(device)
        q = q.to(device)
        client_model_param.data = torch.clamp(client_model_param.data, min=w_o_param.data - q.data,
                                              max=w_o_param.data + q.data)
        delta_f = 2 * q
        new_clipping_bound += torch.norm(delta_f).item()
        if args.no_noise:
            continue
        omega_m = math.sqrt(M) * sigma0 * delta_f
        noise = torch.randn_like(client_model_param.data) * omega_m
        client_model_param.data += noise

    return new_model, new_clipping_bound


def test(client_model, client_testloader):
    client_model.eval()
    client_model = client_model.to(device)

    num_data = 0

    correct = 0
    with torch.no_grad():
        for data, labels in client_testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = client_model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            num_data += labels.size(0)

    accuracy = 100.0 * correct / num_data

    client_model = client_model.to('cpu')

    return accuracy


def main():
    mean_acc_s = []
    acc_matrix = []
    if dataset == 'CIFAR10':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_CIFAR10(args.dir_alpha, num_clients)
        clients_models = [cifar10Net() for _ in range(num_clients)]
        global_model = cifar10Net()
    elif dataset == 'FEMNIST':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_FEMNIST(num_clients)
        clients_models = [femnistNet() for _ in range(num_clients)]
        global_model = femnistNet()
    elif dataset == 'SVHN':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_SVHN(args.dir_alpha, num_clients)
        clients_models = [SVHNNet() for _ in range(num_clients)]
        global_model = SVHNNet()
    else:
        print('undifined dataset')
        quit()
    for client_model in clients_models:
        client_model.load_state_dict(global_model.state_dict())
    global_model.to(device)
    sigma0 = get_sigma0(global_epoch, target_epsilon, target_delta)
    print(f"sigma0:{sigma0}")
    if args.no_noise:
        sigma0 = 0
    clients_clipping_bound = [0 for i in range(num_clients)]

    for epoch in trange(global_epoch):
        sampled_client_indices = random.sample(range(num_clients), max(1, int(user_sample_rate * num_clients)))
        sampled_clients_models = [clients_models[i] for i in sampled_client_indices]
        sampled_clients_train_loaders = [clients_train_loaders[i] for i in sampled_client_indices]
        sampled_clients_test_loaders = [clients_test_loaders[i] for i in sampled_client_indices]
        noise_client_models = []
        clients_accuracies = []
        for idx, (client_model, client_trainloader, client_testloader, client_id) in enumerate(
                zip(sampled_clients_models, sampled_clients_train_loaders, sampled_clients_test_loaders,
                    sampled_client_indices)):
            if not args.store:
                tqdm.write(f'client:{idx + 1}/{args.num_clients}')
            noise_client_model, new_client_clipping_bound = local_update(client_model, client_trainloader, global_model,
                                                                         sigma0,
                                                                         clients_clipping_bound[client_id])
            noise_client_models.append(noise_client_model)
            clients_clipping_bound[client_id] = new_client_clipping_bound
            accuracy = test(client_model, client_testloader)
            clients_accuracies.append(accuracy)
        mean_acc_s.append(sum(clients_accuracies) / len(clients_accuracies))
        print(f"clients_accuracies:{clients_accuracies},mean:{mean_acc_s[-1]}")
        acc_matrix.append(clients_accuracies)
        sampled_client_data_sizes = [client_data_sizes[i] for i in sampled_client_indices]
        sampled_client_weights = [
            sampled_client_data_size / sum(sampled_client_data_sizes)
            for sampled_client_data_size in sampled_client_data_sizes
        ]

        aggregated_params = {name: torch.zeros_like(param.data, device=device) for name, param in
                             global_model.named_parameters()}
        # 然后，遍历每个被采样的客户端模型和相应的权重
        for client_model, weight in zip(noise_client_models, sampled_client_weights):
            # 对每个参数进行加权平均
            for name, param in client_model.named_parameters():
                # 使用PyTorch的in-place操作来更新参数，权重乘以客户端模型参数
                aggregated_params[name].data.add_(param.data.to(device), alpha=weight)
        # 更新全局模型的参数
        global_model.load_state_dict(aggregated_params)

    print(
        f'===============================================================\n'
        f'fedaddp\n'
        f'sigma0 : {sigma0}\n'
        f'beta: {args.beta}\n'
        f'mean accuracy : \n'
        f'{mean_acc_s}\n'
        f'acc matrix : \n'
        f'{torch.tensor(acc_matrix)}\n'
        f'===============================================================\n'
    )


if __name__ == '__main__':
    main()
