import time
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F

import torchvision.models as models


def get_activation(name,out_dict):
    def hook(model, input, output):
        out_dict[name] = output
    return hook
class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
def optimizer_scheduler(optimizer,current_step,lr,total_steps):
    p = float(current_step) / total_steps
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1. + 10 * p) ** 0.75

    return optimizer


def params():
    n_clusters = 31
    dist_loss_lambda = 10
    src_dist_loss_lambda = 0.1
    acc_amount = 0
    return n_clusters,dist_loss_lambda,src_dist_loss_lambda,acc_amount

def get_data_info():
    resnet_type = 50
    num_classes = 31
    return num_classes, resnet_type


def get_net_info(num_classes,net_to_use='resnet101'):
    if net_to_use == 'resnet101':
        all_nets = models.resnet101(pretrained=True).to(os.environ['CUDA_VISIBLE_DEVICES'])
        fc = nn.Linear(2048, num_classes).to(os.environ['CUDA_VISIBLE_DEVICES'])
        net= nn.Sequential(all_nets.conv1,all_nets.bn1,all_nets.relu,all_nets.maxpool,all_nets.layer1,all_nets.layer2,all_nets.layer3,all_nets.layer4)
        classifier = nn.Sequential(all_nets.avgpool,Flatten(),fc)
    elif net_to_use == 'densenet121':
        all_nets = models.densenet121(pretrained=True).to(os.environ['CUDA_VISIBLE_DEVICES'])
        fc = nn.Linear(1024, num_classes).to(os.environ['CUDA_VISIBLE_DEVICES'])
        net = all_nets.features
        classifier = nn.Sequential(nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),Flatten(),fc)

    else:
        assert False
    # net = torch.nn.parallel.DataParallel(models.ResNet50().encoder).to(os.environ['CUDA_VISIBLE_DEVICES'])
    # classifier = torch.nn.parallel.DataParallel(nn.Linear(256, num_classes)).to(os.environ['CUDA_VISIBLE_DEVICES'])
    # head = torch.nn.parallel.DataParallel(models.Head()).to(os.environ['CUDA_VISIBLE_DEVICES'])

    return net, classifier


def get_train_info():
    lr = 0.001
    l2_decay = 5e-4
    momentum = 0.9
    nesterov = False
    return lr, l2_decay, momentum, nesterov


def load_net(args, net, classifier):
    print("Load pre-trained baseline model !")
    save_folder = args.baseline_path
    net.load_state_dict(torch.load(save_folder + '/net_sdm.pt',map_location=torch.device(os.environ['CUDA_VISIBLE_DEVICES'])), strict=False)
    classifier.load_state_dict(torch.load(save_folder + '/classifier_sdm.pt',map_location=torch.device(os.environ['CUDA_VISIBLE_DEVICES'])), strict=False)
    return net, classifier


def save_net(args, models, type):
    save_folder = args.save_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    net, classifier = models[0], models[1]

    torch.save(net.state_dict(), save_folder + '/' + 'net_' + str(type) + '.pt')
    torch.save(classifier.state_dict(), save_folder + '/' + 'classifier_' + str(type) + '.pt')


def set_model_mode(mode='train', models=None):
    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()


def evaluate(models, loader):
    start = time.time()
    total = 0
    correct = 0
    set_model_mode('eval', [models])
    with torch.no_grad():
        for step, tgt_data in enumerate(loader):
            tgt_imgs, tgt_labels,_ = tgt_data
            tgt_imgs, tgt_labels = tgt_imgs.to(os.environ['CUDA_VISIBLE_DEVICES'],non_blocking=True), tgt_labels.to(os.environ['CUDA_VISIBLE_DEVICES'],non_blocking=True)
            tgt_preds = models(tgt_imgs)
            # tgt_preds[:,21]-=5
            pred = tgt_preds.argmax(dim=1, keepdim=True)
            correct += pred.eq(tgt_labels.long().view_as(pred)).sum().item()
            total += tgt_labels.size(0)

    print('Accuracy: {:.2f}%'.format((correct / total) * 100))
    print("Eval time: {:.2f}".format(time.time() - start))
    set_model_mode('train', [models])


def get_sp_loss(input, target, temp):
    criterion = nn.NLLLoss(reduction='none').to(os.environ['CUDA_VISIBLE_DEVICES'])
    loss = torch.mul(criterion(torch.log(1 - F.softmax(input / temp, dim=1)), target.detach()), 1).mean()
    return loss


def get_target_preds(args, x):
    top_prob, top_label = torch.topk(F.softmax(x, dim=1), k=1)
    top_label = top_label.squeeze().t()
    top_prob = top_prob.squeeze().t()
    top_mean, top_std = top_prob.mean(), top_prob.std()
    threshold = top_mean - args.th * top_std
    return top_label, top_prob, threshold


def mixup_criterion_hard(pred, y_a, y_b, lam):
    criterion = nn.CrossEntropyLoss().to(os.environ['CUDA_VISIBLE_DEVICES'])
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_fixmix_loss(net, src_imgs, tgt_imgs, src_labels, tgt_pseudo, ratio):
    mixed_x = ratio * src_imgs + (1 - ratio) * tgt_imgs
    mixed_x = net(mixed_x)
    loss = mixup_criterion_hard(mixed_x, src_labels.detach(), tgt_pseudo.detach(), ratio)
    return loss


def final_eval(models_sd, models_td, tgt_test_loader):
    total = 0
    correct = 0
    set_model_mode('eval', [*models_sd])
    set_model_mode('eval', [*models_td])

    with torch.no_grad():
        for step, tgt_data in enumerate(tgt_test_loader):
            tgt_imgs, tgt_labels,_ = tgt_data
            tgt_imgs, tgt_labels = tgt_imgs.to(os.environ['CUDA_VISIBLE_DEVICES']), tgt_labels.to(os.environ['CUDA_VISIBLE_DEVICES'])
            pred_sd = F.softmax(models_sd(tgt_imgs), dim=1)
            pred_td = F.softmax(models_td(tgt_imgs), dim=1)
            softmax_sum = pred_sd + pred_td
            _, final_pred = torch.topk(softmax_sum, 1)
            correct += final_pred.eq(tgt_labels.long().view_as(final_pred)).sum().item()
            total += tgt_labels.size(0)

    print('Final Accuracy: {:.2f}%'.format((correct / total) * 100))
    set_model_mode('train', [*models_sd])
    set_model_mode('train', [*models_td])
