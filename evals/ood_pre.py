import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models.transform_layers as TL
from utils.utils import set_random_seed, normalize
from evals.evals import get_auroc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)
import tqdm

def eval_ood_detection(P, model, id_loader, ood_loaders_in,ood_loaders_out, ood_scores, train_loader=None, simclr_aug=None):
    auroc_dict = dict()
    auroc_dict_eucl = dict()

    for ood in ood_loaders_in.keys():
        auroc_dict[ood] = dict()
        auroc_dict_eucl[ood] = dict()


    assert len(ood_scores) == 1  # assume single ood_score for simplicity
    ood_score = ood_scores[0]

    base_path = os.path.split(P.load_path)[0]  # checkpoint directory

    prefix = f'{P.ood_samples}'
    if P.resize_fix:
        prefix += f'_resize_fix_{P.resize_factor}'
    else:
        prefix += f'_resize_range_{P.resize_factor}'

    prefix = os.path.join(base_path, f'feats_{prefix}')

    kwargs = {
        'simclr_aug': simclr_aug,
        'sample_num': P.ood_samples,
        'layers': P.ood_layer,
    }

    print('Pre-compute global statistics...')
    feats_train = get_features_prototypes(P, f'{P.dataset}_train', model, train_loader, prefix=prefix, **kwargs)  # (M, T, d)
    #feats_train = get_features(P, f'{P.dataset}_train', model, train_loader, prefix=prefix, **kwargs)  # (M, T, d)

    P.axis = []
    for f in feats_train['simclr'].chunk(P.K_shift, dim=1):
        axis = f.squeeze(1)# f.mean(dim=1)  # (M, d)
        #P.axis.append(normalize(axis, dim=1).to(device))
        P.axis.append(axis.to(device))
    print('axis size: ' + ' '.join(map(lambda x: str(len(x)), P.axis)))

    f_sim = [f.mean(dim=1) for f in feats_train['simclr'].chunk(P.K_shift, dim=1)]  # list of (M, d)
    f_shi = [f.mean(dim=1) for f in feats_train['shift'].chunk(P.K_shift, dim=1)]  # list of (M, 4)

    weight_sim = []
    weight_shi = []
    #for shi in range(P.K_shift):
    #    sim_norm = f_sim[shi].norm(dim=1)  # (M)
    #    shi_mean = f_shi[shi][:, shi]  # (M)
    #    weight_sim.append(1 / sim_norm.mean().item())
    #    weight_shi.append(1 / shi_mean.mean().item())

    if ood_score == 'simclr':
        P.weight_sim = [1]
        P.weight_shi = [0]
    elif ood_score == 'CSI':
        P.weight_sim = weight_sim
        P.weight_shi = weight_shi
    else:
        raise ValueError()

    print(f'weight_sim:\t' + '\t'.join(map('{:.4f}'.format, P.weight_sim)))
    print(f'weight_shi:\t' + '\t'.join(map('{:.4f}'.format, P.weight_shi)))

    print('Pre-compute features ALL...')
    for ood, ood_loader in ood_loaders_in.items():
        feats_id = get_features(P, P.dataset, model, ood_loader, prefix=prefix, **kwargs)  # (N, T, d)
        feats_ood = dict()
    for ood, ood_loader in ood_loaders_out.items():
        if ood == 'interp':
            feats_ood[ood] = get_features(P, ood, model, id_loader, interp=True, prefix=prefix, **kwargs)
        else:
            feats_ood[ood] = get_features(P, ood, model, ood_loader, prefix=prefix, **kwargs)

    print(f'Compute OOD scores ALL... (score: {ood_score})')
    scores_id, scores_eucl_id = get_scores(P, feats_id, ood_score)
    scores_id = scores_id.numpy()
    scores_eucl_id = scores_eucl_id.numpy()
    scores_ood = dict()
    scores_eucl_ood = dict()
    if P.one_class_idx is not None:
        one_class_score = []

    for ood, feats in feats_ood.items():
        scores_ood[ood],scores_eucl_ood[ood] = get_scores(P, feats, ood_score)
        scores_ood[ood] = scores_ood[ood].numpy()
        scores_eucl_ood[ood] = scores_eucl_ood[ood].numpy()

        auroc_dict[ood][ood_score],fpr = get_auroc(scores_id, scores_ood[ood])
        auroc_dict_eucl[ood][ood_score], fpr_eucl = get_auroc(scores_eucl_id, scores_eucl_ood[ood])

        if P.one_class_idx is not None:
            one_class_score.append(scores_ood[ood])

    if P.one_class_idx is not None:
        one_class_score = np.concatenate(one_class_score)
        one_class_total = get_auroc(scores_id, one_class_score)
        print(f'One_class_real_mean: {one_class_total}')

    if P.print_score:
        print_score(P.dataset, scores_id)
        print_score(P.dataset, scores_eucl_id)
        for ood, scores in scores_ood.items():
            print_score(ood, scores)
        for ood, scores in scores_eucl_ood.items():
            print_score(ood, scores)




    '''
    print('Pre-compute features 1-shot...')
    for ood, ood_loader in ood_loaders_in.items():
        feats_id = get_features_one_shot(P, P.dataset, model, ood_loader, prefix=prefix, **kwargs)  # (N, T, d)

    print(f'Compute OOD scores 1-shot... (score: {ood_score})')
    scores_id, scores_eucl_id = get_scores(P, feats_id, ood_score)
    scores_id = scores_id.numpy()
    scores_eucl_id = scores_eucl_id.numpy()
    scores_ood = dict()
    scores_eucl_ood = dict()

    for ood, feats in feats_ood.items():
        scores_ood[ood],scores_eucl_ood[ood] = get_scores(P, feats, ood_score)
        scores_ood[ood] = scores_ood[ood].numpy()
        scores_eucl_ood[ood] = scores_eucl_ood[ood].numpy()

        auroc_dict_one_shot[ood][ood_score],fpr_one_shot = get_auroc(scores_id, scores_ood[ood])
        auroc_dict_one_shot_eucl[ood][ood_score], fpr_one_shot_eucl = get_auroc(scores_eucl_id, scores_eucl_ood[ood])

    '''



    return auroc_dict,fpr,auroc_dict_eucl,fpr_eucl




def get_scores(P, feats_dict, ood_score):
    # convert to gpu tensor
    feats_sim = feats_dict['simclr'].to(device)
    feats_shi = feats_dict['shift'].to(device)
    N = feats_sim.size(0)

    # compute scores
    scores = []
    scores_eucl = []
    for f_sim, f_shi in zip(feats_sim, feats_shi):
        f_sim = [f.mean(dim=0, keepdim=True) for f in f_sim.chunk(P.K_shift)]  # list of (1, d)
        f_shi = [f.mean(dim=0, keepdim=True) for f in f_shi.chunk(P.K_shift)]  # list of (1, 4)

        score = 0
        score_eucl = 0

        for shi in range(P.K_shift):
            score += (f_sim[shi]/f_sim[shi].norm() * P.axis[shi]).sum(dim=1).max().item() * P.weight_sim[shi]
            #score += f_shi[shi][:, shi].item() * P.weight_shi[shi]

            #score_eucl += torch.norm(f_sim[shi]/f_sim[shi].norm() -P.axis[shi],dim=1).min().item() * P.weight_sim[shi]

            score_eucl += 1 / (torch.norm(f_sim[shi] - P.axis[shi], dim=1).min().item()) * P.weight_sim[shi]


        score = score / P.K_shift
        scores.append(score)
        scores_eucl.append(score_eucl)

    scores = torch.tensor(scores)
    scores_eucl = torch.tensor(scores_eucl)

    assert scores.dim() == 1 and scores.size(0) == N  # (N)
    return scores.cpu(),scores_eucl.cpu()


def get_features(P, data_name, model, loader, interp=False, prefix='',
                 simclr_aug=None, sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # load pre-computed features if exists
    feats_dict = dict()
    # for layer in layers:
    #     path = prefix + f'_{data_name}_{layer}.pth'
    #     if os.path.exists(path):
    #         feats_dict[layer] = torch.load(path)

    # pre-compute features and save to the path
    left = [layer for layer in layers if layer not in feats_dict.keys()]
    if len(left) > 0:
        _feats_dict = _get_features(P, model, loader, interp, P.dataset == 'imagenet',
                                    simclr_aug, sample_num, layers=left)

        for layer, feats in _feats_dict.items():
            path = prefix + f'_{data_name}_{layer}.pth'
            torch.save(_feats_dict[layer], path)
            feats_dict[layer] = feats  # update value

    return feats_dict




def get_features_prototypes(P, data_name, model, loader, interp=False, prefix='',
                 simclr_aug=None, sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # load pre-computed features if exists
    feats_dict = dict()
    # for layer in layers:
    #     path = prefix + f'_{data_name}_{layer}.pth'
    #     if os.path.exists(path):
    #         feats_dict[layer] = torch.load(path)

    # pre-compute features and save to the path
    left = [layer for layer in layers if layer not in feats_dict.keys()]
    if len(left) > 0:
        _feats_dict = _get_features_prototypes(P, model, loader, interp, P.dataset == 'imagenet',
                                    simclr_aug, sample_num, layers=left)

        for layer, feats in _feats_dict.items():
            path = prefix + f'_{data_name}_{layer}.pth'
            torch.save(_feats_dict[layer], path)
            feats_dict[layer] = feats  # update value

    return feats_dict



def _get_features_prototypes(P, model, loader, interp=False, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # check if arguments are valid
    assert simclr_aug is not None

    if imagenet is True:  # assume batch_size = 1 for ImageNet
        sample_num = 1

    # compute features in full dataset
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list
    labels = []
    list_of_indices = {}


    for i, (x, label) in enumerate(loader):

        labels = labels + (list(label.numpy()))
        x = x.to(device)  # gpu tensor

        # compute features in one batch

        if label.item() in list_of_indices.keys():
            list_of_indices[label.item()].append(i)
        else:
            list_of_indices[label.item()] = [i]



        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        for seed in range(sample_num):
            set_random_seed(seed)

            x_t = x # No shifting: SimCLR
            x_t = simclr_aug(x_t)


            # compute augmented features
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, output_aux = model(x_t, **kwargs)

            # add features in one batch
            for layer in layers:
                feats = output_aux[layer].cpu()
                if imagenet is False:
                    feats_batch[layer] += feats.chunk(P.K_shift)
                else:
                    feats_batch[layer] += [feats]  # (B, d) cpu tensor

        # concatenate features in one batch
        for key, val in feats_batch.items():
            if imagenet:
                feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
            else:
                feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)

        # add features in full dataset
        for layer in layers:
            feats_all[layer] += [feats_batch[layer]]

    # concatenate features in full dataset
    for key, val in feats_all.items():
        feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)

    # reshape order
    if imagenet is False:
        # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
        for key, val in feats_all.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all[key] = val


    prototypes_ = np.zeros((P.n_classes, 128), dtype=np.float32)
    cont = np.zeros(P.n_classes)


    for element,label in zip(feats_all['simclr'],labels):
        prototypes_[label] =   prototypes_[label]  + element.cpu().detach().numpy()
        cont[label] = cont[label] + 1

    for i,element in enumerate(prototypes_):
        prototypes_[i] = prototypes_[i]/cont[i]


    prototypes = {'simclr': torch.tensor(prototypes_).unsqueeze(1),'shift': torch.tensor(prototypes_).unsqueeze(1)}
    print('The number of known classes is ',P.n_classes)


    import random

    mean_five_random_images = np.zeros((P.n_classes, 128), dtype=np.float32)

    for key in list_of_indices.keys():
        five_indices = random.sample(list_of_indices[key], 5)
        for index in five_indices:
            data_s_feat = feats_all['simclr'][index]
            mean_five_random_images[key] = mean_five_random_images[key] + data_s_feat.cpu().detach().numpy()

    mean_five_random_images = mean_five_random_images / 5

    if P.five_shot_eval:
        print('Five-shot')
        mean_five_random_images_ = {'simclr': torch.tensor(mean_five_random_images).unsqueeze(1), 'shift': torch.tensor(mean_five_random_images).unsqueeze(1)}
        prototypes = mean_five_random_images_

    return prototypes


def get_features_one_shot(P, data_name, model, loader, interp=False, prefix='',
                 simclr_aug=None, sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # load pre-computed features if exists
    feats_dict = dict()
    # for layer in layers:
    #     path = prefix + f'_{data_name}_{layer}.pth'
    #     if os.path.exists(path):
    #         feats_dict[layer] = torch.load(path)

    # pre-compute features and save to the path
    left = [layer for layer in layers if layer not in feats_dict.keys()]
    if len(left) > 0:
        _feats_dict = _get_features_one_shot(P, model, loader, interp, P.dataset == 'imagenet',
                                    simclr_aug, sample_num, layers=left)

        for layer, feats in _feats_dict.items():
            path = prefix + f'_{data_name}_{layer}.pth'
            torch.save(_feats_dict[layer], path)
            feats_dict[layer] = feats  # update value

    return feats_dict



def get_features_five_shot(P, data_name, model, loader, interp=False, prefix='',
                 simclr_aug=None, sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # load pre-computed features if exists
    feats_dict = dict()
    # for layer in layers:
    #     path = prefix + f'_{data_name}_{layer}.pth'
    #     if os.path.exists(path):
    #         feats_dict[layer] = torch.load(path)

    # pre-compute features and save to the path
    left = [layer for layer in layers if layer not in feats_dict.keys()]
    if len(left) > 0:
        _feats_dict = _get_features_five_shot(P, model, loader, interp, P.dataset == 'imagenet',
                                    simclr_aug, sample_num, layers=left)

        for layer, feats in _feats_dict.items():
            path = prefix + f'_{data_name}_{layer}.pth'
            torch.save(_feats_dict[layer], path)
            feats_dict[layer] = feats  # update value

    return feats_dict


def _get_features(P, model, loader, interp=False, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # check if arguments are valid
    assert simclr_aug is not None

    if imagenet is True:  # assume batch_size = 1 for ImageNet
        sample_num = 1

    # compute features in full dataset
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list
    for i, (x, _) in enumerate(loader):
        if interp:
            x_interp = (x + last) / 2 if i > 0 else x  # omit the first batch, assume batch sizes are equal
            last = x  # save the last batch
            x = x_interp  # use interp as current batch

        if imagenet is True:
            x = torch.cat(x[0], dim=0)  # augmented list of x

        x = x.to(device)  # gpu tensor

        # compute features in one batch
        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        for seed in range(sample_num):
            set_random_seed(seed)

            if P.K_shift > 1:
                x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
            else:
                x_t = x # No shifting: SimCLR
            x_t = simclr_aug(x_t)

            # compute augmented features
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, output_aux = model(x_t, **kwargs)

            # add features in one batch
            for layer in layers:
                feats = output_aux[layer].cpu()
                if imagenet is False:
                    feats_batch[layer] += feats.chunk(P.K_shift)
                else:
                    feats_batch[layer] += [feats]  # (B, d) cpu tensor

        # concatenate features in one batch
        for key, val in feats_batch.items():
            if imagenet:
                feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
            else:
                feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)

        # add features in full dataset
        for layer in layers:
            feats_all[layer] += [feats_batch[layer]]

    # concatenate features in full dataset
    for key, val in feats_all.items():
        feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)

    # reshape order
    if imagenet is False:
        # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
        for key, val in feats_all.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all[key] = val

    return feats_all



def compute_mean_images_source_prototypes(P,sources_loader):
        # prepare structures to hold prototypes
        image_size = 224
        if P.dataset == 'PACS_DG':
            known_classes = 6

        else:
            exit('dataset not implemented ',P.dataset)

        prototypes = np.zeros((known_classes, 3,image_size,image_size), dtype=np.float32)
        count_labels = np.zeros((known_classes))

        # forward source data

        for it_s, (data_s, class_l_s) in enumerate(sources_loader):

            prototypes[class_l_s] += data_s.numpy()
            count_labels[class_l_s] += 1


        for idx in range(known_classes):
            prototypes[idx] = prototypes[idx]/count_labels[idx]

        # compute the pixel-wise distance between the mean of the images and all the source images
        all_diff = {}
        nearest_images = np.zeros((known_classes, 3,image_size,image_size), dtype=np.float32)
        for it_s, (data_s, class_l_s) in enumerate(sources_loader):

            diff = np.sum(abs(prototypes[class_l_s]-data_s.squeeze().numpy()))/(3*image_size*image_size)
            if not (class_l_s in all_diff.keys()):
                all_diff[class_l_s] = diff
                nearest_images[class_l_s] = data_s
            else:
                if diff < all_diff[class_l_s]:
                    all_diff[class_l_s] = diff
                    nearest_images[class_l_s] = data_s

        return nearest_images

def _get_features_one_shot(P, model, loader, interp=False, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # check if arguments are valid
    assert simclr_aug is not None

    if imagenet is True:  # assume batch_size = 1 for ImageNet
        sample_num = 1

    # compute features in full dataset
    model.eval()


    nearest_mean_images = compute_mean_images_source_prototypes(P,loader)
    feats_all = {layer: [] for layer in layers}  # initialize: empty list

    for i, x in enumerate(nearest_mean_images):

        x = torch.tensor(x).to(device).unsqueeze(0)  # gpu tensor

        # compute features in one batch
        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        for seed in range(sample_num):
            set_random_seed(seed)

            if P.K_shift > 1:
                x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
            else:
                x_t = x # No shifting: SimCLR

            x_t = simclr_aug(x_t)

            # compute augmented features
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, output_aux = model(x_t, **kwargs)

            # add features in one batch
            for layer in layers:
                feats = output_aux[layer].cpu()
                if imagenet is False:
                    feats_batch[layer] += feats.chunk(P.K_shift)
                else:
                    feats_batch[layer] += [feats]  # (B, d) cpu tensor

        # concatenate features in one batch
        for key, val in feats_batch.items():
            if imagenet:
                feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
            else:
                feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)

        # add features in full dataset
        for layer in layers:
            feats_all[layer] += [feats_batch[layer]]

    # concatenate features in full dataset
    for key, val in feats_all.items():
        feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)

    # reshape order
    if imagenet is False:
        # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
        for key, val in feats_all.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all[key] = val


    return feats_all



@torch.no_grad()
def _get_features_five_shot(P, model, loader, interp=False, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift')):
        # prepare structures to hold prototypes
        import random

        image_size = 224
        if P.dataset == 'PACS_DG':
            known_classes = 6

        else:
            exit('dataset not implemented ',P.dataset)

        mean_five_random_images = np.zeros((known_classes,128), dtype=np.float32)
        # forward source data

        list_of_indices = {}

        for indices, (data_s, class_l_s) in enumerate(loader):
            # forward

            if class_l_s.item() in list_of_indices.keys():
                list_of_indices[class_l_s.item()].append(indices.item())
            else:
                list_of_indices[class_l_s.item()] = [indices.item()]

        model.eval()
        feats_all = {layer: [] for layer in layers}  # initialize: empty list
        for key in list_of_indices.keys():
            five_indices = random.sample(list_of_indices[key], 5)
            for index in five_indices:
                image,_,_ = sources_loader.dataset.__getitem__(index)
                image = image.to(device)
                image = image.unsqueeze(0)
                if self.args.network == "resnet101_simclr":
                    self.contrastive_head.eval()
                    out, feats = self.model(image, apply_fc=True)
                    head_feat = self.contrastive_head(feats)
                    head_feat_list[key] = head_feat_list[key] + (head_feat.cpu() / head_feat.cpu().norm()).detach().numpy()
                else:
                    out, feats = self.model(image)

                mean_five_random_images[key] = mean_five_random_images[key]+feats.cpu().detach().numpy()

        mean_five_random_images = mean_five_random_images/5
        head_feat_list = head_feat_list/5

        return torch.tensor(mean_five_random_images),torch.tensor(head_feat_list)



def _get_features_five_shot(P, model, loader, interp=False, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # check if arguments are valid
    assert simclr_aug is not None

    if imagenet is True:  # assume batch_size = 1 for ImageNet
        sample_num = 1

    # compute features in full dataset
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list
    for i, (x, _) in enumerate(loader):


        x = x.to(device)  # gpu tensor

        # compute features in one batch
        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        for seed in range(sample_num):
            set_random_seed(seed)

            if P.K_shift > 1:
                x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
            else:
                x_t = x # No shifting: SimCLR
            x_t = simclr_aug(x_t)

            # compute augmented features
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, output_aux = model(x_t, **kwargs)

            # add features in one batch
            for layer in layers:
                feats = output_aux[layer].cpu()
                if imagenet is False:
                    feats_batch[layer] += feats.chunk(P.K_shift)
                else:
                    feats_batch[layer] += [feats]  # (B, d) cpu tensor

        # concatenate features in one batch
        for key, val in feats_batch.items():
            if imagenet:
                feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
            else:
                feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)

        # add features in full dataset
        for layer in layers:
            feats_all[layer] += [feats_batch[layer]]

    # concatenate features in full dataset
    for key, val in feats_all.items():
        feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)

    # reshape order
    if imagenet is False:
        # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
        for key, val in feats_all.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all[key] = val

    return feats_all

def print_score(data_name, scores):
    quantile = np.quantile(scores, np.arange(0, 1.1, 0.1))
    print('{:18s} '.format(data_name) +
          '{:.4f} +- {:.4f}    '.format(np.mean(scores), np.std(scores)) +
          '    '.join(['q{:d}: {:.4f}'.format(i * 10, quantile[i]) for i in range(11)]))

