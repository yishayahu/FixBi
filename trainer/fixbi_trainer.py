import os
import pickle
import random
import time
from pathlib import Path
import wandb
import matplotlib.cm as mplcm
import matplotlib.colors as mplcolors
import torch
import torch.nn as nn
import src.utils as utils
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from src.utils import params
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F

def get_best_match_aux(distss):
    n_clusters = len(distss)
    print('n_clusterss',n_clusters)
    res = linear_sum_assignment(distss)[1].tolist()
    targets = [None] *n_clusters
    for x,y in enumerate(res):
        targets[y] = x
    return targets


def get_best_match(sc, tc):
    dists = np.full((sc.shape[0],tc.shape[0]),fill_value=np.inf)
    for i in range(sc.shape[0]):
        for j in range(tc.shape[0]):
            dists[i][j] = np.mean((sc[i]-tc[j])**2)
    best_match = get_best_match_aux(dists.copy())

    return best_match

def train_fixbi(args, loaders, optimizers, models_sd, models_td, sp_params, losses, epoch):
    print("Epoch: [{}/{}]".format(epoch, args.epochs))
    start = time.time()
    src_train_loader, tgt_train_loader = loaders[0], loaders[1]
    optimizer_sd, optimizer_td = optimizers[0], optimizers[1]
    sp_param_sd, sp_param_td = sp_params[0], sp_params[1]
    ce, mse = losses[0], losses[1]

    utils.set_model_mode('train', models=models_sd)
    utils.set_model_mode('train', models=models_td)

    models_sd = nn.Sequential(*models_sd)
    models_td = nn.Sequential(*models_td)

    for step, (src_data, tgt_data) in enumerate(zip(src_train_loader, tgt_train_loader)):
        src_imgs, src_labels = src_data
        tgt_imgs, tgt_labels = tgt_data
        src_imgs, src_labels = src_imgs.cuda(non_blocking=True), src_labels.cuda(non_blocking=True)
        tgt_imgs, tgt_labels = tgt_imgs.cuda(non_blocking=True), tgt_labels.cuda(non_blocking=True)

        x_sd, x_td = models_sd(tgt_imgs), models_td(tgt_imgs)

        pseudo_sd, top_prob_sd, threshold_sd = utils.get_target_preds(args, x_sd)
        fixmix_sd_loss = utils.get_fixmix_loss(models_sd, src_imgs, tgt_imgs, src_labels, pseudo_sd, args.lam_sd)

        pseudo_td, top_prob_td, threshold_td = utils.get_target_preds(args, x_td)
        fixmix_td_loss = utils.get_fixmix_loss(models_td, src_imgs, tgt_imgs, src_labels, pseudo_td, args.lam_td)

        total_loss = fixmix_sd_loss + fixmix_td_loss

        if step == 0:
            print('Fixed MixUp Loss (SDM): {:.4f}'.format(fixmix_sd_loss.item()))
            print('Fixed MixUp Loss (TDM): {:.4f}'.format(fixmix_td_loss.item()))

        # Bidirectional Matching
        if epoch > args.bim_start:
            bim_mask_sd = torch.ge(top_prob_sd, threshold_sd)
            bim_mask_sd = torch.nonzero(bim_mask_sd).squeeze()

            bim_mask_td = torch.ge(top_prob_td, threshold_td)
            bim_mask_td = torch.nonzero(bim_mask_td).squeeze()

            if bim_mask_sd.dim() > 0 and bim_mask_td.dim() > 0:
                if bim_mask_sd.numel() > 0 and bim_mask_td.numel() > 0:
                    bim_mask = min(bim_mask_sd.size(0), bim_mask_td.size(0))
                    bim_td_loss = ce(x_td[bim_mask_sd[:bim_mask]], pseudo_sd[bim_mask_sd[:bim_mask]].to(os.environ['CUDA_VISIBLE_DEVICES']).detach())
                    bim_sd_loss = ce(x_sd[bim_mask_td[:bim_mask]], pseudo_td[bim_mask_td[:bim_mask]].to(os.environ['CUDA_VISIBLE_DEVICES']).detach())

                    total_loss += bim_sd_loss
                    total_loss += bim_td_loss

                    if step == 0:
                        print('Bidirectional Loss (SDM): {:.4f}'.format(bim_sd_loss.item()))
                        print('Bidirectional Loss (TDM): {:.4f}'.format(bim_td_loss.item()))

        # Self-penalization
        if epoch <= args.sp_start:
            sp_mask_sd = torch.lt(top_prob_sd, threshold_sd)
            sp_mask_sd = torch.nonzero(sp_mask_sd).squeeze()

            sp_mask_td = torch.lt(top_prob_td, threshold_td)
            sp_mask_td = torch.nonzero(sp_mask_td).squeeze()

            if sp_mask_sd.dim() > 0 and sp_mask_td.dim() > 0:
                if sp_mask_sd.numel() > 0 and sp_mask_td.numel() > 0:
                    sp_mask = min(sp_mask_sd.size(0), sp_mask_td.size(0))
                    sp_sd_loss = utils.get_sp_loss(x_sd[sp_mask_sd[:sp_mask]], pseudo_sd[sp_mask_sd[:sp_mask]], sp_param_sd)
                    sp_td_loss = utils.get_sp_loss(x_td[sp_mask_td[:sp_mask]], pseudo_td[sp_mask_td[:sp_mask]], sp_param_td)

                    total_loss += sp_sd_loss
                    total_loss += sp_td_loss

                    if step == 0:
                        print('Penalization Loss (SDM): {:.4f}', sp_sd_loss.item())
                        print('Penalization Loss (TDM): {:.4f}', sp_td_loss.item())

        # Consistency Regularization
        if epoch > args.cr_start:
            mixed_cr = 0.5 * src_imgs + 0.5 * tgt_imgs
            out_sd, out_td = models_sd(mixed_cr), models_td(mixed_cr)
            cr_loss = mse(out_sd, out_td)
            total_loss += cr_loss
            if step == 0:
                print('Consistency Loss: {:.4f}', cr_loss.item())

        optimizer_sd.zero_grad()
        optimizer_td.zero_grad()
        total_loss.backward()
        optimizer_sd.step()
        optimizer_td.step()

    print("Train time: {:.2f}".format(time.time() - start))


def train_only_source(args, src_train_loader, models_sd, ce, epoch,sched):
    print("Epoch: [{}/{}]".format(epoch, args.epochs))
    start = time.time()
    utils.set_model_mode('train', models=models_sd)
    losses = []
    models_sd = nn.Sequential(*models_sd)
    for step, src_data in tqdm(enumerate(src_train_loader),desc=f'epoch {epoch}'):
        optimizer = sched(current_step=step+ epoch*len(src_train_loader))
        src_imgs, src_labels,_ = src_data
        src_imgs, src_labels = src_imgs.to(os.environ['CUDA_VISIBLE_DEVICES'],non_blocking=True), src_labels.to(os.environ['CUDA_VISIBLE_DEVICES'],non_blocking=True)
        outputs = models_sd(src_imgs)
        loss = ce(outputs,src_labels)
        losses.append(float(loss))
        if step %10 == 0:
            print(f'loss {np.mean(losses)}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print("Train time: {:.2f}".format(time.time() - start))


class ClusteringTrainer:
    def __init__(self,loaders, models_sd, ce,idx_to_class,sched):
        self.sched = sched
        self.encoder,self.classfier = models_sd[0], models_sd[1]

        self.src_train_loader,self.tgt_train_loader = loaders
        self.n_clusters,self.dist_loss_lambda,self.src_dist_loss_lambda,self.acc_amount = params()
        self.slice_to_cluster = None
        self.source_clusters = None
        self.target_clusters = None
        self.best_matchs = None
        self.best_matchs_indexes = None
        self.accumulate_for_loss = []
        for _ in range(self.n_clusters):
            self.accumulate_for_loss.append([])
        self.slice_to_feature_source = {}
        self.slice_to_feature_target = {}
        self.slice_to_label_source = {}
        self.slice_to_label_target = {}

        self.epoch_cls_loss = []
        self.epoch_dist_loss = []
        self.epoch_src_dist_loss = []
        self.exp_dir = Path('vizviz1')
        self.exp_dir.mkdir(exist_ok=True)
        self.ce = ce
        self.use_dist_loss = False
        self.idx_to_class = idx_to_class
        self.out_dict = {}
        [x[1] for x in self.encoder.named_modules() if x[0] == '6'][0].register_forward_hook(utils.get_activation('layer3',self.out_dict))
        [x[1] for x in self.encoder.named_modules() if x[0] == '5'][0].register_forward_hook(utils.get_activation('layer2',self.out_dict))
        self.step = 0

    def train_clustering(self, epoch):
        vizviz = {}
        self.epoch_cls_loss = []
        self.epoch_dist_loss = []
        self.epoch_src_dist_loss = []
        # freeze_model(models_sd,exclude_layers = ['init_path', 'down','bottleneck.0','bottleneck.1','bottleneck.2','bottleneck.3.conv_path.0','out_path'])


        for step, (src_data, tgt_data) in tqdm(enumerate(zip(self.src_train_loader, self.tgt_train_loader)),desc=f'epoch {epoch}',total=min(len(self.src_train_loader),len(self.tgt_train_loader))):
            optimizer = self.sched(current_step=self.step)
            if self.best_matchs is None:
                self.encoder.eval()
                self.classfier.eval()
            else:
                self.encoder.train()
                self.classfier.train()

            if step ==0 and epoch % 15 == 2:
                self.source_clusters = []
                self.target_clusters = []
                self.accumulate_for_loss = []
                for _ in range(self.n_clusters):
                    self.accumulate_for_loss.append([])
                    self.source_clusters.append([])
                    self.target_clusters.append([])
                p = PCA(n_components=30,random_state=42)
                t = TSNE(n_components=2,learning_rate='auto',init='pca',random_state=42)
                points = []
                source_items = list(self.slice_to_feature_source.items())
                random.shuffle(source_items)
                source_items = source_items[:min(800,len(source_items))]
                for _,feat in source_items:
                    points.append(feat)
                for _,feat in self.slice_to_feature_target.items():
                    points.append(feat)

                points = np.array(points)
                points = points.reshape(points.shape[0],-1)


                print('doing tsne')

                points = p.fit_transform(points)
                points = t.fit_transform(points)
                source_points,target_points = points[:len(source_items)],points[len(source_items):]

                k1 = KMeans(n_clusters=self.n_clusters,random_state=42,n_init=100,max_iter=1000,tol=1e-7)
                print('doing kmean 1')
                sc = k1.fit_predict(source_points)
                k2 = KMeans(n_clusters=self.n_clusters,random_state=42,n_init=100,max_iter=1000,tol=1e-7)
                print('doing kmean 2')
                tc = k2.fit_predict(target_points)
                print('getting best match')
                self.best_matchs_indexes=get_best_match(k1.cluster_centers_,k2.cluster_centers_)
                self.slice_to_cluster = {}

                # items_labels = list(self.slice_to_label_source.items())
                ## labels in every cluster
                # labels_in_cluster_source  =np.zeros((self.n_clusters,31))
                labels_in_cluster_target  =np.zeros((self.n_clusters,31))

                for i in range(len(source_items)):
                    self.source_clusters[sc[i]].append(source_items[i][1])
                    # labels_in_cluster_source[sc[i],items_labels[i][1]]+=1
                    self.slice_to_cluster[source_items[i][0]] = sc[i]
                items = list(self.slice_to_feature_target.items())
                items_labels = list(self.slice_to_label_target.items())
                target_items_amount = len(items)
                for i in range(len(self.slice_to_feature_target)):
                    self.slice_to_cluster[items[i][0]] = tc[i]
                    labels_in_cluster_target[tc[i],items_labels[i][1]]+=1
                # pickle.dump(labels_in_cluster_source, open('labels_in_cluster_source.p','wb'))
                pickle.dump(labels_in_cluster_target, open('labels_in_cluster_target.p','wb'))
                log_log = {'target_items_amount':target_items_amount}
                if self.n_clusters == 31:
                    inds = linear_sum_assignment(labels_in_cluster_target,maximize=True)
                    cluster_base_acc_target = np.sum(labels_in_cluster_target[inds]) / np.sum(labels_in_cluster_target)
                    log_log['cluster_base_acc_target'] = cluster_base_acc_target
                    counter = 0
                    for i in range(len(self.slice_to_feature_target)):
                        if counter >=10:
                            break
                        lbl = int(items_labels[i][1])
                        cluster_num =  self.slice_to_cluster[items[i][0]]
                        predicted_lbl = int(inds[1][cluster_num])
                        if predicted_lbl != lbl:
                            impath = os.path.join('/home/dsi/shaya/office31/dslr/images/',items[i][0].split('_frame')[0],'frame'+items[i][0].split('_frame')[1])
                            log_log[f'bad_examples/predicted_{self.idx_to_class[predicted_lbl]}_real_{self.idx_to_class[lbl]}'] = wandb.Image(impath)
                            counter+=1
                for i in range(len(self.source_clusters)):
                    self.source_clusters[i] = np.mean(self.source_clusters[i],axis=0)
                self.best_matchs = []
                for i in range(len(self.best_matchs_indexes)):
                    self.best_matchs.append(torch.tensor(self.source_clusters[self.best_matchs_indexes[i]]))

                cm = plt.get_cmap('gist_rainbow')
                cNorm  = mplcolors.Normalize(vmin=0, vmax=self.n_clusters-1)
                scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
                colors = []
                for i in range(self.n_clusters):
                    colors.append(scalarMap.to_rgba(i))
                # colors = ['black','blue','cyan','red','orange'
                #     ,'tomato','lime','gold','magenta','dodgerblue'
                #     ,'peru','grey','brown','olive','navy'
                #     ,'blueviolet','darkgreen','maroon','yellow','cadetblue']
                im_path_source =str(self.exp_dir /  f'{step}_source.png')
                fig = plt.figure()
                ax = fig.add_subplot()
                curr_colors = []
                curr_points_x = []
                curr_points_y = []
                for i, slc_name in enumerate(self.slice_to_feature_source.keys()):
                    curr_points_x.append(source_points[i][0])
                    curr_points_y.append(source_points[i][1])
                    curr_colors.append(colors[self.slice_to_cluster[slc_name]])
                ax.scatter(curr_points_x,curr_points_y,marker = '.',c=curr_colors)
                plt.savefig(im_path_source)
                plt.cla()
                plt.clf()
                plt.close()
                im_path_target = str(self.exp_dir /  f'{step}_target.png')

                fig = plt.figure()
                ax = fig.add_subplot()
                curr_colors = []
                curr_points_x = []
                curr_points_y = []
                for i, slc_name in enumerate(self.slice_to_feature_target.keys()):
                    curr_points_x.append(target_points[i][0])
                    curr_points_y.append(target_points[i][1])
                    curr_colors.append(colors[self.best_matchs_indexes[self.slice_to_cluster[slc_name]]])
                ax.scatter(curr_points_x,curr_points_y,marker = '.',c=curr_colors)
                plt.savefig(im_path_target)
                plt.cla()
                plt.clf()
                plt.close()
                im_path_clusters = str(self.exp_dir /  f'{step}_clusters.png')
                fig = plt.figure()
                ax = fig.add_subplot()

                for i,(p,marker) in enumerate([(k1.cluster_centers_,'.'),(k2.cluster_centers_,'^')]):
                    if i ==0:
                        ax.scatter(p[:,0],p[:,1],marker = marker,c=colors[:len(p)])
                    else:
                        ax.scatter(p[:,0],p[:,1],marker = marker,c=[colors[self.best_matchs_indexes[i]] for i in range(len(p))])
                plt.savefig(im_path_clusters)
                plt.cla()
                plt.clf()
                plt.close()

                self.source_clusters = torch.tensor(self.source_clusters)
                self.slice_to_feature_source = {}
                self.slice_to_feature_target = {}
                self.slice_to_label_source = {}
                self.slice_to_label_target = {}

                log_log.update({f'figs/source': wandb.Image(im_path_source),f'figs/target': wandb.Image(im_path_target),
                           f'figs/cluster': wandb.Image(im_path_clusters)})
                wandb.log(log_log,step=epoch*len(self.tgt_train_loader))
            log_log = {}

            images, labels,imnames = src_data
            images = Variable(images).to(os.environ['CUDA_VISIBLE_DEVICES'])
            labels = labels.to(os.environ['CUDA_VISIBLE_DEVICES'])
            features = self.encoder(images)
            outputs = self.classfier(features)
            loss_cls = self.ce(outputs,labels)

            features = features.flatten(1)
            features = torch.cat([features,self.out_dict['layer3'].flatten(1),self.out_dict['layer2'].flatten(1)],dim=1)
            labels = labels.detach().cpu().numpy()
            src_dist_loss = torch.tensor(0.0,device=os.environ['CUDA_VISIBLE_DEVICES'])
            for imname,feature,img,lbl in zip(imnames,features,images,labels):
                self.slice_to_feature_source[imname] = feature.detach().cpu().numpy()
                self.slice_to_label_source[imname] = lbl
                if self.best_matchs is not None and imname in self.slice_to_cluster:
                    src_cluster = self.slice_to_cluster[imname]
                    # other_clusters  = self.source_clusters.clone().detach()
                    # other_clusters[src_cluster] = torch.inf
                    # ll = torch.mean((feature - self.source_clusters.to(os.environ['CUDA_VISIBLE_DEVICES'])) **2)
                    #
                    # src_dist_loss-= ll
                    if f'source_{src_cluster}' not in vizviz or len(vizviz[f'source_{src_cluster}']) < 3:
                        if f'source_{src_cluster}' not in vizviz:
                            vizviz[f'source_{src_cluster}'] = []
                        vizviz[f'source_{src_cluster}'].append(None)
                        im_path =  str(self.exp_dir / f'source_{src_cluster}_{step}_{len(vizviz[f"source_{src_cluster}"])}.png')
                        plt.imsave(im_path,  np.array(img[1].detach().cpu()), cmap='gray')
                        log_log[f'{src_cluster}/source_{len(vizviz[f"source_{src_cluster}"])}'] = wandb.Image(im_path)



            images, labels,imnames = tgt_data
            images = Variable(images).to(os.environ['CUDA_VISIBLE_DEVICES'])
            labels = labels
            features = self.encoder(images)
            outputs = self.classfier(features)
            features = features.flatten(1)
            features = torch.cat([features,self.out_dict['layer3'].flatten(1),self.out_dict['layer2'].flatten(1)],dim=1)
            probs = F.softmax(outputs,-1)
            max_probs = torch.max(probs,dim=1)[0]
            dist_loss = torch.tensor(0.0,device=os.environ['CUDA_VISIBLE_DEVICES'])
            for imname,feature,img,lbl,max_prob in zip(imnames,features,images,labels,max_probs):
                # if max_prob < 0.85:
                #     continue
                self.slice_to_feature_target[imname] = feature.detach().cpu().numpy()
                self.slice_to_label_target[imname] = lbl.detach().cpu().numpy()
                if self.best_matchs is not None and  imname in self.slice_to_cluster:
                    self.accumulate_for_loss[self.slice_to_cluster[imname]].append(feature)
                    src_cluster = self.best_matchs_indexes[self.slice_to_cluster[imname]]
                    if f'target_{src_cluster}' not in vizviz or len(vizviz[f'target_{src_cluster}']) < 3:
                        if f'target_{src_cluster}' not in vizviz:
                            vizviz[f'target_{src_cluster}'] = []
                        vizviz[f'target_{src_cluster}'].append(None)
                        im_path =  str(self.exp_dir /  f'target_{src_cluster}_{step}_{len(vizviz[f"target_{src_cluster}"])}.png')
                        plt.imsave(im_path,  np.array(img[1].detach().cpu()), cmap='gray')
                        log_log[f'{src_cluster}/target_{len(vizviz[f"target_{src_cluster}"])}'] = wandb.Image(im_path)

            self.use_dist_loss = False
            lens1 = [len(x) for x in self.accumulate_for_loss]
            if np.sum(lens1) > self.acc_amount:
                self.use_dist_loss = True
            if self.use_dist_loss:
                total_amount = 0
                dist_losses = [0] * len(self.accumulate_for_loss)
                for i,features in enumerate(self.accumulate_for_loss):
                    if len(features) > 0:
                        curr_amount = len(features)
                        total_amount+=curr_amount
                        features = torch.mean(torch.stack(features),dim=0)
                        dist_losses[i] = torch.mean((features - self.best_matchs[i].to(os.environ['CUDA_VISIBLE_DEVICES']))**2) * curr_amount
                        self.accumulate_for_loss[i] = []
                for l  in dist_losses:
                    if l >0:
                        dist_loss+=l
                        dist_loss/= total_amount
            if float(dist_loss) > 0:
                dist_loss*=self.dist_loss_lambda
                self.epoch_dist_loss.append(float(dist_loss))
            if float(src_dist_loss) < 0:
                src_dist_loss*=self.src_dist_loss_lambda
                self.epoch_src_dist_loss.append((float(src_dist_loss)))
            self.epoch_cls_loss.append(float(loss_cls))

            losses_dict = {'cls_loss': loss_cls,'dist_loss':dist_loss,'src_dist_loss':src_dist_loss,'total':loss_cls+dist_loss}

            if self.use_dist_loss:
                losses_dict['total'].backward()
                optimizer.step()

                optimizer.zero_grad()
            elif self.best_matchs is None:
                pass
                # losses_dict['seg_loss'].backward()
                # optimizer.step()
                # scheduler.step()
                # optimizer.zero_grad()
            else:
                losses_dict['cls_loss'].backward(retain_graph=True)

            log_log['cls_loss'] = float(np.mean(self.epoch_cls_loss))

            if self.epoch_dist_loss:
                log_log['dist_loss'] = float(np.mean(self.epoch_dist_loss))
            if self.epoch_src_dist_loss:
                log_log['src_dist_loss'] = float(np.mean(self.epoch_src_dist_loss))

            wandb.log(log_log,step=step + (epoch*len(self.tgt_train_loader)))

