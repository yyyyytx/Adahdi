from .base_strategy import BaseStrategy
import numpy as np
import torch
from sklearn.metrics import pairwise_distances
import torch.nn.functional as F
from scipy import stats

class kCenterGreedy():

  def __init__(self, features, metric='euclidean'):
    # self.X = X
    # self.y = y
    # self.flat_X = self.flatten_X()
    self.name = 'kcenter'
    self.features = features
    self.metric = metric
    self.min_distances = None
    self.n_obs = self.features.shape[0]
    self.already_selected = []

  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
    """Update min distances given cluster centers.

    Args:
      cluster_centers: indices of cluster centers
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    if reset_dist:
      self.min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
    if cluster_centers:

      x = self.features[cluster_centers]

      dist = pairwise_distances(self.features, x, metric=self.metric)
      if self.min_distances is None:
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.min_distances = np.minimum(self.min_distances, dist)
    # print('min_distances:',self.min_distances)

  def select_batch_(self, already_selected, N, **kwargs):
    try:
      # Assumes that the transform function takes in original data and not
      # flattened data.
      # print('Getting transformed features...')
      # self.features = model.transform(self.X)
      print('Calculating distances...')
      self.update_distances(already_selected, only_new=False, reset_dist=True)
    except:
      print('Using flat_X as features.')
      self.update_distances(already_selected, only_new=True, reset_dist=False)

    new_batch = []

    for i in range(N):
      if self.already_selected is None:
        # Initialize centers with a randomly selected datapoint
        ind = np.random.choice(np.arange(self.n_obs))
      else:
        ind = np.argmax(self.min_distances)
      # New examples should not be in already selected since those points
      # should have min_distance of zero to a cluster center.

      assert ind not in already_selected

      self.update_distances([ind], only_new=True, reset_dist=False)
      new_batch.append(ind)
    print('Maximum distance from cluster centers is %0.2f'
            % max(self.min_distances))

    self.already_selected = already_selected

    return new_batch


class SingleCIRMSampling(BaseStrategy):

    def query(self, n_select):
        # centers = self.trainer.center_loss.centers.detach().cpu()
        # norm_centers = self.l2_norm(centers)
        unlabel_loader, unlabel_ind = self.build_unlabel_loader()
        label_loader, label_ind = self.build_labeled_loader()
        u_embedding_features, u_true_labels, u_ds_inds = self.predict_probs_and_embed(unlabel_loader)
        l_embedding_features, l_true_labels, l_ds_inds = self.predict_probs_and_embed(label_loader)
        centers = []
        for cls in range(self.net.n_label):
            # print(torch.mean(l_embedding_features[l_true_labels == cls], dim=0).shape)
            centers.append(torch.mean(l_embedding_features[l_true_labels == cls], dim=0).unsqueeze(dim=0))
        centers = torch.cat(centers, dim=0)
        # print(centers.shape)

        u_data_ind = torch.arange(len(u_embedding_features))

        # l_mask = torch.zeros((len(l_embedding_features), self.net.n_label), dtype=torch.bool)
        # l_mask = torch.scatter(l_mask, 1, l_true_labels.unsqueeze(dim=0).T, True)

        # l_embedding_features = self.l2_norm(l_embedding_features)
        l_dist = torch.cdist(l_embedding_features, centers)
        # l_dist = torch.cosine_similarity(l_embedding_features.unsqueeze(1), norm_centers.unsqueeze(0), dim=-1)
        # l_centers = centers[l_true_labels]
        # print(l_cen_norm)
        # l_dist = (l_cen_norm * l_embedding_features).sum(dim=-1)
        # print(l_dist.shape)

        # torch.cosine_similarity(strong_feats.unsqueeze(1), center_feats.unsqueeze(0), dim=-1).detach()

        # l_dist =

        cls_dist_t = []
        for cls in range(self.net.n_label):
            cls_dist = l_dist[:, cls]
            cls_mask = l_true_labels == cls
            cls_true_dist = cls_dist[cls_mask]
            cls_false_dist = cls_dist[~cls_mask]
            min_false = torch.max(cls_false_dist)
            tmp_flag = cls_true_dist < min_false
            max_cls_true_dist = torch.max(cls_true_dist[tmp_flag])
            cls_dist_t.append(max_cls_true_dist)
        cls_dist_t = torch.tensor(cls_dist_t)
        cls_dist_t = torch.unsqueeze(cls_dist_t, dim=0).repeat((len(u_embedding_features), 1))

        print(cls_dist_t)
        u_dist = torch.cdist(u_embedding_features, centers)

        flag = u_dist < cls_dist_t
        flag_sum = torch.sum(flag, dim=1)
        mask = flag_sum != 1
        # print(flag_sum.shape)
        # print(flag_sum)

        # print(u_dist.shape)
        # u_dist_value, u_dist_indices = torch.min(u_dist, dim=1)
        # print(cls_dist.shape, u_dist_indices.shape)
        # t = cls_dist[u_dist_indices]
        # print(t.shape, u_dist_value.shape)
        # mask = u_dist_value > t

        coreset_embedding = u_embedding_features[mask]
        coreset_ind = u_data_ind[mask]
        # coreset_embedding = torch.cat(coreset_embedding, dim=0).cpu()
        # coreset_ind = torch.cat(coreset_ind, dim=0).cpu()
        query = kCenterGreedy(features=coreset_embedding)
        str = 'coreset len: %d' % len(coreset_embedding)
        print(str)
        self.trainer.logger.info(str)
        if len(coreset_embedding) < n_select:
            tmp_ind = np.random.permutation(u_data_ind[~mask])[:n_select-len(coreset_embedding)]
            str = 'random select %d' % len(tmp_ind)
            print(str)
            self.trainer.logger.info(str)
            return unlabel_ind[np.append(coreset_ind, tmp_ind)]
        else:
            select_ind = query.select_batch_(already_selected=[], N=n_select)
            return unlabel_ind[coreset_ind[select_ind]]

        # print(coreset_embedding.shape)
        # print(coreset_ind.shape)
        # print(len(select_ind))

        # _, unlabel_ind = self.build_unlabel_loader()
        # tmp_ind = np.random.permutation(range(len(unlabel_ind)))[:n_select]

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def predict_probs_and_embed(self, data_loader, eval=True):
        # probs = []
        logits = []
        embedding_features = []
        ds_inds = []
        true_labels = []


        if eval:
            self.net.eval()
        else:
            self.net.train()

        for x, y, ind, ds_ind in data_loader:
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                out, e1 = self.net(x)
            # prob = F.softmax(out, dim=1)
            # probs[idxs] = prob.cpu()
            # embeddings[idxs] = e1.cpu()

            logits.append(out)
            true_labels.append(y)

            embedding_features.append(e1)
            ds_inds.append(ds_ind)
            # probs.append(prob)

        logits = torch.cat(logits, dim=0).cpu()
        # probs = torch.cat(probs, dim=0).cpu()
        embedding_features = torch.cat(embedding_features, dim=0).cpu()
        true_labels = torch.cat(true_labels, dim=0).cpu()
        ds_inds = torch.cat(ds_inds, dim=0)

        return embedding_features, true_labels, ds_inds