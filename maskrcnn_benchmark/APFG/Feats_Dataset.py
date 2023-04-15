import torch
from maskrcnn_benchmark.modeling.utils import cat
import numpy as np

import os
import pickle
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.comm import get_rank, synchronize

class Feats_DataSet(torch.utils.data.Dataset):
    def __init__(self, cfg, logger=None):
        feats_info_path = cfg.OUTPUT_DIR_feats + '/feats_info.pth'
        self.bg_feats_path = cfg.OUTPUT_DIR_feats + '/bg_feats/'
        self.fg_union_features_path = cfg.OUTPUT_DIR_feats + '/fg_union_feats/'
        print('feats_info load')
        feats_info = torch.load(feats_info_path)
        self.fg_feats = torch.load(cfg.OUTPUT_DIR_feats + '/fg_feats.pth')
        print('end feats_info load')

        self.cfg = cfg
        self.Feats_resampling = cfg.MODEL.Feats_resampling
        self.rels_labels = feats_info['rel_labels']
        self.obj_pair = feats_info['obj_pair_preds']
        self.APFG_fg_count = 50 * self.cfg.MODEL.Rels_each_class

        if self.cfg.MODEL.APFG_train:
            fg_index = np.where(self.rels_labels != 0)[0]
            self.idx_list = fg_index.tolist()
            self.repeat_dict = {}
            self.Feats_resampling = False
        else:
            self.idx_list = list(range(len(self.rels_labels)))
            self.repeat_dict = None


        if self.Feats_resampling:
            if self.cfg.MODEL.APFG_Feats_generation:
                feats_dir = os.path.join(cfg.CLASSIFIER_OUTPUT_DIR, "APFG_feats.pkl")
                repeat_dict_dir = os.path.join(cfg.CLASSIFIER_OUTPUT_DIR, "APFG_repeat_dict.pkl")
                if os.path.exists(feats_dir) and os.path.exists(repeat_dict_dir):
                    print("load APFG_feats from " + feats_dir)
                    print("load APFG_repeat_dict from " + repeat_dict_dir)
                    with open(feats_dir, 'rb') as f:
                        APFG_feats = pickle.load(f)
                    with open(repeat_dict_dir, 'rb') as f:
                        repeat_dict = pickle.load(f)
                else:
                    if get_rank() == 0:
                        print("generation APFG_feats to " + feats_dir)
                        print("generation APFG_repeat_dict to " + repeat_dict_dir)
                        repeat_dict, APFG_feats = feats_repeat_dict_generation(self, logger)
                        with open(feats_dir, "wb") as f:
                            pickle.dump(APFG_feats, f)
                        with open(repeat_dict_dir, "wb") as f:
                            pickle.dump(repeat_dict, f)
                    synchronize()
                    with open(feats_dir, 'rb') as f:
                        APFG_feats = pickle.load(f)
                    with open(repeat_dict_dir, 'rb') as f:
                        repeat_dict = pickle.load(f)
                self.APFG_feats = APFG_feats['APFG_fg_feats']
                self.repeat_dict = repeat_dict
                self.APFG_fg_count = self.APFG_feats.shape[0]

            self.idx_list = self.repeat_dict[:, -1].tolist()

        self.fg_count = len(np.where(self.rels_labels!= 0)[0])
        self.bg_count = len(np.where(self.rels_labels== 0)[0]) - 1

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):

        if self.repeat_dict is not None:
            if index < self.APFG_fg_count and self.cfg.MODEL.APFG_Feats_generation:
                feats = self.APFG_feats[index, :]
                return feats, self.rels_labels[self.idx_list[index]]

            index = self.idx_list[index]

        rels = self.rels_labels[index]
        if rels != 0:
            feats = self.fg_feats[index, :]
        else:
            bg_id = index - self.fg_count
            if 100 * int(self.bg_count/100) <= bg_id:
                feats_grp = torch.load(self.bg_feats_path + str(100 * int(bg_id / 100)) + '_' + str(self.bg_count))
            else:
                feats_grp = torch.load(self.bg_feats_path + str(100*int(bg_id/100)) + '_' + str(100*(int(bg_id/100)+1)-1))
            feats = feats_grp[bg_id % 100, :]

        if self.cfg.MODEL.APFG_train:
            fg_union_feats = torch.load(self.fg_union_features_path + str(index))
            return feats, fg_union_feats, torch.cat([self.rels_labels[index].reshape([-1]), self.obj_pair[:, index]])

        return feats, rels



def feats_repeat_dict_generation(dataset, logger):
    bg = dataset.rels_labels.numpy() == 0
    fg = dataset.rels_labels.numpy() != 0

    num_class = 51
    rels_fg = dataset.rels_labels[fg].long().numpy()
    obj_pair_fg = dataset.obj_pair[:, fg].long().numpy()
    # [rel,o1,o2,feat_i]
    rel_list = [np.array([[0, 0, 0, 0]], dtype=int) for x in range(num_class)]
    fg_index = np.where(dataset.rels_labels.numpy() != 0)[0]

    F_c = np.zeros(num_class, dtype=int)
    bg_dict = np.zeros([bg.sum(), 4], dtype=int)
    bg_dict[:, 3] = np.array(np.where(dataset.rels_labels.numpy() == 0))

    rel_list_info_dir = os.path.join(cfg.CLASSIFIER_OUTPUT_DIR, "rel_list_info.pkl")
    if os.path.exists(rel_list_info_dir):
        print("load rel_list_info from " + rel_list_info_dir)
        with open(rel_list_info_dir, 'rb') as f:
            rel_list_info = pickle.load(f)
        rel_list = rel_list_info['rel_list']
        F_c = rel_list_info['F_c']
    else:
        print("generation rel_list_info to " + rel_list_info_dir)
        for i in range(len(rels_fg)):
            rel_list[rels_fg[i]] = np.concatenate((rel_list[rels_fg[i]], np.array(
                [[rels_fg[i], obj_pair_fg[0, i], obj_pair_fg[1, i], fg_index[i]]])), axis=0)
            F_c[rels_fg[i]] += 1
        rel_list_info = {}
        rel_list_info['rel_list'] = rel_list
        rel_list_info['F_c'] = F_c
        with open(rel_list_info_dir, "wb") as f:
            pickle.dump(rel_list_info, f)
        print("end generation rel_list_info to " + rel_list_info_dir)

    F_c[0] = bg.sum()
    if dataset.cfg.MODEL.Rels_each_class == 0:
        num_per_cls = max(F_c[1:])
    else:
        num_per_cls = dataset.cfg.MODEL.Rels_each_class

    fg_APFG_dict = {}
    fg_real_dict={}
    if dataset.cfg.MODEL.APFG_Feats_generation:
        from maskrcnn_benchmark.APFG.APFG_train import APFG_train
        fg_feats = {}
        APFG_fg_feats = {}
        local_rank = get_rank()
        APFG_model = APFG_train(cfg,local_rank, False , logger)

        for i_rel in range(1, num_class):
            each_relation_dict = rel_list[i_rel][1:, :]
            rels_cls = each_relation_dict.shape[0]

            if num_per_cls <= rels_cls:
                fg_real_dict[i_rel] = each_relation_dict[np.random.choice(np.arange(rels_cls), num_per_cls, replace=False)]
            else:
                fg_real_dict[i_rel] = each_relation_dict
                num_generation = num_per_cls - rels_cls
                fg_APFG_dict[i_rel] = each_relation_dict[np.random.randint(0, rels_cls, num_generation)]
                APFG_feats = APFG_model.feats_sample(dataset.cfg, num_generation, torch.tensor([i_rel]))
                fg_feats[i_rel] = APFG_feats.detach().clone().to(torch.float32).to("cpu")

        fg_APFG_dict = np.vstack([fg_APFG_dict[k] for k in fg_APFG_dict])
        fg_feats= cat([fg_feats[k] for k in fg_feats])
        APFG_fg_feats['APFG_fg_feats'] = fg_feats

        bg_dict = np.zeros([bg.sum(), 4], dtype=int)
        bg_dict[:, 3] = np.array(np.where(dataset.rels_labels.numpy() == 0))

        fg_real_dict  = np.vstack([fg_real_dict[k] for k in fg_real_dict])
        repeat_dict = np.vstack([fg_APFG_dict, fg_real_dict,
                                 bg_dict[np.random.choice(np.arange(bg_dict.shape[0]),
                                                          int(dataset.cfg.MODEL.Num_rels_bg),
                                                          replace=False), :]])
        return repeat_dict, APFG_fg_feats

