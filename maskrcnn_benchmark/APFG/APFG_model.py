import torch
import numpy as np
from torch import nn
from maskrcnn_benchmark.modeling.utils import cat

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Semantic_Encoder(nn.Module):
    def __init__(self, cfg):
        super(Semantic_Encoder, self).__init__()
        latent_size = cfg.MODEL.APFG_latent_size
        layer_sizes_0 = cfg.MODEL.semantic_dim
        self.fc1=nn.Linear(layer_sizes_0, layer_sizes_0)
        self.fc2 = nn.Linear(layer_sizes_0, int(layer_sizes_0/2))
        self.fc3 = nn.Linear(int(layer_sizes_0/2), latent_size*2)
        self.lrelu = nn.LeakyReLU(0.1, True)
        self.linear_means = nn.Linear(latent_size*2, latent_size)
        self.linear_log_var = nn.Linear(latent_size*2, latent_size)
        self.apply(weights_init)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Semantic_Decoder(nn.Module):
    def __init__(self, cfg):
        super(Semantic_Decoder,self).__init__()
        latent_size = cfg.MODEL.APFG_latent_size
        input_size = latent_size
        self.fc0 = nn.Linear(input_size, latent_size*2)
        self.fc1 = nn.Linear(latent_size*2, int(cfg.MODEL.semantic_dim/2))
        self.fc2 = nn.Linear(int(cfg.MODEL.semantic_dim/2), cfg.MODEL.semantic_dim)
        self.fc3 = nn.Linear(cfg.MODEL.semantic_dim, cfg.MODEL.semantic_dim)
        self.lrelu = nn.LeakyReLU(0.1, True)
        self.apply(weights_init)

    def forward(self, z):
        x = self.lrelu(self.fc0(z))
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.fc3(x)
        return x


class Visual_Encoder(nn.Module):
    def __init__(self, cfg):
        super(Visual_Encoder, self).__init__()
        feat_dim = cfg.MODEL.ROI_RELATION_HEAD.LAST_FEATS_DIM
        latent_size = cfg.MODEL.APFG_latent_size
        layer_sizes_0 = feat_dim

        self.fc1=nn.Linear(layer_sizes_0, layer_sizes_0)
        self.fc2 = nn.Linear(layer_sizes_0, int(layer_sizes_0/2))
        self.fc3 = nn.Linear(int(layer_sizes_0/2), latent_size*2)
        self.lrelu = nn.LeakyReLU(0.1, True)
        self.linear_means = nn.Linear(latent_size*2, latent_size)
        self.linear_log_var = nn.Linear(latent_size*2, latent_size)
        self.apply(weights_init)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Visual_Decoder(nn.Module):
    def __init__(self, cfg):
        super(Visual_Decoder,self).__init__()
        feat_dim = cfg.MODEL.ROI_RELATION_HEAD.LAST_FEATS_DIM
        latent_size = cfg.MODEL.APFG_latent_size
        input_size = latent_size + cfg.MODEL.semantic_dim
        self.fc0 = nn.Linear(input_size, latent_size*2)
        self.fc1 = nn.Linear(latent_size*2, int(feat_dim/2))
        self.fc2 = nn.Linear(int(feat_dim/2), feat_dim)
        self.fc3 = nn.Linear(feat_dim, feat_dim)
        self.lrelu = nn.LeakyReLU(0.1, True)
        self.apply(weights_init)

    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)
        x = self.lrelu(self.fc0(z))
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.fc3(x)

        return x




class APFG_MODEL(nn.Module):
    def __init__(self, cfg):
        super(APFG_MODEL, self).__init__()
        self.cfg = cfg
        self.semantic_encoder = Semantic_Encoder(cfg)
        self.semantic_decoder = Semantic_Decoder(cfg)
        self.visual_encoder = Visual_Encoder(cfg)
        self.visual_decoder = Visual_Decoder(cfg)
        self.num_classes = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.latent_dim = cfg.MODEL.APFG_latent_size
        self.embedding = nn.Linear(51 + 151 +151 + 4096, cfg.MODEL.semantic_dim)
        self.lrelu = nn.LeakyReLU(0.1, True)
        self.apply(weights_init)

    def forward(self, feats, rel_one_hot, fg_union_feats):

        semantic_embedding = self.lrelu(self.embedding(torch.cat([rel_one_hot,fg_union_feats],dim=1)))
        mu_semantic, log_var_semantic = self.semantic_encoder(semantic_embedding)
        mu_visual, log_var_visual = self.visual_encoder(feats)
        z_semantic = self.reparameterize(mu_semantic, log_var_semantic)
        z_visual = self.reparameterize(mu_visual, log_var_visual)
        De1_z_semantic = self.semantic_decoder(z_semantic)
        De1_z_visual = self.semantic_decoder(z_visual)
        De2_z_semantic = self.visual_decoder(z_semantic, semantic_embedding)
        De2_z_visual = self.visual_decoder(z_visual, semantic_embedding)

        return [feats, semantic_embedding, De1_z_semantic, De1_z_visual, De2_z_semantic, De2_z_visual,
                mu_semantic, log_var_semantic, mu_visual, log_var_visual]


    def feats_sample(self, cfg, num_samples, label_sample):
        z = torch.randn(num_samples, self.latent_dim)
        device = torch.device(cfg.MODEL.DEVICE)
        feats_path = self.cfg.OUTPUT_DIR_feats + "/feats_info.pth"
        feats_info = torch.load(feats_path)
        rels_labels = feats_info['rel_labels']
        obj_pair = feats_info['obj_pair_preds']
        fg_index = np.where(rels_labels != 0)[0]
        rels_labels = rels_labels[fg_index]
        obj_pair = obj_pair[:, fg_index]
        index = np.where(rels_labels == label_sample)[0]
        sampling_index = index[np.random.randint(0, index.shape[0], num_samples)]

        o1_one_hot = torch.zeros(num_samples, 151).to(device)
        o2_one_hot = torch.zeros(num_samples, 151).to(device)
        rels_one_hot0 = torch.zeros(num_samples, 51).to(device)
        o1_one_hot[list(range(num_samples)), obj_pair[0,sampling_index].long()] = 1
        o2_one_hot[list(range(num_samples)), obj_pair[1,sampling_index].long()] = 1
        rels_one_hot0[list(range(num_samples)), rels_labels[sampling_index].long()] = 1
        rel_one_hot_stack = torch.cat([rels_one_hot0, o1_one_hot, o2_one_hot], dim=1)

        z = z.to(device)
        rel_one_hot_stack = rel_one_hot_stack.to(device)
        fg_union_feats = []
        fg_union_features_path = cfg.OUTPUT_DIR_feats + '/fg_union_feats/'
        for i in sampling_index:
            fg_union_feats.append(torch.load(fg_union_features_path+ str(i)))
        fg_union_feats = cat([fg_union_feats[i].reshape([1,-1]) for i in range(len(fg_union_feats))],dim=0)
        fg_union_feats = fg_union_feats.to(device)
        semantic_embedding = self.lrelu(self.embedding(torch.cat([rel_one_hot_stack,fg_union_feats],dim=1)))
        samples = self.visual_decoder(z, semantic_embedding)
        return samples

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


def loss_function(*args,
                  **kwargs) -> dict:
    feats, semantic_embedding, De1_z_semantic, De1_z_visual, De2_z_semantic, De2_z_visual, \
    mu_semantic, log_var_semantic, mu_visual, log_var_visual = args[0]
    w_kl = args[1]
    w_da = args[2]
    w_cr = args[3]
    L_Re = torch.mean(torch.linalg.norm((semantic_embedding - De1_z_semantic), dim=1) + \
                      torch.linalg.norm((feats - De2_z_visual), dim=1))
    L_CA = torch.mean(torch.linalg.norm((semantic_embedding - De1_z_visual), dim=1) + \
                      torch.linalg.norm((feats - De2_z_semantic), dim=1))
    L_DA = torch.mean((torch.linalg.norm((mu_semantic - mu_visual), dim=1)**2 +
            torch.linalg.norm((torch.exp(0.5 * log_var_semantic)-torch.exp(0.5 * log_var_visual)), dim=1)**2)**0.5)
    L_kl = torch.mean(-0.5 * torch.sum(1 + log_var_semantic - mu_semantic ** 2 - log_var_semantic.exp(), dim = 1), dim = 0) + \
           torch.mean(-0.5 * torch.sum(1 + log_var_visual - mu_visual ** 2 - log_var_visual.exp(), dim = 1), dim = 0)
    L_vae = L_Re + w_kl*L_kl
    loss = L_vae + w_cr*L_CA + w_da*L_DA
    return {'loss': loss, 'Recon_Loss': L_Re, 'CA_Loss': w_cr*L_CA, 'DA_Loss': w_da*L_DA, 'KLD': w_kl*L_kl }
