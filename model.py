from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    message = torch.einsum('bhnm,bdhm->bdhn', prob, value)
    return message, prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        self.norm = nn.InstanceNorm1d(d_model)
        # self.norm2 = nn.InstanceNorm1d(d_model)

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        x = self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))
        x = self.norm(x)
        # x = self.norm2(x)

        return x


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class Transformer(nn.Module):
    def __init__(self, num_head: int, feature_dim: int):
        super().__init__()

        self.attention_layer = AttentionalPropagation(feature_dim, num_head)

    def forward(self, desc0, desc1):
        desc0_ca = self.attention_layer(desc0, desc1)
        desc1_ca = self.attention_layer(desc1, desc0)

        return desc0_ca, desc1_ca

class Conv1dBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(Conv1dBNReLU, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_planes),
            # nn.InstanceNorm1d(out_planes),
            nn.ReLU(inplace=True))


class FCBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(FCBNReLU, self).__init__(
            nn.Linear(in_planes, out_planes, bias=False),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True))


class TNet(nn.Module):
    def __init__(self):
        super(TNet, self).__init__()
        self.encoder = nn.Sequential(
            Conv1dBNReLU(3, 64),
            Conv1dBNReLU(64, 128),
            Conv1dBNReLU(128, 256))
        self.decoder = nn.Sequential(
            FCBNReLU(256, 128),
            FCBNReLU(128, 64),
            nn.Linear(64, 6))

    @staticmethod
    def f2R(f):
        r1 = F.normalize(f[:, :3])
        proj = (r1.unsqueeze(1) @ f[:, 3:].unsqueeze(2)).squeeze(2)
        r2 = F.normalize(f[:, 3:] - proj * r1)
        r3 = r1.cross(r2)
        return torch.stack([r1, r2, r3], dim=2)

    def forward(self, pts):
        f = self.encoder(pts)
        f, _ = f.max(dim=2)
        f = self.decoder(f)
        R = self.f2R(f)

        return R @ pts


def jrmpc_params(gamma1, gamma2, pts1, pts2):
    '''
        Inputs:
            gamma: B x N x J
            pts: B x N x 3
        '''
    ###
    # if torch.isnan(gamma1)
    eps = 1e-10
    d = gamma1.sum(dim=1, keepdim=True) + gamma2.sum(dim=1, keepdim=True) + eps
    # compute xk, BxKx3
    m1 = (gamma1.transpose(2, 1) @ pts1)
    m2 = (gamma2.transpose(2, 1) @ pts2)

    # Bx3xJ
    xk = (m1 + m2).transpose(2, 1) / d

    # diff: B x N x J x 3

    diff1 = pts1.unsqueeze(2) - xk.transpose(2, 1).unsqueeze(1)
    diff2 = pts2.unsqueeze(2) - xk.transpose(2, 1).unsqueeze(1)

    # BxNxK
    diff_x = (diff1.unsqueeze(3) @ diff1.unsqueeze(4)).squeeze() * gamma1
    diff_y = (diff2.unsqueeze(3) @ diff2.unsqueeze(4)).squeeze() * gamma2

    # Bx1xK
    sigma = ((diff_x + diff_y).sum(dim=1, keepdim=True) / (3 * d)) + eps

    W1 = m1.transpose(2, 1) / sigma
    W2 = m2.transpose(2, 1) / sigma

    return xk, sigma, W1, W2


def JRMPC(gamma, mu, sigma, W):
    #  BxJ
    lbd = gamma.sum(dim=1, keepdim=True)
    # Bx1xJ
    lambda_jk = lbd * (1 / sigma)

    # mean of w，B x 3
    mW = W.sum(dim=2, keepdim=True)

    # mean of X, X * b(lambda_jk)， B * 3
    mX = mu @ lambda_jk.transpose(2, 1)

    # sum of weights, B x 1

    sow = (lbd / sigma).sum(dim=2)
    # matlab P
    Ms = mu @ W.transpose(1, 2) - mX @ mW.transpose(2, 1) / sow.unsqueeze(2)

    U, _, V = torch.svd(Ms.cpu())
    U = U.to(gamma.device)
    V = V.to(gamma.device)

    S = torch.eye(3).unsqueeze(0).repeat(U.shape[0], 1, 1).to(U.device)
    S[:, 2, 2] = torch.det(U @ V)

    R_1w = U @ S @ V.transpose(2, 1)
    t_1w = (mX - R_1w @ mW).transpose(2, 1) / sow.unsqueeze(2)

    return R_1w, t_1w


class UGMMReg(nn.Module):
    def __init__(self, args):
        super(UGMMReg, self).__init__()
        self.tnet = TNet()

        self.mlp1 = Conv1dBNReLU(3, 64)
        self.mlp2 = Conv1dBNReLU(64, 64)
        self.Transformer1 = Transformer(1, 64)

        self.mlp3 = Conv1dBNReLU(128, 128)
        self.mlp4 = Conv1dBNReLU(128, 256)
        self.Transformer2 = Transformer(4, 256)

        self.mlp5 = Conv1dBNReLU(512, 512)

        self.Clustering = nn.Sequential(
            Conv1dBNReLU(1024, 1024),
            Conv1dBNReLU(1024, 512),
            Conv1dBNReLU(512, 128),
            Conv1dBNReLU(128, 64),
            nn.Conv1d(64, args.n_components, kernel_size=1)
        )

    def forward(self, src_pts, tgt_pts):
        src_f0 = (src_pts - src_pts.mean(dim=1, keepdim=True)).transpose(1, 2)
        src_f0 = self.tnet(src_f0)

        tgt_f0 = (tgt_pts - tgt_pts.mean(dim=1, keepdim=True)).transpose(1, 2)
        tgt_f0 = self.tnet(tgt_f0)

        N = src_pts.shape[1]
        # local feature extraction
        src_f1 = self.mlp1(src_f0)  # 64
        tgt_f1 = self.mlp1(tgt_f0)

        src_f2 = self.mlp2(src_f1)  # 64
        tgt_f2 = self.mlp2(tgt_f1)

        src_f2_ca, tgt_f2_ca = self.Transformer1(src_f2, tgt_f2)

        src_f_lg = torch.cat([src_f2, src_f2_ca], dim=1)  # 128
        tgt_f_lg = torch.cat([tgt_f2, tgt_f2_ca], dim=1)

        src_f3 = self.mlp3(src_f_lg)  # 128
        tgt_f3 = self.mlp3(tgt_f_lg)

        src_f4 = self.mlp4(src_f3)  # 256
        tgt_f4 = self.mlp4(tgt_f3)

        # Global info interaction
        src_f4_ca, tgt_f4_ca = self.Transformer2(src_f4, tgt_f4)

        src_f_lg = torch.cat([src_f4, src_f4_ca], dim=1)  # 512
        tgt_f_lg = torch.cat([tgt_f4, tgt_f4_ca], dim=1)

        src_f5 = self.mlp5(src_f_lg)
        tgt_f5 = self.mlp5(tgt_f_lg)

        src_f_final = src_f5
        tgt_f_final = tgt_f5

        src_final_g = src_f_final.max(dim=2, keepdim=True)[0].repeat(1, 1, N)
        tgt_final_g = tgt_f_final.max(dim=2, keepdim=True)[0].repeat(1, 1, N)

        gamma0 = F.softmax(self.Clustering(torch.cat([src_f_final, src_final_g], dim=1)).transpose(2, 1),
                           dim=2)  # 2048
        gamma1 = F.softmax(self.Clustering(torch.cat([tgt_f_final, tgt_final_g], dim=1)).transpose(2, 1), dim=2)

        xk, sigma, W0, W1 = jrmpc_params(gamma0, gamma1, src_pts, tgt_pts)

        R_0w, t_0w = JRMPC(gamma0, xk, sigma, W0)
        R_1w, t_1w = JRMPC(gamma1, xk, sigma, W1)

        return R_0w, t_0w, R_1w, t_1w

if __name__ == '__main__':
    from types import SimpleNamespace
    ckpt_path = './checkpoint.pth'
    args = SimpleNamespace(n_components=16)
    model = UGMMReg(args)
    model.load_state_dict(torch.load(ckpt_path)['model'], strict=True)

    print(model)