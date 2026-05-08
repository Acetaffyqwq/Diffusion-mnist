import torch
import torch.nn.functional as F
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.L1=nn.Linear(dim,dim//2)
        self.L2=nn.Linear(dim//2,dim)
    def forward(self,X):
        return F.sigmoid(self.L2(F.relu(self.L1(X))))

class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, poolker):
        super().__init__()
        # 预激活风格

        self.norm1 = nn.GroupNorm(8, in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding=1)

        self.atten=MLP(out_dim)
        self.pool = nn.MaxPool2d(poolker)

    def forward(self, X):
        X = self.conv1(F.relu(self.norm1(X)))
        X = self.conv2(F.relu(self.norm2(X)))
        Y = torch.mean(X,[2,3])
        Y=self.atten(Y)
        X=X*Y.reshape(Y.shape[0],Y.shape[1],1,1)
        return self.pool(X), X


# 还原网格
# 上采样，和 encode 时的浅层合并，然后用卷积整合信息


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = DownBlock(in_dim * 2, out_dim, 1)

    def forward(self, X, Y):
        X = F.interpolate(X, scale_factor=2, mode="bilinear")
        X = torch.cat([X, Y], dim=1)
        return self.conv(X)[0]


class Unet(nn.Module):
    def __init__(self, T, in_dim=1, out_dim=1, cdim=64, D=[16, 32, 64]):
        super().__init__()
        self.embedding = nn.Embedding(T, cdim)
        self.num_embed=nn.Embedding(10,cdim)

        self.convi = nn.Conv2d(in_dim, D[0], 3, padding=1)
        self.enlayers = nn.ModuleList()
        # [n,16,32,32]
        for i in range(1, len(D)):
            self.enlayers.append(DownBlock(D[i - 1], D[i], 2))
        # [n,64,8,8]

        self.Lmid1 = nn.Linear(cdim, D[-1] * 8 * 8, 1)
        self.Lmid2 = nn.Linear(cdim, D[-1] * 8 * 8, 1)
        self.conmid = DownBlock(D[-1], D[-1], 1)

        self.delayers = nn.ModuleList()
        for i in range(len(D) - 1, 0, -1):
            self.delayers.append(UpBlock(D[i], D[i - 1]))
        # [n,16,32,32]
        self.convf = nn.Conv2d(D[0], out_dim, 3, padding=1)

    def forward(self, X, t,val):
        X = self.convi(X)
        feats = []
        dep = len(self.enlayers)
        for i in range(dep):
            X, z = self.enlayers[i](X)
            feats.append(z)

        t = self.Lmid1(self.embedding(t))+self.Lmid2(self.num_embed(val))
        X = X + t.reshape(X.shape)
        X, _ = self.conmid(X)

        for i in range(dep):
            X = self.delayers[i](X, feats[dep - 1 - i])

        return self.convf(X)


class Diffusion(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.register_buffer("beta", torch.zeros([T]))
        self.register_buffer("alpha", torch.zeros([T]))
        self.register_buffer("Salpha", torch.ones([T]))
        self.register_buffer("sigma", torch.zeros([T]))
        for i in range(1, T):
            self.beta[i] = i / T / 50 # 0~0.02 线性增加
            self.alpha[i] = 1 - self.beta[i]
            self.Salpha[i] = self.Salpha[i - 1] * self.alpha[i]
            self.sigma[i] = (
                self.beta[i] * (1 - self.Salpha[i - 1]) / (1 - self.Salpha[i])
            )

        self.unet = Unet(T)
        self.loss_fn = torch.nn.MSELoss()

    def step(self, XT, t,val):
        eps = self.unet(XT, t,val)
        # 将 [batch] 的数值 reshape 为 [batch,1,1,1]，以便与 [batch,1,32,32] 正确广播
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        Salpha_t = self.Salpha[t].view(-1, 1, 1, 1)
        sigma_t = self.sigma[t].view(-1, 1, 1, 1)

        mu = (1 / torch.sqrt(alpha_t)) * (XT - ((1 - alpha_t) / torch.sqrt(1 - Salpha_t)) * eps)

        e = torch.randn(mu.shape, device=mu.device)
        return mu + torch.sqrt(sigma_t) * e

    def forward(self, X0, t,val):
        # X0: [batch,1,32,32]
        # t : [batch]
        # Eps: [batch,1,32,32]
        Eps = torch.randn(X0.shape, device=X0.device)
        # 将 [batch] 的数值 reshape 为 [batch,1,1,1]
        Salpha_t = self.Salpha[t].view(-1, 1, 1, 1)
        XT = torch.sqrt(Salpha_t) * X0 + torch.sqrt(1 - Salpha_t) * Eps

        pred = self.unet(XT, t,val)
        return self.loss_fn(pred, Eps)
