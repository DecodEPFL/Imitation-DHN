import torch

class TwoRobots(torch.nn.Module):
    def __init__(self, xbar, linear=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.xbar = xbar
        self.n_agents = 2
        self.n = 8
        self.m = 4
        self.h = 0.05

        self.Mx = torch.zeros(self.n_agents, self.n, device=device)
        self.Mx[0, 0] = 1
        self.Mx[1, 4] = 1
        self.My = torch.zeros(self.n_agents, self.n, device=device)
        self.My[0, 1] = 1
        self.My[1, 5] = 1

        self.Mvx = torch.zeros(self.n_agents, self.n, device=device)
        self.Mvx[0, 2] = 1
        self.Mvx[1, 6] = 1
        self.Mvy = torch.zeros(self.n_agents, self.n, device=device)
        self.Mvy[0, 3] = 1
        self.Mvy[1, 7] = 1

        self.mv1 = torch.zeros(2, self.n, device=device)
        self.mv1[0, 2] = 1
        self.mv1[1, 3] = 1
        self.mv2 = torch.zeros(2, self.n, device=device)
        self.mv2[0, 6] = 1
        self.mv2[1, 7] = 1
        self.mp = torch.zeros(2, self.n, device=device)
        self.Mp = torch.cat((self.mv1, self.mp, self.mv2, self.mp), 0)

    def f(self, t, x, u, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        m1, m2 = 1, 1
        kspringGround = 2
        cdampGround = 2


        k1, k2 = kspringGround, kspringGround
        c1, c2 = cdampGround, cdampGround

        px = torch.matmul(self.Mx, x)
        py = torch.matmul(self.My, x)
        vx = torch.matmul(self.Mvx, x)
        vy = torch.matmul(self.Mvy, x)



        B = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1 / m1, 0, 0, 0],
            [0, 1 / m1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1 / m2, 0],
            [0, 0, 0, 1 / m2],
        ])

        xt = torch.matmul(self.Mx.float(), self.xbar.float())
        yt = torch.matmul(self.My, self.xbar.float())

        deltaxt = px - xt
        deltayt = py - yt

        projxt = torch.cos(torch.atan2(deltayt, deltaxt))
        projyt = torch.sin(torch.atan2(deltayt, deltaxt))
        projvxt = torch.cos(torch.atan2(vy, vx))
        projvyt = torch.sin(torch.atan2(vy, vx))

        Fc01 = c1 * torch.sqrt(vx[0] ** 2 + vy[0] ** 2)
        Fc02 = c2 * torch.sqrt(vx[1] ** 2 + vy[1] ** 2)

        Fk01 = k1 * torch.sqrt(deltaxt[0] ** 2 + deltayt[0] ** 2)
        Fk02 = k2 * torch.sqrt(deltaxt[1] ** 2 + deltayt[1] ** 2)


        Fground1x = -Fk01 * projxt[0] - Fc01 * projvxt[0]
        Fground1y = -Fk01 * projyt[0] - Fc01 * projvyt[0]

        Fground2x = -Fk02 * projxt[1] - Fc02 * projvxt[1]
        Fground2y = -Fk02 * projyt[1] - Fc02 * projvyt[1]


        A1x = torch.tensor([
            0,
            0,
            (Fground1x) / m1,
            (Fground1y) / m1,
            0,
            0,
            (Fground2x) / m2,
            (Fground2y) / m2,
        ])

        A2x = torch.matmul(self.Mp, x)

        Ax = A1x + A2x

        x1 = x + (Ax + torch.matmul(B, u)) * self.h
        return x1

    def f_withnoise(self, t, x, u, w, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        #matrix used to apply the noise only to the position and scaled with time
        x1 = self.f(t, x, u, w)+w
        return x1

    def forward(self, t, x, u, w):
        #if t == 0:
        #    x1 = w
        #else:
        x1 = self.f_withnoise(t, x, u, w)
        return x1
