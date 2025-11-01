from net import Net
import torch
import os
import matplotlib.pyplot as plt
from Parser_PINN import get_parser
import random
import numpy as np
from torch.autograd import grad

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

def PDE(u, x_f, y_f):
    return d(d(u, x_f), x_f) + d(d(u, y_f), y_f) + 1


def is_neumann_boundary_x(u, x, y):
    return d(u, x)

def is_neumann_boundary_y(u, x, y):
    return d(u, y)

def train(args):
    setup_seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    PINN = Net(seq_net=args.seq_net, activation=args.activation)
    optimizer = args.optimizer(PINN.parameters(), args.lr)

    loss_history = []
    for epoch in range(1000):
        # for epoch in range(args.epochs):
        optimizer.zero_grad()
        # inside
        x_f = ((args.x_left + args.x_right) / 2 + (args.x_right - args.x_left) *
               (torch.rand(size=(args.n_f, 1), dtype=torch.float) - 0.5)
               ).requires_grad_(True)

        y_f = ((args.y_left + args.y_right) / 2 + (args.y_right - args.y_left) *
               (torch.rand(size=(args.n_f, 1), dtype=torch.float, device=device) - 0.5)
               ).requires_grad_(True)

        u_f = PINN(torch.cat([x_f, y_f], dim=1))
        PDE_ = PDE(u_f, x_f, y_f)
        mse_PDE = args.criterion(PDE_, torch.zeros_like(PDE_))

        # 局部采点
        # boundary

        x_rand_1 = ((args.x_left + args.x_right) / 2 + (args.x_right - args.x_left) *
                    (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
                    ).requires_grad_(True)
        x_rand_2 = ((args.x_left + args.x_right) / 2 + (args.x_right - args.x_left) *
                    (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
                    ).requires_grad_(True)

        y_rand_1 = ((args.y_left + args.y_right) / 2 + (args.y_right - args.y_left) *
                    (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
                    ).requires_grad_(True)
        y_rand_2 = ((args.y_left + args.y_right) / 2 + (args.y_right - args.y_left) *
                    (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
                    ).requires_grad_(True)
        xbc_l = (args.x_left * torch.ones_like(x_rand_1)
                 ).requires_grad_(True)

        xbc_r = (args.x_right * torch.ones_like(x_rand_1)
                 ).requires_grad_(True)

        ybc_l = (args.y_left * torch.ones_like(x_rand_1)
                 ).requires_grad_(True)
        ybc_r = (args.y_right * torch.ones_like(x_rand_1)
                 ).requires_grad_(True)

        # is_neumann_boundary  下
        u_b_1 = (PINN(torch.cat([x_rand_1, ybc_l], dim=1)))
        BC_1 = is_neumann_boundary_y(u_b_1, x_rand_1, ybc_l)
        mse_BC_1 = args.criterion(BC_1, torch.zeros_like(BC_1))

        # is_neumann_boundary上
        u_b_2 = PINN(torch.cat([x_rand_2, ybc_r], dim=1))
        BC_2 = is_neumann_boundary_y(u_b_2, x_rand_2, ybc_r)
        mse_BC_2 = args.criterion(BC_2, torch.zeros_like(BC_2))

        # is_dirichlet_boundary左
        u_b_3 = PINN(torch.cat([xbc_l, y_rand_1], dim=1))
        mse_BC_3 = args.criterion(u_b_3, torch.zeros_like(u_b_3))

        # is_dirichlet_boundary右
        u_b_4 = PINN(torch.cat([xbc_r, y_rand_2], dim=1))
        mse_BC_4 = args.criterion(u_b_4, torch.zeros_like(u_b_4))

        mse_BC = mse_BC_1 + mse_BC_2 + mse_BC_3 + mse_BC_4

        # loss
        loss = args.PDE_panelty * mse_PDE + args.BC_panelty * mse_BC
        loss_history.append([mse_PDE.item(), mse_BC.item(), loss.item()])
        if epoch % 100 == 0:
            print(
                'epoch:{:05d}, PDE: {:.08e}, BC: {:.08e}, loss: {:.08e}'.format(
                    epoch, mse_PDE.item(), mse_BC.item(), loss.item()
                )
            )
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10000 == 0:
            plt.cla()
            xx = torch.linspace(-1, 1, 400).cpu()
            yy = torch.linspace(-1, 1, 400).cpu()
            x1, y1 = torch.meshgrid([xx, yy])
            s1 = x1.shape
            x1 = x1.reshape((-1, 1))
            y1 = y1.reshape((-1, 1))
            x = torch.cat([x1, y1], dim=1)
            z = PINN(x)
            z_out = z.reshape(s1)
            out = z_out.cpu().T.detach().numpy()[::-1, :]
            plt.imshow(out, cmap='jet')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    xx = torch.linspace(-1, 1, 400).cpu()
    yy = torch.linspace(-1, 1, 400).cpu()
    x1, y1 = torch.meshgrid([xx, yy])
    s1 = x1.shape
    x1 = x1.reshape((-1, 1))
    y1 = y1.reshape((-1, 1))
    x = torch.cat([x1, y1], dim=1)
    z = PINN(x)
    z_out = z.reshape(s1)
    out = z_out.cpu().T.detach().numpy()[::-1, :]
    im1 = ax[0].imshow(out, cmap='jet')
    plt.colorbar(im1, ax=ax[0])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('T')
    ax[1].plot(loss_history)
    ax[1].set_yscale('log')
    ax[1].legend(('PDE loss', 'BC loss', 'Total loss'))

    # plt.savefig('./result/loss.png')
    plt.show()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    train(args)
