import numpy as np
import torch

def DSNT_f(h, spacial=None):
    B, C, Z, Y, X = h.shape
    print("B, C, Z, Y, X", B, C, Z, Y, X)
    #heatmap = heatmap * 20
    #h = heatmap / torch.sum(heatmap, dim=(2, 3, 4), keepdim=True)
    x = torch.linspace(-1, 1, X)
    y = torch.linspace(-1, 1, Y)
    z = torch.linspace(-1, 1, Z)
    x_cord = x.view([B, 1, 1, 1, X])
    y_cord = y.view([B, 1, 1, Y, 1])
    z_cord = z.view([B, 1, Z, 1, 1])
    px = (h * x_cord).sum(dim=(2, 3)).sum(dim=-1)
    py = (h * y_cord).sum(dim=(2, 4)).sum(dim=-1)
    pz = (h * z_cord).sum(dim=(3, 4)).sum(dim=-1)

    #print(x_cord.shape, px.shape, px.view(B, C, 1, 1, 1).shape)
    var_x = (h * ((x_cord - px.view(B, C, 1, 1, 1)) ** 2)).sum(dim=(2, 3, 4))
    var_y = (h * (y_cord - py.view(B, C, 1, 1, 1)) ** 2).sum(dim=(2, 3, 4))
    var_z = (h * (z_cord - pz.view(B, C, 1, 1, 1)) ** 2).sum(dim=(2, 3, 4))
    return px, py, pz, var_x, var_y, var_z

if __name__ =='__main__':
    x = np.random.rand(1, 1, 200, 128, 128)
    h = torch.from_numpy(x)
    print(h.shape)
    DSNT_f(h)