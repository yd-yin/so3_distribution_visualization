import numpy as np
import torch
from distribution import log_pdf
import matplotlib.cm as cm
import open3d as o3d
from pytorch3d import transforms as trans
from mesh2image import render_mesh

device = "cuda"

Vector3dVector = o3d.utility.Vector3dVector
TriangleMesh = o3d.geometry.TriangleMesh


def prob(fn_type, A, R, grids):
    return torch.exp(log_pdf(fn_type, A, R, grids))


def random_rot_fixed_column(u, col=0):  # u: [*,3]
    v = torch.zeros_like(u, device=device)
    v[..., 0] = u[..., 1]
    v[..., 1] = -u[..., 0]

    index = v.norm(dim=-1) < (1e-6)
    v[index] = torch.tensor([1.0, 0.0, 0.0], device=device)
    v = v / v.norm(dim=-1).unsqueeze(-1)

    if col == 0:
        return trans.rotation_6d_to_matrix(torch.cat([u, v], dim=-1)).transpose(-1, -2)
    elif col == 1:
        return trans.rotation_6d_to_matrix(torch.cat([v, u], dim=-1)).transpose(-1, -2)
    else:
        w = trans.rotation_6d_to_matrix(torch.cat([u, v], dim=-1))[..., 2, :]
        return trans.rotation_6d_to_matrix(torch.cat([v, w], dim=-1)).transpose(-1, -2)


def marginal_prob(fn_type, A, vertices, grids, steps=500):
    thetas = torch.linspace(0, 2 * np.pi, steps, device=device)
    trans_x = random_rot_fixed_column(vertices, col=0)  # [n,3,3]
    trans_y = random_rot_fixed_column(vertices, col=1)
    trans_z = random_rot_fixed_column(vertices, col=2)

    R_x = torch.eye(3, device=device).repeat(steps, 1, 1)  # [steps,3,3]
    R_x[:, 1, 1] = torch.cos(thetas)
    R_x[:, 1, 2] = torch.sin(thetas)
    R_x[:, 2, 1] = -torch.sin(thetas)
    R_x[:, 2, 2] = torch.cos(thetas)

    R_y = torch.eye(3, device=device).repeat(steps, 1, 1)  # [steps,3,3]
    R_y[:, 0, 0] = torch.cos(thetas)
    R_y[:, 0, 2] = -torch.sin(thetas)
    R_y[:, 2, 0] = torch.sin(thetas)
    R_y[:, 2, 2] = torch.cos(thetas)

    R_z = torch.eye(3, device=device).repeat(steps, 1, 1)  # [steps,3,3]
    R_z[:, 0, 0] = torch.cos(thetas)
    R_z[:, 0, 1] = torch.sin(thetas)
    R_z[:, 1, 1] = torch.cos(thetas)
    R_z[:, 1, 0] = -torch.sin(thetas)

    R_xs = torch.matmul(trans_x.unsqueeze(1), R_x.unsqueeze(0))  # [n,steps,3,3]
    R_ys = torch.matmul(trans_y.unsqueeze(1), R_y.unsqueeze(0))  # [n,steps,3,3]
    R_zs = torch.matmul(trans_z.unsqueeze(1), R_z.unsqueeze(0))  # [n,steps,3,3]

    prob_x = prob(fn_type, A, R_xs, grids)
    prob_y = prob(fn_type, A, R_ys, grids)
    prob_z = prob(fn_type, A, R_zs, grids)

    prob_x = torch.mean(prob_x, dim=1)
    prob_y = torch.mean(prob_y, dim=1)
    prob_z = torch.mean(prob_z, dim=1)

    return prob_x, prob_y, prob_z


def plot(A, fn_type, grids, scale_factor=30):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=2., resolution=300)
    vertices = np.array(mesh.vertices)
    center = np.mean(vertices, axis=0)
    vertices -= center
    vertices /= (np.linalg.norm(vertices, axis=1, keepdims=True) + 1e-8)
    mesh.vertices = Vector3dVector(vertices)

    A = A.reshape(1, 3, 3)

    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    probs_x, probs_y, probs_z = marginal_prob(fn_type, A, vertices, grids)

    norm_probs = (probs_x + probs_y + probs_z).detach().cpu().numpy()

    # scale_factor is a hyperparameter used to adapt jet color coding
    # Besides linear adaptation, one may try other methods
    norm_probs = norm_probs / scale_factor

    colors = cm.jet(norm_probs).reshape(-1, 4)[:, :3]

    mesh.vertex_colors = Vector3dVector(colors)
    return mesh



if __name__ == '__main__':
    A = 20 * torch.eye(3).to(device)
    grids = np.load('grids3.npy')
    grids = torch.tensor(grids, dtype=torch.float32, device=device)
    mesh = plot(A, fn_type='RFisher', grids=grids)
    # o3d.io.write_triangle_mesh(f'test.ply', mesh)
    img = render_mesh(mesh)
    o3d.io.write_image('test.png', img)
