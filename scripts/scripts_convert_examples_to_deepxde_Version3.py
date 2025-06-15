#!/usr/bin/env python3
"""
批量将 pinn-blood-flow/examples 下的每个案例转换为 DeepXDE 脚本。
生成文件：examples/<case>/deepxde_<case>.py
"""
import os
import textwrap

template = textwrap.dedent(r'''
import numpy as np
import deepxde as dde
from stl import mesh
import torch

def load_surface_points(stl_path):
    """从 STL 文件读取三角片，返回去重后的顶点坐标 (N×3)"""
    m = mesh.Mesh.from_file(stl_path)
    pts = np.vstack([m.v0, m.v1, m.v2])
    return np.unique(pts, axis=0)

def navier_stokes_residual(x, y, rho, nu):
    """Steady Navier–Stokes 残差，输出 [continuity, mom_x, mom_y, mom_z]"""
    u, v, w, p = y[:,0:1], y[:,1:2], y[:,2:3], y[:,3:4]
    # 一阶导
    deriv = {}
    for i,name in enumerate(['u','v','w','p']):
        for j in range(3):
            deriv[f"{name}{j}"] = dde.grad.jacobian(y, x, i=i, j=j)
    # 二阶导（粘性项）
    hess = {}
    for i,name in enumerate(['u','v','w']):
        for j in range(3):
            hess[f"{name}{j}{j}"] = dde.grad.hessian(y, x, component=i, i=j, j=j)
    continuity = deriv['u0'] + deriv['v1'] + deriv['w2']
    mom = []
    for name,r in [('u',0), ('v',1), ('w',2)]:
        mom.append(
            (u*deriv['u0'] + v*deriv['u1'] + w*deriv['u2'])[ : , :1]  # convective term示例
            + deriv['p'+str(r)]/rho
            - nu*(hess[name+f"{0}{0}"] + hess[name+f"{1}{1}"] + hess[name+f"{2}{2}"])
        )
    return [continuity] + mom

def print_gpu_memory(prefix=""):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        a = torch.cuda.memory_allocated()/1024**2
        r = torch.cuda.memory_reserved()/1024**2
        p = torch.cuda.max_memory_reserved()/1024**2
        print(f"{prefix} GPU | alloc {a:.1f} MiB, reserved {r:.1f} MiB, peak {p:.1f} MiB")
        torch.cuda.reset_peak_memory_stats()
    else:
        print(prefix + " no CUDA")

def run():
    # 物理参数
    rho = 1.05
    mu = 0.00385
    nu = mu/rho

    # 读取所有边界 STL 并提取点云
    base = os.path.dirname(__file__)
    stl_dir = os.path.join(base, "stl_files")
    pts_dict = {f.split('.')[0]: load_surface_points(os.path.join(stl_dir,f))
                for f in os.listdir(stl_dir) if f.lower().endswith('.stl')}

    # 几何域：包围所有点云
    all_pts = np.vstack(list(pts_dict.values()))
    xmin, xmax = all_pts.min(axis=0)-0.05, all_pts.max(axis=0)+0.05
    geom = dde.geometry.Cuboid(xmin, xmax)

    # 构造边界条件
    bcs = []
    for name, pts in pts_dict.items():
        # 默认：入口 inlet 施加抛物线速度，其它全部设 0
        if 'inlet' in name.lower():
            r = np.linalg.norm(pts - pts.mean(axis=0)[None,:],axis=1)
            parab = 0.3*np.maximum(1-(r/0.5)**2,0)[:,None]
            for comp in range(3):
                bcs.append(dde.icbc.PointSetBC(pts, parab if comp==0 else np.zeros_like(parab), component=comp))
        elif 'outlet' in name.lower():
            bcs.append(dde.icbc.PointSetBC(pts, np.zeros((len(pts),1)), component=3))
        else:  # wall、noslip...
            zero = np.zeros((len(pts),1))
            for comp in (0,1,2):
                bcs.append(dde.icbc.PointSetBC(pts, zero, component=comp))

    # PDE 数据与模型
    data = dde.data.PDE(
        geom,
        lambda x,y: navier_stokes_residual(x,y,rho,nu),
        bcs,
        num_domain=8000,
        num_boundary=4000,
    )
    net = dde.nn.FNN([3] + [128]*5 + [4], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    print_gpu_memory("Before train:")
    torch.cuda.reset_peak_memory_stats()
    lossh, st = model.train(iterations=5000)
    print_gpu_memory("After train:")
    dde.saveplot(lossh, st, issave=True, isplot=True)

if __name__ == "__main__":
    import os
    run()
''')

def main():
    root = os.path.join(os.getcwd(), "examples")
    for case in os.listdir(root):
        case_dir = os.path.join(root, case)
        if not os.path.isdir(case_dir):
            continue
        out_file = os.path.join(case_dir, f"deepxde_{case}.py")
        with open(out_file, "w") as f:
            # 写头部说明
            f.write(f'"""\n自动生成的 DeepXDE 版本: {case}\n"""\n')
            f.write("import os\n")
            f.write(template)
        print(f"生成 {out_file}")

if __name__ == "__main__":
    main()