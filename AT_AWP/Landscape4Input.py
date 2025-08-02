import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib.colors import ListedColormap
import datetime
import matplotlib.ticker as ticker
import os
from baselines.attack import ClipPointsLinf
from hessian_eigenthings import compute_hessian_eigenthings
white = (1, 1, 1)

"""

"""

def rand_color() -> tuple:
    return (random.random(), random.random(), random.random())


def get_rand_cmap():
    return ListedColormap((white, rand_color()))


def get_datetime_str(style='dt'):
    cur_time = datetime.datetime.now()
    date_str = cur_time.strftime('%y_%m_%d_')
    time_str = cur_time.strftime('%H_%M_%S')
    if style == 'data':
        return date_str
    elif style == 'time':
        return time_str
    return date_str + time_str


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

modes = ['3D', 'Contour', 'HeatMap', '2D']
alpha = 0.5


class Landscape4Input():
    def __init__(self, args, model, point_index, label,
                 input: torch.tensor,
                 input1: torch.tensor,
                 mode='3D', y_axis_='flat'):
        '''

        :param model: taken input as input, output loss
        :param input:
        '''
        self.args = args
        self.model = model
        self.point_index = point_index
        self.label = label
        self.input = input
        self.input1 = input1
        def loss(input, labels):
            logits = input[0]
            loss1 = nn.CrossEntropyLoss()(logits, labels)
            return loss1
        self.cross_ent = loss
        assert mode in modes
        self.mode = mode
        self.y_axis_ = y_axis_
        self.clip_func = ClipPointsLinf(1.0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def synthesize_coordinates(self,
                               x_min=-16 / 255, x_max=16 / 255, x_interval=1 / 255,
                               y_min=-16, y_max=16, y_interval=33):
        x = np.linspace(x_min, x_max, x_interval)
        y = np.linspace(y_min, y_max, y_interval)
        self.mesh_x, self.mesh_y = np.meshgrid(x, y)
        return self.mesh_x, self.mesh_y

    def assign_coordinates(self, x, y):
        self.mesh_x = x
        self.mesh_y = y

    def assign_unit_vector(self, x_unit_vector, y_unit_vector=None):
        self.x_unit_vector = x_unit_vector
        self.y_unit_vector = y_unit_vector

    # @torch.no_grad()
    def draw(self, axes=None):
        if hasattr(self, 'x_unit_vector') and self.x_unit_vector is not None:
            pass
        else:
            self._find_direction()
        z, z1 = self._compute_for_draw()
        if axes is None and self.mode == '3D':
            axes = plt.axes(projection='3d')
        self._draw3D(self.mesh_x, self.mesh_y, z, z1, axes)

    def _find_direction(self):
        self.x_unit_vector = torch.randn(self.input.shape, device=self.device)
        self.y_unit_vector = torch.randn(self.input.shape, device=self.device)
        self.x_unit_vector /= torch.norm(self.x_unit_vector, p=float('inf'))
        self.y_unit_vector /= torch.norm(self.y_unit_vector, p=float('inf'))  # make sure the l 2 norm is 0
        # keep perpendicular
        # if torch.abs(self.x0.reshape(-1) @ self.y0.reshape(-1)) >= 0.1:
        #     self._find_direction()

    def _compute_for_draw(self):
        result = []
        result1 = []
        if self.mode == '2D':
            self.mesh_x = self.mesh_x[0, :]
            for i in tqdm(range(self.mesh_x.shape[0])):
                # with suppress_stdout_stderr():
                    now_x = self.mesh_x[i]
                    x = self.input + self.x_unit_vector * now_x
                    # x = self.clip_func(x, self.input)
                    x_1 = self.input1 + self.y_unit_vector * now_x
                    # x_1 = self.clip_func(x_1, self.input1)
                    if self.y_axis_=='flat':
                        x.requires_grad_()
                        x_1.requires_grad_()
                        eigenvals, _ = compute_hessian_eigenthings(self.model, [(x.transpose(1,2), self.label)], self.cross_ent, num_eigenthings=20) 
                        if now_x==0:
                            eigenval_init = eigenvals
                        flatness = max(eigenvals)
                        eigenvals1, _ = compute_hessian_eigenthings(self.model, [(x_1.transpose(1,2), self.label)], self.cross_ent, num_eigenthings=20) 
                        if now_x==0:
                            eigenval_init1 = eigenvals1
                        flatness1 = max(eigenvals1)
                        result.append(flatness)
                        result1.append(flatness1)
                    else:
                        # noise = self.project(x-self.input)
                        logits = self.model((x).transpose(1,2))
                        logits_1 = self.model((x_1).transpose(1,2))
                        loss = self.cross_ent(logits, self.label)
                        if now_x==0:
                            loss_init = loss
                        loss_1 = self.cross_ent(logits_1, self.label)
                        if now_x==0:
                            loss_init1 = loss_1
                        result.append(loss.detach().cpu().numpy())
                        result1.append(loss_1.detach().cpu().numpy())
            if self.y_axis_=='flat':
                for i in range(len(result)):
                    result[i] = result[i] - max(eigenval_init)
                    result1[i] = result1[i] - max(eigenval_init1)
                result = np.array(result)
                result1 = np.array(result1)
            else:
                for i in range(len(result)): #####我这里不是减去原始损失了吗？？？
                    result[i] = result[i] -loss_init.detach().cpu().numpy()
                    if result[i] <0:
                        result[i] = -result[i]
                    result1[i] = result1[i] -loss_init1.detach().cpu().numpy()
                    if result1[i] <0:
                        result1[i] = -result1[i]
                result = np.array(result)
                result1 = np.array(result1)
            result = result.reshape(self.mesh_x.shape)
            result1 = result1.reshape(self.mesh_x.shape)
            return result, result1
        else:
            for i in tqdm(range(self.mesh_x.shape[0])):
                # with suppress_stdout_stderr():
                    for j in range(self.mesh_x.shape[1]):
                        now_x = self.mesh_x[i, j]
                        now_y = self.mesh_y[i, j]
                        x = self.input + self.x_unit_vector * now_x + self.y_unit_vector * now_y
                        x = self.clip_func(x, self.input)
                        logits = self.model((x).transpose(1,2))[0]
                        loss = self.cross_ent(logits, self.label)
                        result.append(loss)
            result = np.array(result)
            result = result.reshape(self.mesh_x.shape)
            return result, None

    def _draw3D(self, mesh_x, mesh_y, mesh_z, mesh_z1, axes=None):
        if self.mode == '3D':
            axes.plot_surface(mesh_x, mesh_y, mesh_z, cmap='rainbow')
            plt.savefig("%s_%s.png" %(self.args.model, self.args.attack_methods))
            # plt.savefig("%s_%s.pdf" %(self.args.model, self.args.attack_methods), format='pdf', bbox_inches='tight')

        if self.mode == 'Contour':
            plt.contourf(mesh_x, mesh_y, mesh_z, 1, cmap=get_rand_cmap(), alpha=alpha)

        if self.mode == '2D':
            plt.figure(figsize=(5, 4), dpi=500)
            plt.rc('font', family='Times New Roman', size=12)
            plt.plot(mesh_x, mesh_z, c='red', label="%s" %(self.args.attack_methods), linewidth=2.5)
            plt.plot(mesh_x, mesh_z1, c='green', label="ITAN-%s" %(self.args.attack_methods), linewidth=2.5)
            plt.xlim(self.mesh_x.min()-0.02, self.mesh_x.max()+0.02)
            plt.legend(loc='upper left')
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.xlabel(r'Magnitude $\alpha$')
            if self.y_axis_=='flat':
                plt.ylabel('Flatness')
            else:
                plt.ylabel('Loss') 
            plt.savefig("%s/%s_%s_%s.png" %(self.y_axis_, self.args.model, self.args.attack_methods, self.point_index if self.point_index is not None else '1'), bbox_inches='tight')
            plt.savefig("%s/%s_%s_%s.pdf" %(self.y_axis_, self.args.model, self.args.attack_methods, self.point_index  if self.point_index is not None else '1'), format='pdf', bbox_inches='tight')

    @staticmethod
    def get_datetime_str(style='dt'):
        import datetime
        cur_time = datetime.datetime.now()

        date_str = cur_time.strftime('%y_%m_%d_')
        time_str = cur_time.strftime('%H_%M_%S')

        if style == 'data':
            return date_str
        elif style == 'time':
            return time_str
        else:
            return date_str + time_str

    @staticmethod
    def project(x: torch.tensor, min=-0.18, max=0.18):
        return torch.clamp(x, min, max)
