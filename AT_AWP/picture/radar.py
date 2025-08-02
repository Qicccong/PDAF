import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):

    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, **kwargs):
            self.set_thetagrids(np.degrees(theta), labels, **kwargs)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def example_data():
    # Fill with your actual data
    data = [
        ['userFC', 'itemFC', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6', 'layer.Final'],
        ('Epoch0: fcs.0', [
            [0.502, 0.05, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.05],
            [0.05, 0.300, 0.05, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.05, 1.000, 0.05, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.05, 0.666, 0.05, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.05, 0.472, 0.05, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.05, 0.500, 0.05, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.05, 0.520, 0.05, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.05, 0.514, 0.05],
            [0.05, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.05, 0.461],
        ]),
        ('Epoch10: userFC', [
            [1.000, 0.05, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.05],
            [0.05, 0.439, 0.05, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.05, 0.140, 0.05, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.05, 0.382, 0.05, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.05, 0.257, 0.05, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.05, 0.196, 0.05, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.05, 0.202, 0.05],
            [0.05, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.05, 0.212],
        ]),
        ('Epoch20: fcs.2', [
            [0.630, 0.05, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.05],
            [0.05, 0.470, 0.05, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.05, 1.000, 0.05, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.05, 0.580, 0.05, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.05, 0.350, 0.05, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.05, 0.310, 0.05, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.05, 0.300, 0.05],
            [0.05, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.05, 0.340],
        ]),
        ('Epoch30: itemFC', [
            [0.133, 0.05, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.05],
            [0.05, 1.000, 0.05, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.05, 0.144, 0.05, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.05, 0.199, 0.05, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.05, 0.540, 0.05, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.05, 0.556, 0.05, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.05, 0.533, 0.05],
            [0.05, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.05, 0.611],
        ])
    ]

    return data


def create_radar_chart(data, num_vars, frame='polygon'):
    theta = radar_factory(num_vars, frame=frame)
    spoke_labels = data[0]

    fig, axs = plt.subplots(figsize=(9, 9), nrows=2, ncols=2,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['#2C92E0', '#3ABF99', '#F0A73A'] * 3

    # 绘制子图内容
    for ax, (title, case_data) in zip(axs.flat, data[1:]):
        ax.set_title(title, weight='bold', size=16, position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels, fontsize=14)
        ax.set_yticklabels([])
        # ax.yaxis.grid(False)  # 关闭径向网格线

    # 在整体图右上角添加两列标签
    label_style = {
        'fontsize': 20,
        'weight': 'bold',
        'color': 'black',
        'bbox': dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.8)
    }

    # 标签布局参数
    label_x_start = 0.90  # 起始x位置（右侧）
    label_y_start = 0.92  # 起始y位置（顶部）
    col_spacing = 0.06    # 列间距
    row_spacing = 0.05    # 行间距

    # 定义两列标签位置（左列：A/C，右列：B/D）
    label_positions = [
        (label_x_start, label_y_start),           # A (左列顶部)
        (label_x_start + col_spacing, label_y_start),  # B (右列顶部)
        (label_x_start, label_y_start - row_spacing),  # C (左列底部)
        (label_x_start + col_spacing, label_y_start - row_spacing)  # D (右列底部)
    ]

    # # 添加标签
    # for (x, y), label in zip(label_positions, ['A', 'B', 'C', 'D']):
    #     fig.text(
    #         x=x,
    #         y=y,
    #         s=label,
    #         ha='right',  # 右对齐
    #         va='top',    # 顶部对齐
    #         **label_style
    #     )

    plt.show()


if __name__ == '__main__':
    # plt.rcParams.update({'font.size': 14})  # 全局字体调整
    data = example_data()
    create_radar_chart(data, 9, frame='polygon')