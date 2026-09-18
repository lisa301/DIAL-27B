"""Generate publication-ready figures from test.xls.

Each figure groups only experiments sharing the same independent variable.  The
script intentionally does not graph the CPU/MEM/NPU columns: their units are
not specified consistently in the source workbook (fractions, MB and hardware
utilization are mixed), so a publication-quality comparison would be misleading.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


ROOT = Path('/home/seaway/sdb/ljl/Dial_llama/result')
SOURCE = ROOT / 'test.xls'
CN_FONT = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
MARKERS = ['o', 's', '^', 'D']
LINESTYLES = ['-', '--', '-.', ':']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'axes.linewidth': 0.75,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'savefig.pad_inches': 0.04,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def set_cn_text(ax, *, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontproperties=CN_FONT, fontsize=11.5, pad=7)
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=CN_FONT, fontsize=10.5, labelpad=5)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=CN_FONT, fontsize=10.5, labelpad=5)


def style_axis(ax, *, ygrid=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#3A3A3A')
    ax.spines['bottom'].set_color('#3A3A3A')
    ax.tick_params(axis='both', labelsize=9.5, colors='#222222', width=0.65, length=3)
    if ygrid:
        ax.grid(axis='y', color='#D0D0D0', lw=0.55, linestyle=(0, (2, 2)), zorder=0)


def save(fig, name):
    fig.savefig(ROOT / f'{name}.png', dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(ROOT / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def shared_legend(fig, handles, labels, *, y=0.985, ncol=3):
    fig.legend(handles, labels, prop=CN_FONT, ncol=ncol, loc='upper center',
               bbox_to_anchor=(0.5, y), frameon=False, handlelength=2.2,
               columnspacing=1.45, handletextpad=0.55)


def read_data():
    return {
        'platform': pd.read_excel(SOURCE, sheet_name='rk3588+orin', header=None),
        'single_orin': pd.read_excel(SOURCE, sheet_name='单orin', header=None),
        'quant': pd.read_excel(SOURCE, sheet_name='优化后', header=None),
        'network': pd.read_excel(SOURCE, sheet_name='限速测试', header=None),
        'reuse': pd.read_excel(SOURCE, sheet_name='视频帧复用', header=None),
    }


def plot_platform(d):
    x = d.iloc[2:9, 0].astype(float).to_numpy()
    metrics = [
        ('（a）首 token 延迟（TTFT）', d.iloc[2:9, 9].astype(float).to_numpy(), 'TTFT / s'),
        ('（b）推理吞吐率（TPS）', d.iloc[2:9, 10].astype(float).to_numpy(), 'tokens / s'),
        ('（c）Decode 吞吐率', d.iloc[2:9, 11].astype(float).to_numpy(), 'tokens / s'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35), dpi=600)
    for ax, (title, y, ylabel), color, marker in zip(axes, metrics, COLORS, MARKERS):
        ax.plot(x, y, color=color, lw=1.65, marker=marker, ms=4.5,
                markerfacecolor='white', markeredgewidth=1.15, zorder=3)
        set_cn_text(ax, title=title, xlabel='RK3588 执行层数', ylabel=ylabel)
        ax.set_xticks(x)
        ax.set_xlim(-0.15, 6.15)
        style_axis(ax)
    fig.tight_layout(w_pad=2.0, pad=0.8)
    save(fig, '端侧推理性能')


def plot_quantization(d):
    x = d.iloc[2:6, 0].astype(float).to_numpy()
    methods = [
        ('FNN+DOWN 量化', (1, 2, 3)),
        ('Im_head 量化', (5, 6, 7)),
        ('未量化', (11, 12, 13)),
    ]
    panels = [
        ('（a）首 token 延迟（TTFT）', 0, 'TTFT / s'),
        ('（b）推理吞吐率（TPS）', 1, 'tokens / s'),
        ('（c）Decode 吞吐率', 2, 'tokens / s'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.45), dpi=600)
    handles = []
    for panel_idx, (ax, (title, metric_idx, ylabel)) in enumerate(zip(axes, panels)):
        for method_idx, (label, cols) in enumerate(methods):
            y = d.iloc[2:6, cols[metric_idx]].astype(float).to_numpy()
            line, = ax.plot(x, y, color=COLORS[method_idx], linestyle=LINESTYLES[method_idx],
                            lw=1.6, marker=MARKERS[method_idx], ms=4.5,
                            markerfacecolor='white', markeredgewidth=1.1,
                            label=label, zorder=3 + method_idx)
            if panel_idx == 0:
                handles.append(line)
        set_cn_text(ax, title=title, xlabel='RK3588 执行层数', ylabel=ylabel)
        ax.set_xticks(x)
        ax.set_xlim(-0.12, 3.12)
        style_axis(ax)
    shared_legend(fig, handles, [m[0] for m in methods], ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=2.0, pad=0.8)
    save(fig, '量化')


def network_matrices(d):
    # Read only raw rows carrying an explicit bandwidth label. The worksheet may
    # contain derived summary tables or omit a bandwidth group altogether.
    records = []
    for _, row in d.iterrows():
        label = row.iloc[1]
        if not isinstance(label, str) or not label.endswith(('Mb/s', 'Kb/s')):
            continue
        unit = 0.001 if label.endswith('Kb/s') else 1.0
        bandwidth = float(label.removesuffix('Kb/s').removesuffix('Mb/s')) * unit
        if pd.isna(row.iloc[2]) or pd.isna(row.iloc[3]) or pd.isna(row.iloc[4]) or pd.isna(row.iloc[5]):
            continue
        records.append({
            'bandwidth': bandwidth,
            'layers': int(row.iloc[2]),
            'ttft': float(row.iloc[3]),
            'prefill': float(row.iloc[4]),
            'decode': float(row.iloc[5]),
        })
    result = pd.DataFrame(records).sort_values('bandwidth')
    if result.empty or set(result['layers']) != set(range(4)):
        raise ValueError('限速测试工作表缺少完整的 0--3 层原始测量数据。')
    bandwidth = np.array(sorted(result['bandwidth'].unique()))
    matrices = {}
    for metric in ['ttft', 'prefill', 'decode']:
        matrices[metric] = np.vstack([
            result.loc[result.layers == layer].set_index('bandwidth').loc[bandwidth, metric].to_numpy()
            for layer in range(4)
        ])
    return bandwidth, matrices


def plot_network(d):
    bandwidth, matrices = network_matrices(d)
    panels = [
        ('（a）首 token 延迟（TTFT）', 'ttft', 'TTFT / s'),
        ('（b）推理吞吐率（TPS）', 'prefill', 'tokens / s'),
        ('（c）Decode 吞吐率', 'decode', 'tokens / s'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.55), dpi=600)
    handles = []
    for panel_idx, (ax, (title, metric, ylabel)) in enumerate(zip(axes, panels)):
        for layer in range(4):
            line, = ax.plot(bandwidth, matrices[metric][layer], color=COLORS[layer],
                            linestyle=LINESTYLES[layer], lw=1.55, marker=MARKERS[layer],
                            ms=4.2, markerfacecolor='white', markeredgewidth=1.0,
                            label=f'RK3588 执行 {layer} 层', zorder=3)
            if panel_idx == 0:
                handles.append(line)
        ax.set_xscale('log')
        ax.set_xlim(bandwidth.min() * 0.85, bandwidth.max() * 1.25)
        # 80 and 100 Mb/s are both measured points but too close to label
        # separately on a log axis; markers still show both observations.
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(['1', '10', '100'])
        set_cn_text(ax, title=title, xlabel='网络带宽 / Mb/s', ylabel=ylabel)
        style_axis(ax, ygrid=True)
        ax.grid(axis='x', color='#E3E3E3', lw=0.45, linestyle=(0, (2, 2)), zorder=0)
    shared_legend(fig, handles, [f'RK3588 执行 {i} 层' for i in range(4)], y=0.99, ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=2.0, pad=0.8)
    save(fig, '带宽影响')

    # A publication-sized TTFT panel is also kept under the original filename.
    fig, ax = plt.subplots(figsize=(6.3, 4.1), dpi=600)
    for layer in range(4):
        ax.plot(bandwidth, matrices['ttft'][layer], color=COLORS[layer],
                linestyle=LINESTYLES[layer], lw=1.75, marker=MARKERS[layer], ms=4.8,
                markerfacecolor='white', markeredgewidth=1.1,
                label=f'RK3588 执行 {layer} 层', zorder=3)
    ax.set_xscale('log')
    ax.set_xlim(bandwidth.min() * 0.85, bandwidth.max() * 1.25)
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(['1', '10', '100'])
    set_cn_text(ax, title='网络带宽对首 token 延迟的影响', xlabel='网络带宽 / Mb/s', ylabel='TTFT / s')
    style_axis(ax)
    ax.grid(axis='x', color='#E3E3E3', lw=0.45, linestyle=(0, (2, 2)), zorder=0)
    ax.legend(prop=CN_FONT, frameon=False, loc='upper right', handlelength=2.0,
              handletextpad=0.5, fontsize=9.5)
    fig.tight_layout(pad=0.8)
    save(fig, '首token')

    # The two throughput metrics belong together and replace the misleading old single plot.
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), dpi=600)
    for ax, (title, metric) in zip(axes, [('（a）推理吞吐率（TPS）', 'prefill'), ('（b）Decode 吞吐率', 'decode')]):
        for layer in range(4):
            ax.plot(bandwidth, matrices[metric][layer], color=COLORS[layer],
                    linestyle=LINESTYLES[layer], lw=1.5, marker=MARKERS[layer], ms=4.0,
                    markerfacecolor='white', markeredgewidth=0.9, zorder=3)
        ax.set_xscale('log')
        ax.set_xlim(bandwidth.min() * 0.85, bandwidth.max() * 1.25)
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(['1', '10', '100'])
        set_cn_text(ax, title=title, xlabel='网络带宽 / Mb/s', ylabel='tokens / s')
        style_axis(ax)
        ax.grid(axis='x', color='#E3E3E3', lw=0.45, linestyle=(0, (2, 2)), zorder=0)
    fig.legend(handles, [f'RK3588 执行 {i} 层' for i in range(4)], prop=CN_FONT,
               loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False,
               handlelength=2.0, columnspacing=1.15, handletextpad=0.45)
    fig.tight_layout(rect=(0, 0, 1, 0.82), w_pad=1.8, pad=0.8)
    save(fig, '推理速度')


def plot_reuse(d):
    x = d.iloc[3:7, 1].astype(float).to_numpy()
    before = d.iloc[3:7, 2].astype(float).to_numpy()
    after = d.iloc[3:7, 3].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(6.3, 4.1), dpi=600)
    for y, label, color, marker, linestyle in [
        (before, '未复用', COLORS[0], 'o', '-'),
        (after, '视频帧复用', COLORS[1], 's', '--'),
    ]:
        ax.plot(x, y, color=color, lw=1.85, linestyle=linestyle, marker=marker, ms=5.0,
                markerfacecolor='white', markeredgewidth=1.2, label=label, zorder=3)
    set_cn_text(ax, title='视频帧复用对首 token 延迟的影响',
                xlabel='RK3588 执行层数', ylabel='TTFT / s')
    ax.set_xticks(x)
    ax.set_xlim(-0.12, 3.12)
    ax.set_ylim(0, 8)
    ax.legend(prop=CN_FONT, loc='upper left', frameon=False, handlelength=2.15,
              handletextpad=0.5, fontsize=10)
    style_axis(ax)
    fig.tight_layout(pad=0.8)
    save(fig, '复用')


def plot_single_orin(d):
    labels = ['优化前', '优化后']
    values = np.array([
        [float(d.iloc[2, 0]), float(d.iloc[5, 0])],
        [float(d.iloc[2, 1]), float(d.iloc[5, 1])],
        [float(d.iloc[2, 2]), float(d.iloc[5, 2])],
    ])
    panels = [('（a）首 token 延迟（TTFT）', 'TTFT / s'), ('（b）推理吞吐率（TPS）', 'tokens / s'), ('（c）Decode 吞吐率', 'tokens / s')]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35), dpi=600)
    for idx, (ax, (title, ylabel)) in enumerate(zip(axes, panels)):
        y = values[idx]
        ax.plot([0, 1], y, color=COLORS[idx], lw=1.7, marker=MARKERS[idx], ms=5.0,
                markerfacecolor='white', markeredgewidth=1.2, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontproperties=CN_FONT)
        ax.set_xlim(-0.18, 1.18)
        margin = (y.max() - y.min()) * 0.28
        ax.set_ylim(y.min() - margin, y.max() + margin)
        set_cn_text(ax, title=title, xlabel='单 Orin 配置', ylabel=ylabel)
        style_axis(ax)
    fig.tight_layout(w_pad=2.0, pad=0.8)
    save(fig, '单Orin优化对比')


def main():
    data = read_data()
    plot_platform(data['platform'])
    plot_quantization(data['quant'])
    plot_network(data['network'])
    plot_reuse(data['reuse'])
    plot_single_orin(data['single_orin'])


if __name__ == '__main__':
    main()
