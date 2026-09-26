"""Render measured Wave versus CMA-ES comparisons without rerunning experiments."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tests/optimization/reports/population-sweep-20d-100k'
PROBLEMS = {'quadratic': 'Quadratic bowl', 'bbob_10': 'Rotated ellipsoid',
            'rastrigin': 'Rastrigin', 'bbob_15': 'Rotated Rastrigin',
            'rosenbrock': 'Rosenbrock', 'bbob_5': 'Boundary optimum'}
METHODS = ['Gaussian', 'Local covariance', 'Bounded adaptive', 'BIPOP-active CMA-ES']
COLORS = ['#718096', '#187f97', '#d17825', '#208451']
FLOOR = 1e-14


def stats(values):
    q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
    return statistics.median(values), q1, q3


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=190, facecolor=fig.get_facecolor())
    fig.savefig(OUT / f'{name}.pdf', facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    rows = [json.loads(line) for line in (OUT / 'runs.jsonl').read_text().splitlines()]
    cma = [json.loads(line) for line in (OUT / 'cma-reference.jsonl').read_text().splitlines()]
    assert len(rows) == 1200 and len(cma) == 60
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.facecolor': '#fafaf8',
                         'axes.facecolor': '#fafaf8'})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.subplots_adjust(top=.80, bottom=.15, left=.075, right=.97, hspace=.65, wspace=.32)
    fig.text(.075, .95, 'Fractal movement strategies vs CMA-ES', fontsize=23, weight='bold')
    fig.text(.075, .90, '20 dimensions · 100,000-evaluation cap · 10 seeds · Wave: 128 walkers, 5 elites', fontsize=12)
    for ax, (problem, label) in zip(axes.flat, PROBLEMS.items()):
        upper, lower = [], []
        for i, (method, color) in enumerate(zip(METHODS, COLORS)):
            group = [r for r in (cma if i == 3 else rows) if r['problem'] == problem
                     and r['variant'] == method and (i == 3 or r['initial_walkers'] == 128)]
            assert len(group) == 10
            med, lo, hi = stats([r['regret'] for r in group])
            y = 3-i
            ax.plot([max(FLOOR,lo),max(FLOOR,hi)], [y,y], color=color, lw=5, alpha=.45, solid_capstyle='round')
            ax.scatter([max(FLOOR,med)], [y], color=color, s=65, zorder=3)
            ax.text(.98, y + .26, f'{med:.3g}', transform=ax.get_yaxis_transform(), va='center', ha='right', color=color, fontsize=9)
            upper.append(hi)
            lower.append(max(FLOOR, lo))
        ax.set_xscale('log')
        ax.set_xlim(min(lower) / 5, max(upper)*10)
        ax.set_ylim(-.6,3.6)
        ax.set_yticks(range(4), ['CMA-ES','Adaptive','Local cov.','Gaussian'])
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x',alpha=.18)
        ax.set_title(label,loc='left',pad=13)
        ax.set_xlabel('Objective error ↓  (log scale)',fontsize=10)
    fig.text(.075,.075,'Dots: median · thick intervals: middle 50% of seeds · printed values: exact medians (rounded)', fontsize=11)
    fig.text(.075,.04,'Errors below 10⁻¹⁴ are drawn at 10⁻¹⁴. CMA-ES uses its own population. Controller-enabled adaptive results overlap exactly: no restarts.', fontsize=10)
    save(fig,'fractal-vs-cma-128')

    fig, axes = plt.subplots(1,2,figsize=(13,6.8))
    fig.subplots_adjust(top=.77,bottom=.25,left=.075,right=.98,wspace=.25)
    fig.text(.075,.94,'Rastrigin: does a larger swarm close the gap?',fontsize=22,weight='bold')
    fig.text(.075,.885,'20 dimensions · 100,000-evaluation cap · 10 seeds · 5 elites for Wave',fontsize=12)
    sizes=[128,512,1024,2048,5000]
    for ax, problem in zip(axes,['rastrigin','bbob_15']):
        for method,color in zip(METHODS[:3],COLORS[:3]):
            summaries=[stats([r['regret'] for r in rows if r['problem']==problem and r['variant']==method and r['initial_walkers']==n]) for n in sizes]
            med,lo,hi=zip(*summaries)
            ax.plot(sizes,med,color=color,marker='o',lw=2,label=method)
            ax.fill_between(sizes,lo,hi,color=color,alpha=.12)
        med,lo,hi=stats([r['regret'] for r in cma if r['problem']==problem])
        ax.axhspan(lo,hi,color=COLORS[3],alpha=.12)
        ax.axhline(med,color=COLORS[3],ls='--',lw=2)
        ax.annotate(f'CMA-ES median: {med:.2f}',xy=(.03,med),xycoords=('axes fraction','data'),xytext=(0,8),textcoords='offset points',color=COLORS[3],fontsize=11)
        ax.set_xscale('log');ax.set_yscale('log')
        ax.set_xticks(sizes,[f'{n:,}' for n in sizes])
        ax.set_ylim(1,1000)
        ax.set_title(PROBLEMS[problem],loc='left',pad=12)
        ax.set_xlabel('Starting walkers (Wave only)');ax.set_ylabel('Objective error ↓  (log scale)')
        ax.grid(axis='y',alpha=.2)
    handles=[Line2D([0],[0],color=c,lw=2,ls='--' if i==3 else '-',label=m) for i,(m,c) in enumerate(zip(METHODS,COLORS))]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.10),ncol=4,frameon=False)
    fig.text(.075,.065,'Lines: medians · shading: middle 50% of seeds. CMA-ES reference is unchanged across the population axis.',fontsize=10)
    fig.text(.075,.027,'128 walkers received 706–780 movement steps across the suite; 5,000 received 15–19. No controller restarts occurred.',fontsize=10)
    save(fig,'fractal-vs-cma-rastrigin')


if __name__ == '__main__':
    main()
