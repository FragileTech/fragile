"""Render measured cloning-guided ablations, marking groups with failed runs."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tests/optimization/reports/cloning-guided-20d-100k'
METHODS=['Gaussian','Local covariance','Bounded adaptive','Cloning geometry','Cloning drift','Cloning combined','BIPOP-active CMA-ES']
LABELS=['Gaussian','Local cov.','Bounded','Clone geometry','Clone drift','Combined','CMA-ES']
COLORS=['#82919d','#298091','#cc7e31','#7860ad','#b74c77','#244d92','#24835c']
PROBLEMS={'quadratic':'Quadratic bowl','bbob_10':'Rotated ellipsoid','rastrigin':'Rastrigin','bbob_15':'Rotated Rastrigin','rosenbrock':'Rosenbrock','bbob_5':'Boundary optimum'}

def main():
    rows=[json.loads(l) for l in (OUT/'runs.jsonl').read_text().splitlines()]
    assert len(rows)==420
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafaf8','axes.facecolor':'#fafaf8'})
    fig,axes=plt.subplots(2,3,figsize=(15,10));fig.subplots_adjust(left=.10,right=.98,top=.82,bottom=.14,wspace=.42,hspace=.43)
    fig.text(.10,.945,'Does cloning information improve movement?',fontsize=23,weight='bold')
    fig.text(.10,.90,'20 dimensions · 100,000-evaluation cap · 10 seeds · Wave: 128 walkers, 5 elites',fontsize=12)
    floor=1e-14
    for ax,(problem,title) in zip(axes.flat,PROBLEMS.items()):
        lower,upper=[],[]
        for i,(method,color) in enumerate(zip(METHODS,COLORS)):
            group=[r for r in rows if r['problem']==problem and r['variant']==method];assert len(group)==10
            vals=[r['regret'] for r in group];med=statistics.median(vals);q1,_,q3=statistics.quantiles(vals,n=4,method='inclusive');y=len(METHODS)-1-i
            failed=sum(bool(r['error']) for r in group)
            ax.plot([max(floor,q1),max(floor,q3)],[y,y],color=color,lw=5,alpha=.4,solid_capstyle='round')
            ax.scatter([max(floor,med)],[y],color=color,s=42,marker='x' if failed else 'o',zorder=3)
            ax.text(.98,y+.28,f'{med:.3g}'+(f'  [{failed}/10 failed]' if failed else ''),transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8,color=color)
            lower.append(max(floor,q1));upper.append(q3)
        ax.set_xscale('log');ax.set_xlim(min(lower)/4,max(upper)*12);ax.set_ylim(-.65,6.7)
        ax.set_yticks(range(7),list(reversed(LABELS)));ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',alpha=.18);ax.set_title(title,loc='left',weight='bold',pad=12);ax.set_xlabel('Objective error ↓  (log scale)')
    fig.text(.10,.075,'Dots: medians · thick intervals: middle 50% · crosses: groups with execution errors (last best value retained)',fontsize=10)
    fig.text(.10,.04,'Errors below 10⁻¹⁴ are drawn at 10⁻¹⁴. CMA-ES uses its own population and restarts; fractal automatic restarts are disabled.',fontsize=10)
    fig.savefig(OUT/'comparison.png',dpi=170);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
if __name__=='__main__':main()
