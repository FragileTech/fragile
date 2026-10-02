import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root=Path(__file__).resolve().parent
rows=[json.loads(line) for line in (root/'runs.jsonl').read_text().splitlines()]
manifest=json.loads((root/'manifest.json').read_text())
assert len(rows)==len(manifest['problems'])*len(manifest['variants'])*manifest['seeds']
variants=manifest['variants']
labels=['GAS noise · 10 swarms','GAS noise · single','Local covariance · 10 swarms','Local covariance · single','BIPOP-active CMA-ES']
colors=['#136f92','#83afc0','#713f9b','#b49bc9','#414141']
fig,axes=plt.subplots(2,3,figsize=(14,7.4),layout='constrained')
for ax,(problem,title) in zip(axes.flat,manifest['problems'].items()):
 for i,(variant,color) in enumerate(zip(variants,colors)):
  vals=[r['regret'] for r in rows if r['problem']==problem and r['variant']==variant]
  display=[max(v,1e-16) for v in vals]
  y=4-i
  ax.plot([min(display),max(display)],[y,y],color=color,alpha=.6,linewidth=2)
  ax.scatter(display,[y+.05*(j-2) for j in range(len(vals))],s=24,color=color,alpha=.6,zorder=3)
  ax.scatter([max(statistics.median(vals),1e-16)],[y],marker='D',s=55,color=color,edgecolor='white',linewidth=.7,zorder=4)
 ax.set_xscale('log')
 ax.set_yticks(range(5),list(reversed(labels)),fontsize=8)
 ax.set_title(title,loc='left',fontsize=11,fontweight='bold')
 ax.grid(axis='x',alpha=.15)
 ax.set_ylim(-.5,4.5)
 ax.set_xlabel('Error from optimum (lower is better)',fontsize=8)
 for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
 ax.tick_params(axis='y',length=0)
fig.suptitle('GAS: ten exchanging swarms versus one swarm and CMA-ES\n20 dimensions · 5 seeds · 100,000 evaluations per run',fontsize=15)
fig.supxlabel('Dots: individual seeds · Diamond: median · Line: observed range · Zero errors displayed at 10⁻¹⁶\n10 × 32 versus 1 × 320 walkers; heterogeneous population scales. Boundary handling and precision differ for CMA-ES.',fontsize=9)
fig.savefig(root/'comparison.png',dpi=160)
fig.savefig(root/'comparison.pdf')
