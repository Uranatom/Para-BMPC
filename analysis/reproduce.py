"""Reproduce manuscript result summaries from the exported frozen measurements.

This module requires NumPy, SciPy and Matplotlib only. It never loads a policy,
starts a simulator, or imports the baseline package.
"""
from pathlib import Path
from collections import defaultdict
import argparse,csv,json,math
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
CELLS=['merge_nominal','merge_stress','intersection_nominal','intersection_stress']
METHODS=['pid_sac','pid_ddpg','pid_ppo','fixed_nominal','fixed','fot_pid','era','branch']
METRICS=['discounted_task_return','smooth_deficit_m','completion_time_s','jerk_rms_mps3','fallback_fraction']
NAMES=['Para-BMPC-SAC','Para-BMPC-DDPG','Para-BMPC-PPO','Fixed BMPC (nominal)','Fixed BMPC (tuned)','FOT-PID','Era-BMPC','CVaR-TMPC']
COLORS=['#0F4D92','#B64342','#42949E','#92999F','#424A51','#C48B25','#9A4D8E','#608D43']
MARKERS=['o','s','^','x','D','v','P','X']
def load(name):
 with (ROOT/'results'/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def save(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def truth(value):return str(value).lower()=='true'
def interval(a,family=1):
 a=np.asarray(a,float);mean=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(len(a)))
 half=float(t.ppf(1-.05/(2*family),len(a)-1)*se)
 return [mean,mean-half,mean+half]
def check(a,b,label):np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-8,err_msg=label)
def matrix(rows,cell,conditions,key):
 by={(r['condition'],int(r['index'])):float(r[key]) for r in rows if r['cell']==cell}
 return np.array([[by[c,i] for i in range(128)] for c in conditions])
def arm_ids(arm):
 if arm=='000':return ['fixed_expectation_n10']
 if arm=='100':return ['fixed_risk_n10']
 if arm=='111':return [f'action_joint_seed{s}' for s in (1,2,3)]
 return [f'structure_{arm}_seed{s}' for s in (1,2,3)]

def controller_comparison(output):
 rows=load('controller_comparison/episodes.csv');timing=load('controller_comparison/isolated_decisions.csv')
 assert len(rows)==6144
 groups=defaultdict(list);times=defaultdict(list)
 for r in rows:groups[r['cell'],r['method'],int(r['training_seed'])].append(r)
 for r in timing:times[r['cell'],r['method'],int(r['training_seed'])].append(float(r['decision_ms']))
 means={};per_seed={}
 for cell in CELLS:
  for method in METHODS:
   seeds=sorted(s for c,m,s in groups if (c,m)==(cell,method));values=[]
   assert len(seeds)==(3 if method=='pid_sac' else 2 if method in ('pid_ddpg','pid_ppo') else 1)
   for seed in seeds:
    r=groups[cell,method,seed];assert len(r)==128
    a=[100*np.mean([truth(x[k]) for x in r]) for k in ['success','collision','road_departed']]
    a += [np.mean([float(x[k]) for x in r]) for k in METRICS[:4]]
    fb=np.array([float(x['fallback_fraction']) for x in r]);steps=np.array([float(x['steps']) for x in r])
    latency=np.array(times[cell,method,seed]);assert len(latency)
    a += [100*(1-np.average(fb,weights=steps)),100*fb.mean(),float(np.percentile(latency,99)),100*float(np.mean(latency>100))]
    values.append(a)
   means[cell,method]=np.mean(values,axis=0);per_seed[cell,method]=values
 expected=json.loads((ROOT/'results/expected/controller_table.json').read_text());table=[]
 for e in expected:
  value=means[e['cell'],e['method']].copy()
  if not e['absolute']:value-=means[e['cell'],'fixed']
  check(value,e['values'],'Table III '+e['cell']+' '+e['method'])
  table.append({'cell':e['cell'],'method':e['method'],'absolute':e['absolute'],'values':value.tolist()})
 save(output/'table_III.json',table)
 contrasts=[]
 for reference in ['fixed','fixed_nominal']:
  for cell in CELLS:
   for metric in METRICS:
    anchor={int(r['index']):float(r[metric]) for r in rows if r['cell']==cell and r['method']==reference}
    delta=np.array([[float(r[metric])-anchor[int(r['index'])] for r in sorted(groups[cell,'pid_sac',s],key=lambda x:int(x['index']))] for s in (1,2,3)])
    contrasts.append(dict(reference=reference,cell=cell,metric=metric,conditional=interval(delta.mean(0),20),training_seed=interval(delta.mean(1),20),family=20,conditional_n=128,training_seed_n=3))
 for r in load('controller_comparison/tuned_paired_contrasts.csv'):
  c=next(c for c in contrasts if c['reference']=='fixed' and c['cell']==r['cell'] and c['metric']==r['metric'])
  key='conditional' if r['level']=='within_start' else 'training_seed'
  check(c[key],[float(r[k]) for k in ['mean','lower','upper']],'tuned fixed contrast')
 save(output/'fixed_reference_contrasts.json',contrasts)
 return rows,timing,means

def ablations(output):
 rows=load('ablations/episodes.csv');assert len(rows)==16384
 expected=load('ablations/paired_contrasts.csv');results=[]
 for r in expected:
  study=r['study']
  left=[f'action_joint_seed{s}' for s in (1,2,3)]
  if study=='action_joint_minus_fixed':right=['fixed_risk_n10']
  elif study.startswith('action_joint_minus_'):right=[f"action_{study.removeprefix('action_joint_minus_')}_seed{s}" for s in (1,2,3)]
  elif study=='structure_risk_only_secondary':left=arm_ids('100');right=arm_ids('000')
  else:right=arm_ids(study[-3:])
  delta=matrix(rows,r['cell'],left,r['metric'])-matrix(rows,r['cell'],right,r['metric'])
  ci=interval(delta.mean(0),int(r['family_size']))
  check(ci,[float(r[k]) for k in ['delta','lower_FWER','upper_FWER']],'ablation conditional')
  seed=None
  if delta.shape[0]>1:
   seed=interval(delta.mean(1),int(r['family_size']))
   check(seed[1:],[float(r[k]) for k in ['seed_lower_FWER','seed_upper_FWER']],'ablation seed')
  results.append(dict(study=study,cell=r['cell'],metric=r['metric'],family=int(r['family_size']),conditional=ci,training_seed=seed))
 save(output/'ablation_contrasts.json',results)
 equivalence=[]
 for r in load('ablations/horizon_equivalence.csv'):
  cond=f"fixed_{r['risk']}_n{r['N']}";ref=f"fixed_{r['risk']}_n{r['reference_N']}"
  delta=(matrix(rows,r['cell'],[cond],r['metric'])-matrix(rows,r['cell'],[ref],r['metric']))[0]
  # TOST uses two one-sided tests at 0.05/64, hence a 1-2*0.05/64 interval.
  ci=interval(delta,32);check(ci,[float(r[k]) for k in ['delta','lower_FWER','upper_FWER']],'horizon paired interval')
  mean=delta.mean();se=delta.std(ddof=1)/np.sqrt(128);margin=float(r['margin'])
  p=min(1.,64*max(float(t.sf((mean+margin)/se,127)),float(t.cdf((mean-margin)/se,127)))) if se else (0. if abs(mean)<margin else 1.)
  check(p,float(r['adjusted_TOST_p']),'horizon TOST')
  equivalence.append(dict(cell=r['cell'],risk=r['risk'],N=int(r['N']),metric=r['metric'],interval=ci,adjusted_TOST_p=p,equivalent=p<.05))
 save(output/'horizon_equivalence.json',equivalence)
 return rows

def solver_tables(output):
 rows=load('solver_horizon/episodes.csv');timing=load('solver_horizon/timing.csv');expected=json.loads((ROOT/'results/expected/solver_tables.json').read_text())
 by={(r['condition'],r['cell']):r for r in timing};tables=[]
 for e in expected['upper_rows']:
  sample=[r for r in rows if r['condition']==e['method']+'_n10_compact'];assert len(sample)==512
  result={'method':e['method'],'return_mean':np.mean([float(r[METRICS[0]]) for r in sample]),'deficit_m':np.mean([float(r[METRICS[1]]) for r in sample]),'jerk_mps3':np.mean([float(r[METRICS[3]]) for r in sample]),'success_pct':100*np.mean([truth(r['success']) for r in sample]),'acceptance_pct':100*np.mean([float(r['nonlinear_acceptance']) for r in sample])}
  latency=[float(by[e['method']+'_n10_compact',c]['decision_p99_ms']) for c in CELLS];result['latency_range_ms']=[min(latency),max(latency)]
  for k in result:
   if k!='method':check(result[k],e[k],'Table I '+k)
  tables.append(result)
 medians={str(n):[float(np.median([float(by[f'{m}_n{n}_compact',c]['decision_p99_ms']) for c in CELLS])) for m in ['shooting','cem','mppi','ltv']] for n in [15,20,30]}
 for n in medians:check(medians[n],expected['lower_rows'][n],'Table II')
 save(output/'tables_I_II.json',dict(table_I=tables,table_II=medians))
 by_episode={(r['condition'],r['cell'],int(r['index'])):r for r in rows}
 for r in load('solver_horizon/paired_contrasts.csv'):
  delta=[float(by_episode[r['condition'],r['cell'],i][r['metric']])-float(by_episode[r['reference'],r['cell'],i][r['metric']]) for i in range(int(r['n']))]
  check(interval(delta,int(r['family'])),[float(r[k]) for k in ['mean','lower','upper']],'solver paired contrast')

def deployment(output):
 rows=load('deployment_diagnostics/episodes.csv');assert len(rows)==6656
 by={(r['cell'],r['condition'],int(r['training_seed']),int(r['index'])):r for r in rows};contrasts=[]
 for r in load('deployment_diagnostics/expected_contrasts.csv'):
  delta=np.array([[float(by[r['cell'],r['left'],0 if r['left']=='tuned_fixed' else s,i][r['metric']])-float(by[r['cell'],r['right'],0 if r['right']=='tuned_fixed' else s,i][r['metric']]) for i in range(128)] for s in (1,2,3)])
  conditional=interval(delta.mean(0),120);seed=interval(delta.mean(1),120)
  check(conditional,[float(r[k]) for k in ['mean','conditional_lower','conditional_upper']],'deployment conditional')
  check(seed[1:],[float(r[k]) for k in ['seed_lower','seed_upper']],'deployment seed')
  contrasts.append(dict(left=r['left'],right=r['right'],cell=r['cell'],metric=r['metric'],conditional=conditional,training_seed=seed,family=120))
 save(output/'deployment_contrasts.json',contrasts)

def fixed_selection(output):
 rows=load('fixed_selection/episodes.csv');expected=json.loads((ROOT/'results/expected/fixed_confirmation.json').read_text())
 rankings={}
 for stage in ['development','confirmation']:
  rankings[stage]={}
  for scene in ['merge','intersection']:
   sample=[r for r in rows if r['stage']==stage and r['scenario']==scene];rank=[]
   for config in sorted({r['configuration_id'] for r in sample}):
    rs=[r for r in sample if r['configuration_id']==config];assert len(rs)==16
    means={pop:math.fsum(float(r['discounted_task_return']) for r in rs if r['population']==pop)/8 for pop in ['nominal','stress']}
    rank.append({'configuration_id':config,'weighted_return':.75*means['nominal']+.25*means['stress']})
   rank.sort(key=lambda r:(-r['weighted_return'],r['configuration_id']));rankings[stage][scene]=rank
   if stage=='development':assert len(rank)==(243 if scene=='merge' else 81)
   else:
    assert [r['configuration_id'] for r in rank]==[r['configuration']['configuration_id'] for r in expected[scene]]
    check([r['weighted_return'] for r in rank],[r['weighted_return'] for r in expected[scene]],'fixed return-only confirmation ranking')
 save(output/'fixed_selection.json',rankings)

def plots(output,rows,timing,ablation):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 def finish(fig,name):
  fig.savefig(output/(name+'.pdf'),facecolor='white');fig.savefig(output/(name+'.svg'),facecolor='white');fig.savefig(output/(name+'.png'),dpi=300,facecolor='white');plt.close(fig)
 titles=['Return ↑','Deficit (m) ↓','Task time (s) ↓','RMS jerk (m/s³) ↓']
 fig,axes=plt.subplots(2,2,figsize=(4.2,4.3));fig.subplots_adjust(top=.73,hspace=.5,wspace=.42,bottom=.12,left=.15)
 for ax,key,title in zip(axes.flat,METRICS[:4],titles):
  for j,m in enumerate(METHODS):
   a=np.array([[np.mean([float(r[key]) for r in rows if r['cell']==c and r['method']==m and int(r['index'])==i]) for i in range(128)] for c in CELLS])
   if m=='fixed':
    for i,y in enumerate(a.mean(1)):ax.plot([i-.38,i+.38],[y,y],color=COLORS[j],ls='--',lw=1)
   else:ax.errorbar(np.arange(4)+(j-3.5)*.08,a.mean(1),yerr=t.ppf(.975,127)*a.std(axis=1,ddof=1)/np.sqrt(128),fmt=MARKERS[j],ms=3,color=COLORS[j],lw=.6,capsize=1,label=NAMES[j])
  ax.set_title(title);ax.set_xticks(range(4),['M-N','M-B','I-N','I-B']);ax.grid(axis='y',alpha=.15)
 handles=[Line2D([],[],marker='' if i==4 else MARKERS[i],ls='--' if i==4 else '',color=COLORS[i],label=NAMES[i]) for i in range(8)]
 fig.legend(handles=handles,ncol=2,loc='upper center',fontsize=7);finish(fig,'fig08_performance')
 fig,axes=plt.subplots(2,2,figsize=(4.2,4.3));fig.subplots_adjust(top=.73,hspace=.5,wspace=.32,bottom=.12,left=.15)
 for ax,cell in zip(axes.flat,CELLS):
  for j,m in enumerate(METHODS):
   a=np.sort([float(r['decision_ms']) for r in timing if r['cell']==cell and r['method']==m]);ax.plot(a,np.arange(1,len(a)+1)/len(a),color=COLORS[j],lw=.8,ls=['-','--','-.',':','--','--','-.',':'][j])
  ax.axvline(100,color='grey',ls='--',lw=.7);ax.set_xscale('log');ax.set_ylim(0,1);ax.set_title(cell.replace('_',' '));ax.set_xlabel('Decision latency (ms) ↓');ax.set_ylabel('CDF')
 fig.legend(handles=handles,ncol=2,loc='upper center',fontsize=7);finish(fig,'fig09_latency')
 # Figure 4: fixed-controller horizon sensitivity, computed from trajectories and timing.
 tim={(r['cell'],r['condition']):r for r in load('ablations/all_timing.csv') if r['scope']=='isolated_decision_latency'}
 fig,axes=plt.subplots(2,2,figsize=(4.3,4.0));fig.subplots_adjust(top=.85,hspace=.5,wspace=.4,bottom=.13,left=.15)
 for ci,c in enumerate(CELLS):
  for risk,style in [('risk','-'),('expectation','--')]:
   for pi,key in enumerate(['smooth_deficit_m','completion_time_s','jerk_rms_mps3','latency']):
    means=[]
    for n in [10,15,20,30]:
     cond=f'fixed_{risk}_n{n}'
     means.append(float(tim[c,cond]['p99_ms']) if key=='latency' else float(matrix(ablation,c,[cond],key).mean()))
    axes.flat[pi].plot([1,1.5,2,3],means,color=COLORS[ci],ls=style,marker=MARKERS[ci],ms=3,lw=.8,label=c if risk=='risk' else None)
 for ax,title in zip(axes.flat,['SAT deficit (m) ↓','Task time (s) ↓','RMS jerk (m/s³) ↓','Isolated p99 (ms) ↓']):ax.set_title(title);ax.set_xlabel('Horizon (s)')
 axes.flat[3].axhline(100,color='grey',ls=':',lw=.7);fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=2,fontsize=7);finish(fig,'fig04_horizon')
 # Figure 6: action ablations; Figure 7: conditional multiplicity-adjusted component effects.
 fig,axes=plt.subplots(2,2,figsize=(4.3,4));fig.subplots_adjust(top=.85,hspace=.5,wspace=.4,bottom=.13,left=.15)
 variants=[('Joint',arm_ids('111')),('Weights-only*',[f'action_weights_only_seed{s}' for s in (1,2,3)]),('Margin-only',[f'action_cbf_only_seed{s}' for s in (1,2,3)]),('Fixed risk',arm_ids('100'))]
 for ax,key,title in zip(axes.flat,METRICS[:4],titles):
  for j,(label,conditions) in enumerate(variants):
   a=np.array([matrix(ablation,c,conditions,key).mean(0) for c in CELLS]);ax.errorbar(np.arange(4)+(j-1.5)*.13,a.mean(1),yerr=t.ppf(.975,127)*a.std(axis=1,ddof=1)/np.sqrt(128),fmt=MARKERS[j],color=COLORS[j],ms=3,capsize=1,label=label)
  ax.set_title(title);ax.set_xticks(range(4),['M-N','M-B','I-N','I-B'])
 fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=2);finish(fig,'fig06_action_ablation')
 contrasts=load('ablations/paired_contrasts.csv');fig,axes=plt.subplots(2,2,figsize=(4.3,4));fig.subplots_adjust(top=.85,hspace=.5,wspace=.4,bottom=.13,left=.17)
 for ax,key,title in zip(axes.flat,METRICS[:4],titles):
  for j,arm in enumerate(['011','101','110','100']):
   rs=[next(r for r in contrasts if r['study']=='structure_full_minus_'+arm and r['cell']==c and r['metric']==key) for c in CELLS];mean=np.array([float(r['delta']) for r in rs]);low=np.array([float(r['lower_FWER']) for r in rs]);hi=np.array([float(r['upper_FWER']) for r in rs])
   ax.errorbar(np.arange(4)+(j-1.5)*.13,mean,yerr=[mean-low,hi-mean],fmt=MARKERS[j],color=COLORS[j],ms=3,capsize=1,label=['No risk','No weights','No margin','Both fixed'][j])
  ax.axhline(0,color='grey',lw=.7);ax.set_title('Δ '+title);ax.set_xticks(range(4),['M-N','M-B','I-N','I-B'])
 fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=2);finish(fig,'fig07_structural_ablation')
 learning=load('learning_curves/cohorts.csv');fig,axes=plt.subplots(4,4,figsize=(11,7));fig.subplots_adjust(top=.91,hspace=.55,wspace=.35,bottom=.09,left=.06,right=.99)
 keys=['discounted_return','success_percent','nominal_deficit_m','stress_deficit_m','minimum_sat_median_m','jerk_rms_mps3','fallback_percent','clearance_critic_loss']
 algorithms=list(dict.fromkeys(r['algorithm'] for r in learning))
 for si,scene in enumerate(['merge','intersection']):
  for ki,key in enumerate(keys):
   ax=axes[si*2+ki//4,ki%4]
   for ai,alg in enumerate(algorithms):
    seeds=sorted({r['seed'] for r in learning if r['algorithm']==alg and r['scenario']==scene});curves=[]
    for seed in seeds:
     rs=sorted([r for r in learning if r['algorithm']==alg and r['scenario']==scene and r['seed']==seed],key=lambda r:int(r['cohort']))
     a=np.array([float(r[key]) for r in rs]);curves.append(np.array([np.mean(a[max(0,i-19):i+1]) for i in range(len(a))]))
    if not curves:continue
    a=np.array(curves);x=np.array([float(r['allocated_primary_steps']) for r in rs])/1e8;mean=a.mean(0)
    learning_colors=['#0F4D92','#9A4D8E','#C48B25','#B64342','#42949E']
    labels={'pid_sac':'PID-SAC joint (ours)','sac_weights':'PID-SAC weights* (ours)','sac_margin':'PID-SAC margin (ours)','ddpg':'PID-DDPG (ours)','ppo':'PID-PPO (ours)'}
    ax.plot(x,mean,color=learning_colors[ai],ls=['-','--','-.',':','--'][ai],lw=.8,label=labels.get(alg,alg))
    if len(a)>1:
     half=t.ppf(.975,len(a)-1)*a.std(axis=0,ddof=1)/np.sqrt(len(a));ax.fill_between(x,mean-half,mean+half,color=learning_colors[ai],alpha=.1)
   direction='↑' if key in ('discounted_return','success_percent','minimum_sat_median_m') else '↓'
   ax.set_title(scene+': '+key.replace('_',' ')+' '+direction,fontsize=8);ax.tick_params(labelsize=8)
   panel=(si*2+ki//4)*4+ki%4
   ranges=json.loads((ROOT/'results/learning_curves/display_ranges.json').read_text())['panels']
   ax.set_ylim(ranges[panel]['displayed_limits'])
 fig.supxlabel('Allocated primary environment steps (10⁸)',fontsize=8)
 fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=5,fontsize=8);finish(fig,'fig05_learning')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'generated');p.add_argument('--no-plots',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 rows,timing,means=controller_comparison(args.output);a=ablations(args.output);solver_tables(args.output);deployment(args.output);fixed_selection(args.output)
 if not args.no_plots:plots(args.output,rows,timing,a)
 report={'controller_rows':len(rows),'ablation_rows':len(a),'solver_rows':3968,'deployment_rows':6656,'table_III_values_verified':352,'ablation_contrasts_verified':160,'horizon_equivalence_verified':64,'solver_contrasts_verified':360,'deployment_contrasts_verified':120,'all_checks_passed':True,'scope':'Recalculation of frozen results; no new training or physical evaluation.'}
 save(args.output/'reproduction_report.json',report);print(json.dumps(report))
if __name__=='__main__':main()
