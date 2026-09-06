"""Local paired hybrid ablation with frozen PK validation chemistry and no test scoring."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from osmo.chemistry import features
from osmo.hybrid import Hybrid,FORMS
from osmo.metrics import evaluate
from osmo.io import write_json,digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--adme',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--epochs',type=int,default=60);p.add_argument('--max-seconds-per-arm',type=int,default=180)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
 ds=json.loads((a.root/'processed/latest.json').read_text())['dataset_id']
 pk=pd.read_parquet(a.root/'processed'/ds/'observations.parquet')
 pk=pk[pk.species.eq('rat')&pk.tissue.eq('plasma')&pk.split.isin(['train','validation'])].reset_index(drop=True)
 ad=pd.read_parquet(a.adme);ad=ad[ad.aux_split.isin(['train','aux_validation'])].reset_index(drop=True)
 TASKS=sorted(ad.task.unique())
 # Fit the common transform only on allowed training structures, never validation.
 allowed=pd.concat([pk.loc[pk.split.eq('train'),'smiles'],ad.loc[ad.aux_split.eq('train'),'smiles']]).unique()
 cache={s:features(s) for s in set(pk.smiles)|set(ad.smiles)}
 ref=np.stack([cache[s] for s in allowed]);mean=ref.mean(0);std=ref.std(0);std[std<.01]=1
 def tensor_x(smiles):return torch.tensor((np.stack([cache[s] for s in smiles])-mean)/std)
 px=tensor_x(pk.smiles);ax=tensor_x(ad.smiles)
 form=torch.tensor(np.column_stack([pk.formulation.eq(f) for f in FORMS]).astype(np.float32))
 ts=torch.tensor(pk.time_h.to_numpy(np.float32));oral=torch.tensor(pk.route.eq('oral').to_numpy(np.float32))
 infusion=torch.tensor(pk.infusion_h.to_numpy(np.float32))
 target=torch.tensor(np.log10(pk.concentration.to_numpy()/pk.dose_mg_kg.to_numpy()).astype(np.float32))
 train=np.flatnonzero(pk.split.eq('train'));val=np.flatnonzero(pk.split.eq('validation'))
 weights=np.zeros(len(pk),np.float32)
 for ids in [train,val]:
  q=pk.iloc[ids];w=1/(q.groupby('compound_id').curve_id.transform('nunique')*q.groupby(['compound_id','curve_id']).curve_id.transform('size'))
  weights[ids]=(w/w.mean()).to_numpy(np.float32)
 weights=torch.tensor(weights)
 task=np.array([TASKS.index(t) for t in ad.task]);atrain=np.flatnonzero(ad.aux_split.eq('train'));aval=np.flatnonzero(ad.aux_split.eq('aux_validation'))
 amean=np.array([ad.loc[(task==i)&ad.aux_split.eq('train'),'target'].mean() for i in range(len(TASKS))],np.float32)
 astd=np.array([ad.loc[(task==i)&ad.aux_split.eq('train'),'target'].std() for i in range(len(TASKS))],np.float32)
 if not np.all(np.isfinite(astd)&(astd>0)):raise ValueError('Insufficient auxiliary task coverage')
 ay=torch.tensor((ad.target.to_numpy(np.float32)-amean[task])/astd[task]);at=torch.tensor(task)
 train_by_task=[atrain[task[atrain]==i] for i in range(len(TASKS))]
 val_by_task=[aval[task[aval]==i] for i in range(len(TASKS))]
 def auxloss(model,groups,sample=False):
  losses=[]
  for ids in groups:
   if sample:ids=ids[torch.randint(len(ids),(min(64,len(ids)),)).numpy()]
   y=model.auxiliary(ax[ids])[torch.arange(len(ids)),at[ids]]
   losses.append(torch.nn.functional.smooth_l1_loss(y,ay[ids]))
  return torch.stack(losses).mean()
 def predict(model,ids):
  return torch.cat([model(px[j],form[j],ts[j],oral[j],infusion[j]) for j in np.array_split(ids,max(1,int(np.ceil(len(ids)/512))))])
 write_json(a.output/'protocol.json',{'pk_dataset':ds,'pk_input_hash':digest(a.root/'processed'/ds/'observations.parquet'),
  'adme_hash':digest(a.adme),'arms':['scratch','adme_assisted'],'epochs':a.epochs,'seconds_per_arm':a.max_seconds_per_arm,
  'pretrain_max_epochs':80,'pretrain_patience':12,'pk_patience':15,'seed':4107,
  'selection':'weighted validation log10 MAE','adme_joint_weight':.1,'test_evaluated':False,
  'scope':'rat plasma; direct baseline cohort retained including IV infusion; no human-organ claim'})
 results=[]
 for assisted in [False,True]:
  torch.manual_seed(4107);np.random.seed(4107);model=Hybrid(tasks=TASKS);label='adme_assisted' if assisted else 'scratch'
  prehistory=[]
  if assisted:
   opt=torch.optim.AdamW(list(model.encoder.parameters())+list(model.aux.parameters()),lr=1e-3,weight_decay=1e-3)
   best=float('inf');stale=0;state=None
   for epoch in range(80):
    model.train()
    for _ in range(8):opt.zero_grad();loss=auxloss(model,train_by_task,True);loss.backward();opt.step()
    model.eval()
    with torch.no_grad():vl=float(auxloss(model,val_by_task))
    prehistory.append({'epoch':epoch,'validation_loss':vl})
    if vl<best-1e-5:best=vl;state=copy.deepcopy(model.state_dict());stale=0
    else:stale+=1
    if stale>=12:break
   model.load_state_dict(state)
   # Persist the ADME-only checkpoint before any PK fine tuning.
   torch.save({'model':state,'feature_mean':mean,'feature_std':std,'target_mean':amean,'target_std':astd,'tasks':TASKS},a.output/'adme-pretrained.pt')
   with torch.no_grad():ap=model.auxiliary(ax[aval]).numpy()[np.arange(len(aval)),task[aval]]*astd[task[aval]]+amean[task[aval]]
   av=ad.iloc[aval][['compound_id','task','target','source_id']].copy();av['prediction_log_target']=ap;av.to_csv(a.output/'adme-validation-predictions.csv',index=False)
   write_json(a.output/'adme-pretraining.json',{'history':prehistory,'per_task_log10_mae':av.assign(error=np.abs(av.target-ap)).groupby('task').error.mean().to_dict()})
  opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-3)
  best=float('inf');beststate=None;stale=0;history=[];start=time.monotonic()
  for epoch in range(a.epochs):
   model.train();order=np.random.permutation(train);losses=[]
   for j in np.array_split(order,int(np.ceil(len(order)/256))):
    opt.zero_grad();pred=model(px[j],form[j],ts[j],oral[j],infusion[j])
    loss=(torch.nn.functional.smooth_l1_loss(pred,target[j],reduction='none')*weights[j]).mean()
    if assisted:loss=loss+.1*auxloss(model,train_by_task,True)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()))
   model.eval()
   with torch.no_grad():pv=predict(model,val);vl=float((torch.abs(pv-target[val])*weights[val]).mean())
   history.append({'epoch':epoch,'training_loss':float(np.mean(losses)),'validation_log10_mae':vl,'seconds':time.monotonic()-start})
   if vl<best-1e-5:best=vl;beststate=copy.deepcopy(model.state_dict());bestepoch=epoch;stale=0
   else:stale+=1
   print(label,epoch,'val_log10_mae',round(vl,4),'seconds',round(time.monotonic()-start,1),flush=True)
   if stale>=15 or time.monotonic()-start>a.max_seconds_per_arm:break
  model.load_state_dict(beststate);model.eval()
  with torch.no_grad():pred=10**predict(model,val).numpy()*pk.iloc[val].dose_mg_kg.to_numpy()
  vf=pk.iloc[val].copy();vf['prediction']=pred;vf.to_csv(a.output/f'{label}-validation-predictions.csv',index=False)
  score=evaluate(vf,pred);score.update(arm=label,pk_species='rat',pk_tissue='plasma',best_epoch=bestepoch,
    training_seconds=time.monotonic()-start,history=history,test_evaluated=False,human_organ_target_established=False)
  write_json(a.output/f'{label}-metrics.json',score)
  torch.save({'model':beststate,'feature_mean':mean,'feature_std':std,'species':'rat','tissue':'plasma','forms':FORMS,
    'pk_dataset':ds,'adme_assisted':assisted,'tasks':TASKS,'unit':'ng/mL','architecture':'OSMO two-compartment hybrid v0.2'},a.output/f'{label}.pt')
  results.append({k:v for k,v in score.items() if k!='history'});print('RESULT',label,score['score'],flush=True)
 write_json(a.output/'comparison.json',{'results':results,'test_evaluated':False,'human_organ_target_established':False})
if __name__=='__main__':main()
