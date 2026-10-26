from runtime import *


@torch.inference_mode()
def calibrate():
    check();assert not (D/'results/calibration_frozen.json').exists()
    rows=[r for r in read('data/families.json')['rows'] if r['split']=='development']
    records=[];paths=[]
    for architecture in Q['architectures']:
        for seed in Q['seeds']:
            net=load(architecture,seed);ns=[native(net,r) for r in rows]
            logits=np.zeros((4,30,2,4,8),np.float32)
            native_logits=np.asarray([n['logits'][:2,-1].numpy() for n in ns])
            refs=np.asarray([r['targets'] for r in rows])[:,:2,0]
            for dose,rho in enumerate(Q['candidate_doses']):
                for f,n in enumerate(ns):
                    for a in range(2):
                        h=n['states'][a,2];t=n['tails'][a,4:]
                        for q,d in enumerate(directions(h,f,a,'development')):
                            logits[dose,f,a,q]=decode(net,frozen(h+rho*d,t)).numpy()
            base_correct=native_logits.argmax(-1)==refs
            base_loss=logits_loss(native_logits,refs)
            doses=[]
            for dose,rho in enumerate(Q['candidate_doses']):
                loss=logits_loss(logits[dose],np.broadcast_to(refs[...,None],(30,2,4)))
                correct=logits[dose].argmax(-1)==refs[...,None]
                doses.append(dict(rho=rho,damage_nll=float((loss-base_loss[...,None]).mean()),
                                  accuracy_drop=float(base_correct.mean()-correct.mean()),frozen_accuracy=float(correct.mean())))
            records.append(dict(architecture=architecture,seed=seed,native_first_accuracy=float(base_correct.mean()),
                                 feasibility_85_percent=bool(base_correct.mean()>=.85),doses=doses))
            path=D/f'results/development-{architecture}-{seed}.npz'
            np.savez_compressed(path,frozen_logits=logits,native_logits=native_logits,refs=refs)
            paths.append(path)
            print('Development',architecture,seed,records[-1],flush=True)
    candidates=[]
    for j,rho in enumerate(Q['candidate_doses']):
        stats={}
        for architecture in Q['architectures']:
            values=[r['doses'][j] for r in records if r['architecture']==architecture]
            stats[architecture]=dict(damage_nll=float(np.mean([r['damage_nll'] for r in values])),
                                    accuracy_drop=float(np.mean([r['accuracy_drop'] for r in values])))
        admissible=all(.10<=s['accuracy_drop']<=.40 and s['damage_nll']>=.05 for s in stats.values())
        candidates.append(dict(rho=rho,architectures=stats,admissible=admissible))
    chosen=next((c['rho'] for c in candidates if c['admissible']),3.)
    write('results/calibration.json',dict(status='common_informative_dose_found' if any(c['admissible'] for c in candidates) else 'no_common_informative_dose_fallback',
          selected_rho=chosen,primary_cut=2,all_candidates=candidates,per_model=records,
          scope='Development only; no evaluation outputs read'))
    files=paths+[D/'results/calibration.json',D/'results/training.json',D/'results/protocol_frozen.json']+list((D/'checkpoints').glob('*.pt'))
    write('results/calibration_frozen.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            status='Dose and final weights sealed before evaluation',files={str(p.resolve()):sha(p) for p in files}))
    check()


if __name__=='__main__':calibrate()
