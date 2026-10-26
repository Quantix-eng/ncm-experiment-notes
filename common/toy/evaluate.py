from runtime import *


def check_calibration():
    check()
    for p,h in read('results/calibration_frozen.json')['files'].items():assert sha(p)==h,p


@torch.inference_mode()
def evaluate():
    check_calibration();assert not (D/'results/evaluation').exists(),'Refusing to replace evaluation'
    target=D/'results/evaluation';target.mkdir()
    rows=[r for r in read('data/families.json')['rows'] if r['split']=='evaluation']
    rho=read('results/calibration.json')['selected_rho']
    start=time.monotonic()
    for architecture in Q['architectures']:
        for seed in Q['seeds']:
            net=load(architecture,seed);ns=[native(net,r) for r in rows]
            np.savez_compressed(target/f'{architecture}-{seed}-native.npz',
                **{key:np.asarray([n[key].numpy() for n in ns]) for key in ns[0]},
                refs=np.asarray([r['targets'] for r in rows]))
            for cut in Q['cuts']:
                cs=[[crossed(net,n,f,a,cut,rho,'evaluation') for a in range(2)] for f,n in enumerate(ns)]
                np.savez_compressed(target/f'{architecture}-{seed}-cut{cut}-C.npz',
                    **{key:np.asarray([[v[key].numpy() for v in source] for source in cs]) for key in cs[0][0]})
                es=[[[exchange(net,n,cut,comparison,track) for comparison in range(2)]
                       for track in ('retained','whole')] for n in ns]
                np.savez_compressed(target/f'{architecture}-{seed}-cut{cut}-E.npz',
                    **{key:np.asarray([[[v[key].numpy() for v in track] for track in source] for source in es]) for key in es[0][0][0]})
                print('Evaluation collected',architecture,seed,'cut',cut,flush=True)
    check_calibration()
    write('results/execution.json',dict(status='complete',models=6,evaluation_families=40,cuts=Q['cuts'],
             elapsed_seconds=time.monotonic()-start,calibration_freeze_sha256=sha(D/'results/calibration_frozen.json')))


if __name__=='__main__':evaluate()
