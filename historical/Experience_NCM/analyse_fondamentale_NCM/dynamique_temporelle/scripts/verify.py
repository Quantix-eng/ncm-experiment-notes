"""Independent arithmetic checks, immutable-input audit and one-batch causal replay."""
from base import *
from observations import filehash,ratio_ci,change_fits
from runtime import Engine
from interventions import finish
import argparse,zipfile

checks=[]
def check(name,value):
    if not bool(value):raise AssertionError(name)
    checks.append(name)
def close(name,x,y,tol=1e-10):check(name,np.allclose(x,y,rtol=tol,atol=tol,equal_nan=True))
def read(name):return json.loads((D/'results'/name).read_text())
def ci_check(name,d,v):
    q=paired_mean_ci(v);close(name+'.mean',q['mean'],d['mean']);close(name+'.ci95',q['ci95'],d['ci95'])

def prior_integrity():
    hashes=json.loads((D/'sources/previous_manifests.json').read_text())
    for rel,h in hashes.items():
        p=ROOT/rel;check('prior_manifest:'+rel,filehash(p)==h)
        manifest=json.loads(p.read_text());root=p.parent.parent
        for path,digest in manifest.items():check('prior_file:'+str((root/path).relative_to(ROOT)),filehash(root/path)==digest)
    frozen=read('protocol_frozen.json');check('frozen.protocol',frozen['protocol']==Q)
    for key,p in [('protocol_sha256',D/'protocol.json'),('objectives_sha256',D/'OBJECTIFS_ET_PROTOCOLE.md'),('previous_manifest_sha256',ROOT/'results/final_manifest.json'),('split_sha256',ROOT/'results/splits.json')]:check('frozen.'+key,filehash(p)==frozen[key])

def observations():
    data=read('observations.json')
    for name,out in data.items():
        task=name.split('_')[0];a=np.load(ROOT/f'results/{task}.npz');ix=np.array(SPLIT['test']) if name=='hard_test' else np.arange(len(a['reference']));ids=a['decoded'][ix];ref=a['reference'][ix];prefix=a['prefix'][ix];cells=ids[:,:,CELL];good=cells==ref[:,None,CELL]
        for k,p in enumerate(out['profiles']):
            check(f'{name}.{k}.solved',p['solved']==np.all(ids[:,k]==ref,axis=1).sum());close(f'{name}.{k}.cells',p['cell_accuracy'],good[:,k].mean());check(f'{name}.{k}.81_only',p['solved_cells_only']==good[:,k].all(1).sum());check(f'{name}.{k}.sep',p['bad_separator_grids']==np.any(ids[:,k][:,SEP]!=10,axis=1).sum());close(f'{name}.{k}.error_quantiles',p['cell_error_quantiles'],np.quantile((~good[:,k]).sum(1),[0,.1,.25,.5,.75,.9,1]))
            if k:
                for kind,mask in [('all_cells',np.ones_like(good[:,k])),('blank_cells',prefix[:,1:-1][:,CELL]==0)]:
                    b=good[:,k-1];g=good[:,k];changed=cells[:,k-1]!=cells[:,k];values=dict(corrected=int((~b&g&mask).sum()),degraded=int((b&~g&mask).sum()),wrong_to_wrong=int((~b&~g&changed&mask).sum()),changed=int((changed&mask).sum()));check(f'{name}.{k}.{kind}',values==out['transitions'][k-1][kind])
        fixed=~good[:,7]&good[:,8];laterbad=(~good[:,9:]).any(1);return8=(cells[:,9:]==cells[:,7,None,:]).any(1)
        pairs={'corrected9_then_wrong_any':(fixed&laterbad,fixed),'corrected9_then_wrong_final':(fixed&~good[:,-1],fixed),'correct9_then_wrong_any':(good[:,8]&laterbad,good[:,8]),'corrected9_then_back_to_value8_any':(fixed&return8,fixed)}
        for key,(num,den) in pairs.items():
            r=out['reversals'][key];check(name+'.'+key+'.count',r['numerator']==num.sum() and r['denominator']==den.sum());re=ratio_ci(num.sum(1),den.sum(1));close(name+'.'+key+'.rate',r['rate'],re['rate']);close(name+'.'+key+'.ci',r['ci95'],re['ci95'])
        close(name+'.later_change',out['reversals']['fraction_values9_changed_later'],(cells[:,9:]!=cells[:,8,None,:]).any(1).mean());check(name+'.final_agreement',out['reversals']['grids9_identical_to12']==np.all(ids[:,8]==ids[:,-1],axis=1).sum())
        last=np.where(cells[:,1:]!=cells[:,:-1],np.arange(2,13)[None,:,None],1).max(1);check(name+'.last_change',np.array_equal(out['last_observed_change_histogram'],np.bincount(last.ravel(),minlength=13)[1:]))
        solved=np.all(ids==ref[:,None],axis=-1)
        for k,d in out['stop_minus_final'].items():ci_check(name+'.stop'+k,d,solved[:,int(k)-1].astype(float)-solved[:,-1])
        q=good.mean(-1);ci_check(name+'.acceleration',out['secondary_acceleration'],q[:,8]-q[:,7]-.5*((q[:,7]-q[:,6])+(q[:,9]-q[:,8])))
        for j,ex in enumerate(out['counterexamples']):
            row=int(np.where(ix==ex['source_index'])[0][0]);col=ex['cell'];check(name+f'.example{j}',np.array_equal(ex['values8_to12'],cells[row,7:,col]) and fixed[row,col] and laterbad[row,col] and ex['reference']==ref[row,CELL[col]])
        fitted=change_fits(q)
        for axis in fitted:
            check(name+'.knot.'+axis,fitted[axis]['best']['knot']==out['descriptive_fits'][axis]['best']['knot']);close(name+'.knot_preference.'+axis,list(fitted[axis]['bootstrap_preference'].values()),list(out['descriptive_fits'][axis]['bootstrap_preference'].values()))

def saved_interventions():
    a=np.load(D/'results/interventions.npz');report=read('interventions.json');ref=a['reference'];p=a['prefix'];native=a['native_ids'];cache=np.load(ROOT/'results/hard.npz');ix=a['indices'];check('fixed_subset',np.array_equal(ix,np.array(SPLIT['test'])[:128]));check('native.final',np.array_equal(native,cache['decoded'][ix,-1]));check('native.codes',np.array_equal(a['native_codes'],cache['codes'][ix]))
    for name,q in report.items():
        ids=a[name+'_ids'];n=len(ids);truth=ref[:n];base=native[:n];was=np.all(base==truth,1);now=np.all(ids==truth,1)
        actual=evaluate(ids,truth,p[:n]);check(name+'.metrics',actual==q['metrics']);check(name+'.changed',q['changed_grids']==np.any(ids!=base,1).sum());check(name+'.introduced',q['introduced']==(was&~now).sum());check(name+'.rescued',q['rescued']==(~was&now).sum());check(name+'.codes',q['changed_codes']==(a[name+'_codes']!=a['native_codes'][:n]).sum());ci_check(name+'.delta',q['delta_success'],now.astype(float)-was)
        if name.startswith('omit'):
            close(name+'.removed_norm',q['mean_removed_norm'],a[name+'_removed_norm'].mean(),1e-6);close(name+'.relative_norm',q['mean_relative_removed_norm'],(a[name+'_removed_norm']/a[name+'_previous_state_norm']).mean(),1e-6)
    for key in ['ids','codes']:
        check('gamma1.sham.'+key,np.array_equal(a['gamma_1.0_'+key],a['native_'+key]));check('gamma0.omit9.'+key,np.array_equal(a['gamma_0.0_'+key],a['omit_9_'+key]));check('omit12.native_codes.'+key,np.array_equal(a['omit_12_codes'],a['native_codes']))
    r=np.load(D/'results/reader.npz');rr=read('reader.json');check('reader.subset',np.array_equal(r['indices'],ix))
    for gamma,q in rr.items():
        logits=r[gamma+'_logits'];ids=r[gamma+'_ids'];check('reader.argmax.'+gamma,np.array_equal(ids,logits.argmax(-1)));close('reader.accuracy.'+gamma,q['cell_accuracy'],(ids[:,CELL]==ref[:,CELL]).mean());check('reader.solved.'+gamma,q['solved']==np.all(ids==ref,1).sum())
        # Independent double-precision log-sum-exp and margin arithmetic.
        l=logits.astype(float);maximum=l.max(-1);normalizer=maximum+np.log(np.exp(l-maximum[:,:,None]).sum(-1));gold=np.take_along_axis(l,ref[:,:,None],axis=-1)[:,:,0];close('reader.nll.'+gamma,q['nll_cells'],(normalizer-gold)[:,CELL].mean(),1e-6);np.put_along_axis(l,ref[:,:,None],-np.inf,axis=-1);close('reader.margin.'+gamma,q['gold_margin_mean'],(gold-l.max(-1))[:,CELL].mean(),1e-6)
    check('reader.z8',np.array_equal(r['0.0_ids'],cache['decoded'][ix,7]));check('reader.z9',np.array_equal(r['1.0_ids'],cache['decoded'][ix,8]))
    s=read('susceptibility.json');n=Q['susceptibility_n']
    for dose,v in s.items():
        per={}
        for timing in ['before','after']:
            ids=np.stack([a[f'noise_{dose}_{seed}_{timing}_ids'] for seed in Q['susceptibility_seeds']]);change=np.any(ids!=native[None,:n],axis=-1).mean(0);success=np.all(ids==ref[None,:n],axis=-1).mean(0);per[timing]=(change,success);close(f'noise.{dose}.{timing}.change',v[timing]['per_puzzle_change'],change);close(f'noise.{dose}.{timing}.success',v[timing]['per_puzzle_success'],success)
            for seed in Q['susceptibility_seeds']:
                key=f'noise_{dose}_{seed}_';check(key+'identical_norm',np.array_equal(a[key+'before_delta_norm'],a[key+'after_delta_norm']));close(key+timing+'.dose',a[key+timing+'_delta_norm'],float(dose)*a[key+timing+'_z8norm'],1e-6)
        ci_check('noise.'+dose+'.change_ci',v['after_minus_before_change'],per['after'][0]-per['before'][0]);ci_check('noise.'+dose+'.success_ci',v['after_minus_before_success'],per['after'][1]-per['before'][1])

@torch.inference_mode()
def replay():
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');a=np.load(D/'results/interventions.npz');r=np.load(D/'results/reader.npz');ix=a['indices'][:16]
    with Engine(model,a['prefix'][:16]) as e:
        z8,c8=e.first8();native=e.branch(z8,c8);check('replay.native.codes',np.array_equal(native['codes'].numpy(),a['native_codes'][:16]));check('replay.native.ids',np.array_equal(native['final_ids'].numpy(),a['native_ids'][:16]));r9=e.contribution(8,native['codes9']);z9=z8+r9
        for gamma in [0.,.5,1.]:
            logits=model.vqvae.decoder(z8+gamma*r9);check('replay.reader.logits.'+str(gamma),np.array_equal(logits.numpy(),r[str(gamma)+'_logits'][:16]));out=finish(e,z8+gamma*r9,c8+[native['codes9']],9)
            for key in ['ids','codes']:check('replay.gamma.'+str(gamma)+'.'+key,np.array_equal(out[key].numpy(),a['gamma_'+str(gamma)+'_'+key][:16]))
        for level in Q['omit_levels']:
            z=torch.zeros_like(z8);cc=[]
            for k in range(12):z,c,_,_=e.step(k,z,omit_contribution=k==level-1);cc.append(c)
            check(f'replay.omit{level}.codes',np.array_equal(torch.cat(cc,1).numpy(),a[f'omit_{level}_codes'][:16]));check(f'replay.omit{level}.ids',np.array_equal(model.vqvae.decoder(z).argmax(-1).numpy(),a[f'omit_{level}_ids'][:16]))
        dose=Q['susceptibility_strengths'][1];seed=Q['susceptibility_seeds'][0];unit=np.stack([np.random.default_rng(seed+int(j)*100003).normal(size=(89,256)) for j in ix]);unit/=np.linalg.norm(unit.reshape(16,-1),axis=1)[:,None,None];delta=torch.tensor(unit,dtype=torch.float32)*(z8.flatten(1).norm(dim=1)*dose)[:,None,None]
        before=e.branch(z8,c8,delta);after=finish(e,z9+delta,c8+[native['codes9']],9)
        for timing,ids,codes in [('before',before['final_ids'],before['codes']),('after',after['ids'],after['codes'])]:
            check('replay.noise.'+timing+'.ids',np.array_equal(ids.numpy(),a[f'noise_{dose}_{seed}_{timing}_ids'][:16]));check('replay.noise.'+timing+'.codes',np.array_equal(codes.numpy(),a[f'noise_{dose}_{seed}_{timing}_codes'][:16]))

def publication():
    with zipfile.ZipFile(D/'sources_temporelles.zip') as z:
        check('archive.crc',z.testzip() is None)
        for p in (D/'scripts').glob('*.py'):check('archive.current.'+p.name,z.read('scripts/'+p.name)==p.read_bytes())
    for name in ['RAPPORT.md','PAPER.md','PAPER.tex','PAPER.pdf','README.md','OBJECTIFS_ET_PROTOCOLE.md','JOURNAL.md','run.sh']:check('artifact.'+name,(D/name).is_file())
    for i,name in enumerate(['native_stops','corrections_reversals','reader_interpolation','temporal_interventions','susceptibility','descriptive_bends'],1):
        for ext in ['png','svg','pdf']:check('figure.'+name+'.'+ext,(D/f'figures/{i:02}_{name}.{ext}').stat().st_size>1000)
    sys.path.insert(0,str(ROOT/'generalisation_architecturale/.deps'));import fitz
    doc=fitz.open(D/'PAPER.pdf');txt='\n'.join(p.get_text() for p in doc);check('pdf.pages',len(doc)>=5);check('pdf.no_replacement','\ufffd' not in txt);check('pdf.primary_counts','3088' in txt and '12333' in txt);check('pdf.main_result','103/128' in txt and '105/128' in txt)

def main():
    args=argparse.ArgumentParser();args.add_argument('--finalize',action='store_true');options=args.parse_args()
    if (D/'results/final_manifest.json').exists() and not options.finalize:
        for path,h in read('final_manifest.json').items():check('current_file:'+path,filehash(D/path)==h)
    prior_integrity();observations();saved_interventions();replay();publication()
    write(D/'results/verification.json',dict(status='passed',n_checks=len(checks),checks=checks,replay='16 fixed hard test puzzles: native, readout gamma 0/0.5/1, continued gamma 0/0.5/1, omissions 8..12, matched noise 5% seed 106128 before/after. All saved metrics recalculated; four previous manifests audited.',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    if options.finalize:
        paths=[]
        for folder in ['scripts','results','sources','figures']:
            paths += [p for p in (D/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
        paths += [p for p in D.iterdir() if p.is_file()]
        excluded={'results/final_manifest.json','results/verification.json','results/progress.json'};manifest={str(p.relative_to(D)):filehash(p) for p in sorted(set(paths)) if str(p.relative_to(D)) not in excluded}
        write(D/'results/final_manifest.json',manifest)
    log('temporal_verified',checks=len(checks),finalized=options.finalize)

if __name__=='__main__':main()
