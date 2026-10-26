"""Scientific figures, French report, English paper and source archive."""
from base import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zipfile,subprocess,shutil
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def read(n):return json.loads((D/'results'/n).read_text())
def pc(v):return f'{100*v:.2f}%'
def ci(v):return '['+', '.join(f'{x:.5f}' for x in v)+']'
def save(fig,name):
    fig.tight_layout(rect=(0,0,1,.92))
    for ext in ['png','svg','pdf']:fig.savefig(D/f'figures/{name}.{ext}',bbox_inches='tight',dpi=160)
    plt.close(fig)
def figures(c,t,o):
    fig,ax=plt.subplots(1,3,figsize=(11,3.1));names=['Y00','Y10','Y11','Y01']
    for a,(n,q) in zip(ax,c.items()):a.bar(range(4),[q[k]['mean_nll_blank'] for k in names],color=['.5','#b7814d','#36779d','#7aa1a5']);a.set(xticks=range(4),xticklabels=names,title=n,ylabel='Blank-cell NLL (nats)')
    fig.suptitle('Crossed states and tails; lower loss is better');save(fig,'01_crossed_nll')
    fig,ax=plt.subplots(1,2,figsize=(10,3.1))
    for a,key in zip(ax,['R','I']):
        for j,(n,q) in enumerate(c.items()):
            v=q['contrasts'][key];m=v['mean'];lo,hi=v['ci975'];a.errorbar(j,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color='#36779d')
        a.axhline(0,color='.6',lw=1);a.set(xticks=range(3),xticklabels=list(c),ylabel='NLL contrast (nats)',title='R = L10 - L11' if key=='R' else 'I = L10 + L01 - L11 - L00');a.tick_params(axis='x',labelrotation=10)
    fig.suptitle('97.5% paired puzzle-bootstrap intervals; two prespecified co-primary contrasts');save(fig,'02_contrasts')
    a=np.load(D/'results/fresh_crossed.npz');fig,ax=plt.subplots(1,2,figsize=(10,3.1));ax[0].hist(a['cosine_to_negative_delta'],bins=20);ax[0].set(xlabel='cos(tail change, removed residual)',ylabel='New puzzles (count)');ax[1].hist(a['remaining_delta_ratio'],bins=20);ax[1].axvline(1,color='.5',ls=':');ax[1].set(xlabel='Norm(delta + tail change) / norm(delta)',ylabel='New puzzles (count)');fig.suptitle('Latent response on 256 fresh Sudoku puzzles');save(fig,'03_geometry')
    fig,ax=plt.subplots(1,2,figsize=(10,3.1));x=range(1,13)
    for key,label in [('all_positions','All positions'),('final_content_positions','Final non-special')]:
        q=t[key];ax[0].plot(x,100*np.array(q['agreement_by_level']),'o-',label=label);ax[1].plot(range(2,13),[100*v['adoption_then_abandonment']['rate'] if v['adoption_then_abandonment']['rate'] is not None else np.nan for v in q['transitions']],'o-',label=label)
    ax[0].set(xlabel='Generation level',ylabel='Agreement with final raw token (%)',xticks=x);ax[1].set(xlabel='Arrival level of adoption',ylabel='Adoptions later abandoned (%)',xticks=range(2,13))
    for a in ax:a.legend(fontsize=8)
    fig.suptitle('128 LM1B small native trajectories; final-token agreement is not correctness');save(fig,'04_text_generation')
    q=o['reference_nonpad'];fig,ax=plt.subplots(1,2,figsize=(10,3.1));ax[0].plot(x,100*np.array(q['agreement_by_level']),'o-');ax[0].set(xlabel='Codec level',ylabel='Correct reference tokens, non-PAD (%)',xticks=x);ax[1].plot(range(2,13),[100*v['adoption_then_abandonment']['rate'] if v['adoption_then_abandonment']['rate'] is not None else np.nan for v in q['transitions']],'o-');ax[1].set(xlabel='Arrival level of correction',ylabel='Corrections later degraded (%)',xticks=range(2,13));fig.suptitle('LM1B codec oracle: 128 reused validation rows');save(fig,'05_text_oracle')
def archive():
    with zipfile.ZipFile(D/'sources_compensation.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['scripts','sources']:
            for p in (D/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in str(p):z.write(p,str(p.relative_to(D)))
        for n in ['protocol.json','OBJECTIFS_ET_PROTOCOLE.md','README.md','JOURNAL.md','run.sh','artifacts/lm1b-small/config.yaml']:z.write(D/n,n)
        for n in ['scripts/common.py','scripts/solvers.py','protocol.json','results/splits.json','requirements.txt','sources/requirements-lock.txt','mecanismes_revision/scripts/audit.py','mecanismes_revision/ANALYSE_ET_PISTES.md']:z.write(ROOT/n,'parent/'+n)
        for p in (ROOT/'vendor/ncm_mac').rglob('*'):
            if p.is_file() and p.suffix in ['.py','.yaml','.yml','.json','.toml','.txt'] and '__pycache__' not in str(p):z.write(p,'parent/'+str(p.relative_to(ROOT)))
def pdf():
    build=D/'.cache/tex-build';build.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env['TEXMFVAR']=str(D/'.cache/tex-var');env['TEXMFCONFIG']=str(D/'.cache/tex-config');env['TMPDIR']=str(D/'.cache/tmp');Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
    subprocess.run(['/opt/homebrew/bin/pandoc','PAPER.md','-s','--number-sections','-V','colorlinks=true','-o','PAPER.tex'],cwd=D,env=env,check=True)
    with (D/'logs/pdf.log').open('w') as f:
        for _ in range(2):subprocess.run(['/Library/TeX/texbin/pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(build),'PAPER.tex'],cwd=D,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    shutil.copy2(build/'PAPER.pdf',D/'PAPER.pdf')
def main():
    from writing import documents
    c={n:read(n+'_crossed.json') for n in Q['sudoku_sets']};t=read('lm1b_generation.json');o=read('lm1b_oracle.json');figures(c,t,o);documents(c,t,o);pdf();archive();log('compensation_published')
if __name__=='__main__':main()
