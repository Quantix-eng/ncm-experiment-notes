"""Publish a repair supplement and an integrated paper without rewriting earlier work."""
from repair import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil,subprocess,zipfile,csv,re
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def read(name):return json.loads((D/'results'/name).read_text())
def pc(x):return f'{100*x:.2f}%'
def ci(x):return '['+', '.join(f'{100*v:.2f}' for v in x)+']'
def save(fig,name):
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','svg','pdf']:fig.savefig(D/f'figures/{name}.{ext}',bbox_inches='tight',dpi=180)
    plt.close(fig)

def figures(screen,held,noise,selected):
    names=list(screen);xx=np.arange(len(names));fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    axes[0].bar(xx,[max(screen[n]['raw_variance'],1e-20) for n in names]);axes[0].set(ylabel='Raw pooled inter-solution variance',yscale='log')
    axes[1].bar(xx,[screen[n]['occupancy']['positional_entropy'] for n in names]);axes[1].set(ylabel='Position-conditioned code entropy (nats)',xticks=xx,xticklabels=names)
    axes[1].tick_params(axis='x',labelrotation=35);fig.suptitle('Level-5 validation screen: 128 known hard solutions; reconstruction results reported separately');save(fig,'05_repair_screen')
    fig,axes=plt.subplots(1,2,figsize=(11,3.7))
    for ax,name in zip(axes,['native',selected]):
        counts=np.array(held['hard_test'][name]['occupancy']['counts']);im=ax.imshow(counts/counts.sum(1,keepdims=True),vmin=0,vmax=1,aspect='auto',cmap='Blues')
        ax.set(title=name,xlabel='VQ code identifier',ylabel='Level-5 position',xticks=range(16),yticks=range(9));ax.tick_params(axis='x',labelsize=7)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Fraction of 600 test solutions',fraction=.025,pad=.03)
    fig.suptitle('Native and selected quantization: same codebook, weights and 9 output positions')
    for ext in ['png','svg','pdf']:fig.savefig(D/f'figures/06_repair_codes.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    levels=[float(v) for v in noise];fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in ['native',selected]:axes[0].plot(levels,[noise[str(v)][name]['mean_exact']*100 for v in levels],'o-',label=name)
    vals=[noise[str(v)]['paired_delta'] for v in levels];m=np.array([v['mean'] for v in vals])*100;bounds=np.array([v['ci95'] for v in vals])*100
    axes[1].errorbar(levels,m,yerr=np.stack([m-bounds[:,0],bounds[:,1]-m]),fmt='o-',capsize=4);axes[1].axhline(0,color='gray',lw=.8)
    axes[0].set(ylabel='Exact reconstruction (%)',ylim=(-2,102));axes[0].legend();axes[1].set(ylabel='Selected - native (percentage points)')
    for ax in axes:ax.set(xlabel='Latent noise SD / train encoder RMS',xticks=levels)
    fig.suptitle('128 paired test solutions, 5 seeds per noise level; 95% CI resamples puzzles');save(fig,'07_repair_noise')

def main():
    screen=read('screen.json');selection=read('selection_frozen.json');selected=selection['selected'];held=read('heldout.json');noise=read('noise.json');sym=read('symmetries.json');parity=read('parity.json');align=read('generator_alignment.json');order=read('order_controls.json');comp=read('compensation.json')
    figures(screen,held,noise,selected);h=held['hard_test'][selected];native=held['hard_test']['native'];ex=held['external'][selected]
    supplement=[]
    def add(s):supplement.append(s.strip()+'\n')
    add(r'''# A fixed-size, order-sensitive pooling repair without retraining

## Construction and what it corrects

We replace only the level-5 downsampling operator, preserving nine output positions, the original codebooks, learned upsampling, decoder and all other levels. For each original adaptive window $B_j$, let $t_{ji}$ vary linearly from -1 to +1 and define

$$
u_j^{(\alpha)}=\frac{1}{|B_j|}\sum_{i\in B_j}(1+\alpha t_{ji})r_i
=u_j^{(0)}+\frac{\alpha}{|B_j|}\sum_{i\in B_j}t_{ji}r_i.
$$

The weights sum to one. They are nonnegative for $0\leq\alpha\leq1$; the diagnostic variant $\alpha=2$ permits negative weights. There are no learned parameters or new dimensions. For the additive raw embedding representation, swapping tokens $a,b$ at positions $i,h$ changes the weighted output by

$$
\frac{\alpha(t_{ji}-t_{jh})}{|B_j|}\big[e(b)-e(a)\big].
$$

If the positional weights and token embeddings differ, this breaks the particular within-window permutation invariance, even when histograms are identical. It does not make the map injective, establish semantic sufficiency, or prove a globally better tokenizer. The operation also differs from appending an identical separator: separators already exist in the Sudoku serialization, and an additional fixed token does not help if every window retains identical token counts.

A deterministic input-independent bias is a negative control: $\mathrm{Var}_x[u(x)+b]=\mathrm{Var}_x[u(x)]$ at fixed position. Such a bias may move nearest-code boundaries and alter observed assignments without adding order information. Pure token-embedding max-pooling is itself permutation invariant; positional embeddings and residuals can change its behavior. Neither higher variance nor higher occupancy alone is a valid quality criterion.

## Finite screen and frozen selection

Before experiments, we specify eleven variants: native; a zero-strength weighted sham; weighted strengths 0.05, 0.25, 1 and 2; right-shift mixtures 0.25 and 1, with right-edge replication; mean/max mixtures 0.1 and 1; and a fixed pseudo-random bias with RMS 0.25. All changes are applied at level 5 only. There is no training or codebook refitting.

The first 128 hard validation solutions screen the variants. Candidates must introduce no exact reconstruction failures relative to native and increase position-conditioned level-5 code entropy. We maximize that entropy, break ties by the smaller RMS input change, then lexical name. The sham and bias are excluded from selection. The selected variant is frozen before the new test evaluations. This is a further exploratory extension of an already inspected Sudoku study, not a new untouched confirmatory experiment.
''')
    for name,r in screen.items():
        add(f"**{name}:** {r['exact']}/128 exact reconstructions; raw pooled variance {r['raw_variance']:.5g}; residual-input variance {r['input_variance']:.5g}; {r['occupancy']['codes_used']}/16 codes, positional entropy {r['occupancy']['positional_entropy']:.4f} nats; {r['changed_codes5']} level-5 codes and {r['changed_codes']} total codes changed.")
    add(f"The frozen choice is **{selected}**. It {'passes' if selection['passes_validation_preservation'] else 'does not pass'} the validation preservation criterion. The zero-strength sham reproduces all native codes and reconstructions. In the paired raw-embedding swap control, relative L2 change is {order['native']['swap_relative_l2']:.3g} for native averaging and {order[selected]['swap_relative_l2']:.3g} for the selected operator; this control uses synthetic within-row swaps, not valid new Sudoku tasks.")
    add('![Validation screen separates variance recovery from discrete code entropy. The fixed bias is an input-independent negative control.](figures/05_repair_screen.pdf)')
    add('''## Held-out reconstruction and checkpoint replication

Known solutions, rather than puzzle clues, are encoded. Reconstruction is exact over all 89 tokens including separators. We evaluate native, the selected operator, weighted strength 1 and max strength 1 on 600 hard test and 256 external solutions, deduplicating identical variants. Native and the selected operator are also tested on 600 easy and medium test solutions. The checkpoints have different weights but shared solution families, so these are correlated replications. Bootstrap intervals are paired by solution, conditional on the fixed operator; zero observed discordances give a degenerate interval and do not establish population equivalence.
''')
    for corpus,rs in held.items():
        for name,r in rs.items():
            add(f"**{corpus}, {name}:** {r['exact']}/{r['n']} exact; {r['token_errors']} token errors; {r['occupancy']['codes_used']}/16 level-5 codes; positional entropy {r['occupancy']['positional_entropy']:.4f} nats. Versus native: {r['introduced']} failures introduced, {r['rescued']} repaired; exact-rate difference {100*r['delta_exact']['mean']:.2f} percentage points, paired 95% CI {ci(r['delta_exact']['ci95'])}; {r['changed_codes5']} level-5 codes changed.")
    add('![Level-5 code distributions on hard test solutions. Changed encoder targets do not imply that the frozen generator learned to predict them.](figures/06_repair_codes.pdf)')
    add('''## Robustness: valid transformations and latent corruption

On the first 128 hard test solutions, a fixed digit permutation and transposition provide valid transformed grids. Native and selected encoders reconstruct each transformed reference independently. These tests do not evaluate solving under transformed clues.
''')
    for kind,rs in sym.items():
        for name,r in rs.items():add(f"**{kind}, {name}:** {r['exact']}/{r['n']} exact, {r['introduced']} failures introduced and {r['rescued']} repaired versus native.")
    add('''We also add Gaussian noise to the full pre-quant encoder state before residual quantization, with SD 0.05, 0.2 or 0.5 times the RMS measured only on 256 train solutions. Identical noise draws are used between operators for each source and seed. Five fixed seeds are averaged within each puzzle; 2,000 bootstrap draws resample puzzles. This is a codec latent-noise stress test, not Sudoku difficulty, input corruption robustness, or an end-user generation benchmark. The three noise contrasts are exploratory and not corrected for multiplicity.
''')
    for sigma,r in noise.items():add(f"**Noise {sigma}:** native exact reconstruction {pc(r['native']['mean_exact'])}, {selected} {pc(r[selected]['mean_exact'])}; paired difference {100*r['paired_delta']['mean']:.2f} percentage points, 95% CI {ci(r['paired_delta']['ci95'])}. Secondary token accuracy: native {pc(r['native']['token_accuracy'])}, selected {pc(r[selected]['token_accuracy'])}; these means do not treat individual tokens as independent statistical units.")
    noisy=np.load(D/'results/noise.npz');clean=np.load(D/'results/hard_test.npz');codechanges={}
    for sigma in noise:
        codechanges[sigma]={name:int((noisy[sigma+'_'+name+'_codes']!=clean[name+'_codes'][None,:Q['robustness']['n_puzzles']]).sum()) for name in ['native',selected]}
    write(D/'results/noise_code_changes.json',codechanges)
    if all(r['native']['mean_exact']==1 and r[selected]['mean_exact']==1 for r in noise.values()):
        add(f"Both codecs remain at the reconstruction ceiling at every specified noise amplitude. The test therefore cannot establish a robustness advantage or equivalence beyond these perturbations. Noise nevertheless changes discrete assignments: at SD 0.5, {codechanges['0.5']['native']:,} native codes and {codechanges['0.5'][selected]:,} selected-operator codes differ from their clean counterparts across the five seeds. Code variation does not imply a decoded reconstruction failure.")
    add('![Latent-noise stress test with uncertainty clustered by source solution. Clean reconstruction preservation and noise robustness are distinct claims.](figures/07_repair_noise.pdf)')
    add(f'''## Post-hoc contribution and fine-code compensation control

After observing preserved clean reconstruction, a documented post-hoc amendment tests the first 128 hard test solutions with fixed fine codes. An artificial hybrid replaces only the nine level-5 codes by their modified-encoder values, retaining all other native codes. This combination can be outside the native training distribution and is not a deployable repair.

Native, fully reencoded and hybrid streams reconstruct {comp['native']['exact']}/128, {comp['reencoded']['exact']}/128 and {comp['hybrid']['exact']}/128 solutions exactly. The hybrid introduces {comp['hybrid']['token_errors']} token errors and changes {comp['hybrid']['changed_grids_vs_native']} final grids relative to native. Full reencoding changes {comp['changed_fine_codes']} fine codes after level 5. Thus final reconstruction should not be interpreted from occupancy alone; this control measures whether replacing the code while holding its downstream context fixed behaves like full reencoding.

Decoded after level 5, cell accuracy is {pc(comp['native']['cell_accuracy5'])} natively and {pc(comp['reencoded']['cell_accuracy5'])} after the repair. The paired difference is {100*comp['paired_delta_level5_cell_accuracy']['mean']:.2f} percentage points, CI {ci(comp['paired_delta_level5_cell_accuracy']['ci95'])}. The level-4 decoding is a common pre-intervention control. These are partial reconstructions of known solutions, not predictions from puzzle clues.
''')
    if comp['hybrid']['exact']==128 and comp['hybrid']['token_errors']==0:
        add('On these 128 contexts, changing the fine codes is not required to preserve final argmax reconstruction. The extra occupancy is therefore not evidence of a necessary new contribution to the final decoded answer. The poorer partial level-5 accuracy further limits an interpretation as better intermediate task representation. Probability changes, redundancy outside these contexts and other decoders remain untested.')
    add(f'''## Frozen-generator compatibility and interpretation

Patching the actual quantizer method agrees with the explicit modified codec on eight test solutions: {parity['patched_encode_codes']}. The modified method is called {parity['encode_pool_calls']} time(s) during the encoded batch, but **{parity['generation_pool_calls']} times during native generation** on the corresponding puzzles. Generator codes, logits and answers remain bit-identical: {parity['generator_codes_identical']}, {parity['generator_logits_identical']}, {parity['generator_ids_identical']}. The tested tokenizer state dictionaries before and after are identical.

This equality has a structural explanation: the released Sudoku inference path embeds the puzzle and generates codes without encoding its unknown solution through the modified pooling operation. Therefore changing only this encoder operation cannot improve that inference path. Unchanged generated answers are not evidence that the new codes were learned or that the generator became more robust.

On the 600 test solutions, agreement between native generated level-5 codes and native solution encodings is {pc(align['native']['level5_match'])}, compared with {pc(align[selected]['level5_match'])} for the modified encoder targets. This is a changed-target diagnostic, not a measure of semantic correctness or a new solving score.

The proposed operator is thus an explicit remedy for the raw averaging invariance. Whether it is also a useful encoder retrofit depends on the held-out reconstruction and stress-test results reported above. Even perfect clean reconstruction would not establish a free improvement to the complete NCM: later quantization levels can compensate, the generator still uses its old target distribution, and the added computation has not been benchmarked. A future jointly trained system could initialize a positive position-weighted filter or learn such weights, while controlling codebook assignments, reconstruction and downstream quality. That recommendation remains an architectural proposal, not a training result from this experiment.
''')
    add('''## Repair evidence and reproduction

The repair protocol and selection snapshot precede held-out repair evaluations. All modified encoder inputs, codes and reconstructions are retained by example. `repair.py` supplies the fixed-size operator and a reversible patch context manager; `experiments.py` executes the finite screen, held-out evaluations and generator controls. `verify.py` recomputes statistics and paired intervals, checks previous campaign manifests, and replays the modified native path. `./run.sh` reproduces this repair extension; the earlier cross-domain campaign has its separate runner at `../generalisation_architecturale/run.sh`.
''')
    extension='\n'.join(supplement);(D/'SUPPLEMENT.md').write_text(extension)
    previous=ROOT/'generalisation_architecturale';paper=(D/'sources/PAPER_pre_repair.md').read_text()
    paper=paper.replace('Architectural Invariance and Bypassable Levels in Next-Concept Models: A Cross-Domain Audit','Architectural Invariance and Order-Sensitive Pooling in Next-Concept Models')
    paper=paper.replace('geometry: margin=0.85in','geometry: margin=0.8in')
    paper=re.sub(r'\nThe earlier level-9 revision study[^\n]*\n?', '\n', paper)
    paper=paper.replace('# Related work and interpretation',extension+'\n# Related work and interpretation',1)
    abstract_add=f" We additionally test a parameter-free position-weighted pooling repair at fixed output size. The validation-selected {selected} variant reconstructs {h['exact']}/600 held-out hard solutions exactly, versus {native['exact']}/600 natively, while using {h['occupancy']['codes_used']} level-5 codes. Its encoder reconstruction and latent-noise results are reported separately from the frozen generator, which never calls the modified operation during tested inference."
    paper=paper.replace('\n# Introduction and scope',abstract_add+'\n\n# Introduction and scope',1)
    paper=paper.replace('Run `./run.sh` from the study folder to reproduce the extension sequentially, or `../.venv/bin/python scripts/verify.py` with the cache variables in that runner to verify existing evidence. The verifier checks manifests, recomputes metrics and bootstrap intervals from saved predictions, validates selection, replays native text encoding and verifies the PDF.',
      'The previous cross-domain extension is reproduced with `../generalisation_architecturale/run.sh`; its verification evidence remains preserved. Run `./run.sh` here to reproduce the added repair experiments and integrated article. The local verifier recomputes repair metrics and paired intervals, validates repair selection, checks the earlier campaign manifests, replays the modified native Sudoku encoder and checks the PDF. It does not rerun the earlier text models.')
    (D/'PAPER.md').write_text(paper)
    for path in (previous/'figures').iterdir():shutil.copyfile(path,D/'figures'/path.name)
    for name in ['text_levels.csv','sudoku_levels.csv']:shutil.copyfile(previous/'results'/name,D/'results'/name)
    description=f'''# Correction du pooling sans réentraînement

L'[article intégré](PAPER.md), son [PDF](PAPER.pdf) et le [supplément expérimental anglais](SUPPLEMENT.md) ajoutent une proposition explicite et ses tests. Les versions antérieures restent intactes.

## Ce qui est corrigé

La pondération dépendant de la position casse l'invariance de la moyenne aux permutations dans une fenêtre, à taille de sortie inchangée. Aucun poids, codebook ni décodeur n'est réentraîné. Le cas alpha=1 utilise des poids non négatifs de somme 1 ; alpha=2 est seulement un diagnostic avec poids éventuellement négatifs. Un biais fixe ne peut pas ajouter de variance entre entrées ; il est testé comme contrôle négatif.

La sélection sur 128 solutions validation retient **{selected}** : {screen[selected]['exact']}/128 reconstructions exactes et {screen[selected]['occupancy']['codes_used']}/16 codes du niveau 5. Le sham reproduit les codes natifs.

## Résultats réservés

'''
    for corpus,rs in held.items():
        n=rs['native'];r=rs[selected];description+=f"- **{corpus}** : natif {n['exact']}/{n['n']}, modification {r['exact']}/{r['n']} ; {r['introduced']} échecs introduits, {r['rescued']} corrigés ; {r['occupancy']['codes_used']}/16 codes, entropie conditionnelle {r['occupancy']['positional_entropy']:.4f} nat.\n"
    description+='\nLes reconstructions utilisent des solutions connues, pas seulement les indices. Un résultat propre identique ne prouve pas une équivalence populationnelle ni une amélioration de résolution. Les réplications easy/medium partagent les familles de solutions.\n\n## Robustesse mesurée\n\n'
    for kind,rs in sym.items():description+=f"- **{kind}**, 128 solutions transformées : natif {rs['native']['exact']}/128, modification {rs[selected]['exact']}/128.\n"
    for sigma,r in noise.items():description+=f"- **Bruit latent sigma={sigma} × RMS train**, cinq graines sur 128 solutions : natif {pc(r['native']['mean_exact'])}, modification {pc(r[selected]['mean_exact'])} ; variation {100*r['paired_delta']['mean']:.2f} points, IC 95 % {ci(r['paired_delta']['ci95'])}.\n"
    description+=f'''
Ce stress-test porte sur le codec sous bruit latent. Il ne mesure ni résistance à des indices faux ni robustesse du générateur, et ses trois contrastes sont exploratoires.

Les deux codecs restent au plafond sur les trois amplitudes : aucun avantage de robustesse n'est démontré. Le bruit agit pourtant sur les codes : à sigma=0,5, {codechanges['0.5']['native']} codes natifs et {codechanges['0.5'][selected]} codes modifiés diffèrent de leurs versions propres, sur l'ensemble des cinq graines.

Un contrôle post-hoc remplace seulement les neuf codes du niveau 5 en gardant les autres codes natifs : {comp['hybrid']['exact']}/128 reconstructions exactes, contre {comp['reencoded']['exact']}/128 lorsque toute la solution est réencodée. Les niveaux fins changent alors de {comp['changed_fine_codes']} codes. La lecture partielle après niveau 5 atteint {pc(comp['reencoded']['cell_accuracy5'])} de cases exactes contre {pc(comp['native']['cell_accuracy5'])} nativement ; ce n'est pas une résolution à partir des indices. Le contrôle et sa motivation sont dans [AMENDEMENT_COMPENSATION.md](AMENDEMENT_COMPENSATION.md).

## La limite de la « solution gratuite »

Le pooling corrigé est appelé pendant encode, mais **jamais pendant les huit générations natives contrôlées**. Les codes, logits et réponses générés restent identiques. Le patch répare donc un opérateur d'encodage, sans modifier ce chemin de résolution. Le taux d'accord des codes générés du niveau 5 avec les cibles d'encodage passe de {pc(align['native']['level5_match'])} à {pc(align[selected]['level5_match'])}. Le générateur gelé n'a pas appris ces nouvelles cibles.

Il faut donc distinguer correction mathématique, reconstruction propre, robustesse du codec et amélioration du NCM complet. Nous fournissons les quatre contrôles sans déduire les derniers des premiers. Un futur apprentissage conjoint de la pondération et des cibles est une piste motivée, pas un résultat accompli ici.

## Sources et vérification

`sources_correction.zip` contient les scripts utilisés et les sources des modèles. Les tableaux détaillés et les prédictions par solution sont dans `results/`. Les figures sont en PNG/SVG/PDF. Le [protocole](OBJECTIFS_ET_PROTOCOLE.md), le [journal](JOURNAL.md), `results/verification.json` et `results/final_manifest.json` décrivent la méthode et les contrôles passés. Depuis ce dossier : `./run.sh` pour reproduire la correction ; le runner de la campagne texte reste dans le dossier précédent.
'''
    (D/'RAPPORT.md').write_text(description)
    with (D/'results/repair_summary.csv').open('w') as out:
        w=csv.writer(out);w.writerow(['corpus','variant','n','exact','token_errors','codes_used','positional_entropy','raw_variance','input_variance'])
        for corpus,rs in [('validation',screen),*held.items()]:
            for name,r in rs.items():w.writerow([corpus,name,r['n'],r['exact'],r['token_errors'],r['occupancy']['codes_used'],r['occupancy']['positional_entropy'],r['raw_variance'],r['input_variance']])
    env=os.environ.copy()
    for key,name in [('TMPDIR','tmp'),('TEXMFVAR','tex-var'),('TEXMFCONFIG','tex-config')]:
        path=D/'.cache'/name;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    build=D/'.cache/tex-build';build.mkdir(exist_ok=True)
    with (D/'logs/paper_build.log').open('w') as out:
        subprocess.run(['pandoc','PAPER.md','--standalone','--number-sections','-V','colorlinks=true','-o','PAPER.tex'],cwd=D,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
        for _ in range(2):subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(build),'PAPER.tex'],cwd=D,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
    shutil.copyfile(build/'PAPER.pdf',D/'PAPER.pdf');archive_sources();log('publication_complete',selected=selected,words=len(paper.split()))

def archive_sources():
    with zipfile.ZipFile(D/'sources_correction.zip','w',zipfile.ZIP_DEFLATED) as z:
        paths=list((D/'scripts').glob('*.py'))+list(D.glob('*.md'))+[D/'PAPER.tex',D/'protocol.json',D/'run.sh']+list((D/'sources').glob('*'))
        for p in paths:
            if p.is_file():z.write(p,str(p.relative_to(ROOT)))
        for name in ['scripts/common.py','protocol.json','results/splits.json','requirements.txt']:
            z.write(ROOT/name,name)
        for p in (ROOT/'vendor/ncm_mac').rglob('*'):
            if p.is_file() and '__pycache__' not in str(p):z.write(p,str(p.relative_to(ROOT)))
        for p in (ROOT/'artifacts/sudoku').glob('*/config.yaml'):z.write(p,str(p.relative_to(ROOT)))
        for name in ['sources_utilisees.zip','directions_mecanistes/sources_directions_mecanistes.zip','generalisation_architecturale/sources_generalisation.zip']:
            z.write(ROOT/name,name)

if __name__=='__main__':main()
