"""Produce a self-contained scientific narrative and exportable static figures."""
from common import *
import csv, zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from textwrap import dedent

FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})

def read(name): return json.loads((ROOT/f'results/{name}.json').read_text())
def pct(v): return f'{100*v:.2f} %'
def f(v): return 'indéfini (une seule classe)' if v is None else f'{v:.3f}'
def interval(v,percent=False): return '['+', '.join(pct(x) if percent else f'{x:.3f}' for x in v)+']'
def save(fig,name):
    fig.tight_layout(rect=(0,.16,1,.94) if name=='03_probes' else (0,0,1,.95)); fig.savefig(FIG/(name+'.png'),bbox_inches='tight'); fig.savefig(FIG/(name+'.svg'),bbox_inches='tight'); plt.close(fig)

def main():
    tok=read('tokenizer'); probes=read('probes'); sym=read('symmetry'); stability=read('stability'); cross=read('cross_models')
    causal=read('causal'); ext=read('external'); unique=read('official_uniqueness'); fun=read('functional_substitutions'); naked=read('naked_single')
    trans=read('transfer_confidence'); repl=read('probe_replications'); rep={d:read('reproduction_'+d) for d in ['easy','medium','hard']}
    upstream=read('upstream_parity'); k=P['primary_level']-1; primary=probes['primary']; levels=np.arange(1,13)
    headline=('Le gain conditionnel est positif dans cette partition et son intervalle exclut zéro.' if primary['delta_ci95'][0]>0 else
       ('La combinaison fait moins bien que la baseline dans cette partition, avec un intervalle entièrement négatif.' if primary['delta_ci95'][1]<0 else
        "L'intervalle recouvre zéro : cette expérience n'établit pas une valeur ajoutée des états au-delà de la baseline."))
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for d in rep:
        ax[0].plot(levels,[x['utilization']*100 for x in tok[d]['encoded']],marker='.',label=d)
        ax[1].plot(levels,[x['cell_accuracy']*100 for x in tok[d]['reconstruction']],marker='.',label=d)
    ax[0].set(xlabel='Niveau du tokenizer',ylabel='Entrées observées du codebook (%)',title='Utilisation sur les solutions officielles',ylim=(0,105))
    ax[1].set(xlabel='Niveau du tokenizer',ylabel='Cases reconstruites correctement (%)',title='Reconstruction cumulative (solution connue)')
    for a in ax: a.legend(); a.grid(alpha=.2)
    fig.suptitle('Source : résultats locaux, 2 000 solutions par checkpoint, 24 octobre 2026'); save(fig,'01_tokenizer')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for d in rep:
        ax[0].plot(levels,[x['solved_rate']*100 for x in tok[d]['generation']],marker='.',label=d)
        ax[1].plot(levels,[x['cell_accuracy']*100 for x in tok[d]['generation']],marker='.',label=d)
    ax[0].set(xlabel='Niveau de génération',ylabel='Grilles exactes (%)',title='Résolution : réponses intermédiaires')
    ax[1].set(xlabel='Niveau de génération',ylabel='Cases correctes (%)',title='Résolution : précision des cases')
    for a in ax:a.legend();a.grid(alpha=.2)
    fig.suptitle('Source : résultats locaux, puzzles officiels, 2 000 par difficulté'); save(fig,'02_generation')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    names={'difficulty':'Difficulté seule','baseline':'Sorties + difficulté','hidden':'États seuls','combined':'Sorties + états','shuffled_hidden':'Labels mélangés'}
    for name,label in names.items():
        ax[0].plot(levels,[l['models'][name]['auc'] for l in probes['levels']],marker='.',label=label)
        ax[1].plot(levels,[l['models'][name]['brier'] for l in probes['levels']],marker='.',label=label)
    for a in ax:a.axvline(6,color='black',ls=':',lw=1);a.grid(alpha=.2);a.set_xlabel('Niveau observé (avant toute observation future)')
    ax[0].axhline(.5,color='gray',ls='--');ax[0].set(ylabel='AUROC de la réussite finale',title='Discrimination sur 600 puzzles hard test'); handles,legend_labels=ax[0].get_legend_handles_labels();fig.legend(handles,legend_labels,loc='lower center',ncol=3,fontsize=8)
    ax[1].set(ylabel='Score de Brier (plus bas = meilleur)',title='Calibration et discrimination conjointes')
    fig.suptitle('Source : sondes locales ; niveau 6 principal, autres niveaux exploratoires');save(fig,'03_probes')
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for name in ['prior','baseline','hidden','combined']:
        m=probes['levels'][k]['models'][name]; rr=m['reliability']; curve=m['risk_coverage']
        ax[0].plot([v['confidence'] for v in rr],[v['accuracy'] for v in rr],'.-',label=name)
        ax[1].plot([v['coverage'] for v in curve],[v['risk'] for v in curve],label=name)
    ax[0].plot([0,1],[0,1],'k--',lw=1);ax[0].set(xlabel='Probabilité prédite moyenne',ylabel='Fréquence de réussite',title='Fiabilité, 10 intervalles fixes');ax[0].legend()
    ax[1].set(xlabel='Couverture (fraction des puzzles retenus)',ylabel='Risque (fraction d’échecs retenus)',title='Sélection par confiance décroissante');ax[1].legend()
    fig.suptitle('Source : niveau 6, hard test n=600 ; groupes de scores égaux conservés');save(fig,'04_calibration')
    fig,ax=plt.subplots(figsize=(6,4)); ds=list(rep); matrix=np.array([[cross[f'{m}_on_{d}']['solved_rate'] for d in ds] for m in ds])*100
    im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues'); ax.set(xticks=range(3),xticklabels=ds,yticks=range(3),yticklabels=ds,xlabel='Corpus évalué (256 puzzles)',ylabel='Checkpoint',title='Résolution exacte : modèle × difficulté (%)')
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]>60 else 'black')
    fig.colorbar(im,ax=ax,label='Grilles exactes (%)');save(fig,'05_cross_models')
    fig,ax=plt.subplots(figsize=(9,4)); conditions=causal['conditions']; labels=list(conditions); delta=np.array([conditions[n]['delta_solved'] for n in labels])*100
    bounds=np.array([conditions[n]['delta_ci95'] for n in labels])*100
    ax.errorbar(range(len(labels)),delta,yerr=np.stack([delta-bounds[:,0],bounds[:,1]-delta]),fmt='o',capsize=3,color='black')
    ax.axhline(0,color='gray',lw=1);ax.set(xticks=range(len(labels)),xticklabels=labels,ylabel='Variation de réussite (points de %)',xlabel='Intervention au niveau 6, couche 8',title='Ablation de confiance et directions aléatoires à norme appariée')
    fig.suptitle('Source : hard test n=600 ; IC bootstrap apparié par puzzle, 2 000 tirages');save(fig,'06_causal')
    fig,ax=plt.subplots(1,2,figsize=(11,4)); mat=probes['maturity']
    ax[0].plot(levels,[v['linear_blank_accuracy']*100 for v in mat],'.-',label='Sonde états, projection fixe 128D')
    ax[0].plot(levels,[v['puzzle_only_linear_blank_accuracy']*100 for v in mat],'--',label='Sonde indices seuls, projection 128D')
    ax[0].set(xlabel='Niveau de génération',ylabel='Cases initialement vides correctes (%)',title='Décodabilité de la solution (test uniquement)');handles,legend_labels=ax[0].get_legend_handles_labels();fig.legend(handles,legend_labels,loc='lower center',ncol=3,fontsize=8)
    transitions=tok['hard']['transitions']; ax[1].plot(range(2,13),[t['incorrect_to_correct'] for t in transitions],'.-',label='Incorrect → correct')
    ax[1].plot(range(2,13),[t['correct_to_incorrect'] for t in transitions],'.-',label='Correct → incorrect')
    ax[1].set(xlabel='Niveau atteint',ylabel='Nombre de cases (sur 2 000 × 81)',title='Révisions du décodage hard');ax[1].legend()
    fig.suptitle('Source : résultats locaux ; progression descriptive, pas test de transition de phase');save(fig,'07_hierarchy')
    # Flat CSVs allow inspection without Python, while complete arrays remain in NPZ.
    with (ROOT/'results/summary.csv').open('w') as out:
        w=csv.writer(out);w.writerow(['checkpoint','n','solved','solved_rate','cell_accuracy','tokenizer_mean_utilization'])
        for d in rep:
            m=rep[d]['metrics'];w.writerow([d,m['n'],m['solved'],m['solved_rate'],m['cell_accuracy'],np.mean([v['utilization'] for v in tok[d]['encoded']])])
    with (ROOT/'results/probes.csv').open('w') as out:
        w=csv.writer(out);w.writerow(['level','probe','n','failures','auc','brier','ece10','C'])
        for l in probes['levels']:
            for name,m in l['models'].items():w.writerow([l['level'],name,m['n'],m['failures'],m['auc'],m['brier'],m['ece10'],m.get('fit',{}).get('C','')])
    with (ROOT/'results/predictions.csv').open('w') as out:
        w=csv.writer(out);w.writerow(['checkpoint','index','exact','prediction_tokens','reference_tokens'])
        for d in rep:
            a=np.load(ROOT/f'results/{d}.npz')
            for i,(pred,ref) in enumerate(zip(a['decoded'][:,-1],a['reference'])):w.writerow([d,i,int(np.array_equal(pred,ref)),' '.join(map(str,pred)),' '.join(map(str,ref))])
    text=[]
    def add(s):text.append('\n'.join(line[4:] if line.startswith('    ') else line for line in s.strip('\n').splitlines()).strip()+'\n')
    add(f'''# Analyse fondamentale des checkpoints NCM Sudoku

    Rapport expérimental — 24 octobre 2026. Analyse locale CPU FP32. Périmètre : trois checkpoints publiés Sudoku, avec analyse approfondie de hard. Les modèles de texte ne sont pas évalués.

    ## Résultat principal

    Au niveau 6, sur {probes['levels'][k]['models']['baseline']['n']} puzzles hard réservés au test, la baseline obtient une AUROC de **{f(probes['levels'][k]['models']['baseline']['auc'])}**, contre **{f(probes['levels'][k]['models']['combined']['auc'])}** après ajout des états internes. Le gain est **{primary['delta']:.4f}**, IC 95 % apparié **{interval(primary['delta_ci95'])}**. {headline}

    Ce contraste évalue ces sondes linéaires et leur procédure de régularisation ; une absence de gain n'établit pas l'absence d'information dans les états. Il s'agit de la prédiction externe de la réussite d'un modèle gelé. Aucun de ces résultats ne démontre que NCM évalue lui-même sa fiabilité ou utilise une évaluation pour contrôler sa génération.

    ## 1. Questions, méthode et indépendance

    Le protocole révisé couvre les 13 items de la TODO : audit du tokenizer ; stabilité ; symétries ; généralisation et difficulté ; discrimination/calibration ; sondes de contraintes, ablations et progression de la représentation. Le protocole original gelé est conservé dans [sources/TODO_protocol_v1.md](sources/TODO_protocol_v1.md), avec ses paramètres dans [protocol.json](protocol.json), le gel horodaté dans [results/protocol_frozen.json](results/protocol_frozen.json) et les amendements dans [JOURNAL.md](JOURNAL.md).

    L'audit des state_dict confirme que les trois tokenizers et les trois générateurs ont des poids distincts ; leurs analyses ne sont donc pas mutualisées sur la seule base de leurs architectures identiques. Les entrées, poids, sources et environnements sont copiés ou installés dans ce dossier. Aucun résultat antérieur du projet n'est utilisé comme résultat de cette campagne. Le runner CPU provient de la copie locale ncm_mac ; le dépôt amont intact est aussi conservé et exécuté pour le contrôle de parité. Les dépendances système préexistantes sont accessibles en lecture à l'environnement local ; les installations et caches de cette campagne sont locaux.

    Les corpus des trois difficultés partagent les mêmes 2 000 familles de solutions (confirmé par audit croisé), avec des puzzles différents. Les réplications entre difficultés sont donc corrélées ; elles ne sont pas des réplications indépendantes de familles de solutions. Les 2 000 puzzles de chaque difficulté ont été validés quant au format, aux indices et à la validité des références. Chaque corpus possède 2 000 puzzles et 2 000 solutions distincts, même après normalisation des chiffres et les huit symétries du carré testées. Cela n'exclut ni toutes les isomorphies Sudoku ni une présence dans l'entraînement.

    **Partition hard** : 1 000 train, 400 validation, 600 test ; graine {P['seed']}. Les groupes de puzzles équivalents sous les transformations testées restent ensemble. La standardisation et les poids sont ajustés sur train seulement ; C est choisi dans {P['logistic_C']} par log-loss de validation, sans réentraînement sur le test. Les variantes géométriques utilisent 128 puzzles de test, sans entraîner de nouvelle sonde. Les contrôles de stabilité et le croisement modèle/corpus sont descriptifs et n'entraînent aucune sonde.

    **États** : sortie de la couche Transformer 8 avant la normalisation finale ; moyenne et écart-type population sur les positions du niveau courant, soit 1 024 valeurs. Au niveau 1, l'écart-type est nul car il n'y a qu'une position. Les états sont extraits pendant une vraie génération à partir du puzzle seul. Les niveaux de génération, de longueurs 1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67 et 89, sont distincts des huit couches Transformer.

    **Baseline** : 6 proxies de difficulté issus du puzzle, 8 statistiques des distributions de codes et du décodeur, 5 mesures de validité/indices/conflits et 1 taux de changement depuis le décodage précédent. Ces 20 valeurs sont calculées sans solution de référence. Les distributions du décodeur au niveau courant nécessitent un décodage intermédiaire supplémentaire ; il s'agit d'une baseline d'observation post-hoc, pas d'une évaluation du coût d'un système déployé. Les sondes logistiques prédisent la réussite exacte finale. Les analyses par niveau sont exploratoires, à l'exception du contraste annoncé au niveau 6.

    **Incertitude** : Wilson pour les taux de réussite ; bootstrap percentile apparié, {P['bootstrap']} tirages par puzzle, pour le contraste principal et les interventions. Les sondes ne sont pas réentraînées dans le bootstrap : les IC sont conditionnels à ce découpage et à ces ajustements. Pas de correction pour comparaisons multiples exploratoires ; aucune découverte de circuit n'est affirmée sur ces seules comparaisons.

    ## 2. Reproduction et fidélité du runner
    ''')
    for d in rep:
        m=rep[d]['metrics']; up=upstream[d]
        add(f"- **{d}** : {m['solved']}/{m['n']} grilles exactes, soit {pct(m['solved_rate'])}, IC Wilson {interval(m['solved_wilson95'],True)} ; cases correctes {pct(m['cell_accuracy'])} ; validité avec indices {pct(m['valid_and_clues_rate'])}. Parité avec le code amont intact sur {up['n']} puzzles : codes={up['codes_equal']}, réponses={up['outputs_equal']}, écart maximal de logits={up['logits_max_abs']:.3g}.")
    add('''L'article rapporte 99,5 %, 96,5 % et 86,1 % de grilles résolues pour easy, medium et hard. Les résultats ci-dessus concernent les fichiers publiés effectivement disponibles et une comparaison de séquences complète, séparateurs compris. Un petit écart avec le tableau de l'article n'est pas à lui seul une preuve d'erreur : version des poids, protocole d'évaluation et arrondis sont des explications à vérifier. Les contrôles de parité ne portent que sur 16 puzzles par checkpoint ; ils ne prouvent pas l'identité numérique pour toutes les entrées. [Article, §4.4](https://arxiv.org/html/2610.14162v1#S4.SS4)

    L'instrumentation vérifie à chaque batch l'alignement argmax/logits/codes, le nettoyage du cache et des hooks, l'absence d'alias entre accumulateurs intermédiaires, leur égalité à une accumulation explicite et l'égalité du dernier décodage avec la sortie normale. L'incident de sérialisation ListConfig et sa récupération depuis les NPZ intacts sont documentés dans JOURNAL.md ; les contrôles de parité ont été rejoués. Le code amont brut présente un aliasing dans idx_to_zhats ; la copie CPU utilise clone() pour conserver un véritable historique. Ce correctif concerne l'observation des historiques ; le calcul final est vérifié séparément contre l'amont.

    ![Progression des décodages](figures/02_generation.png)

    ## 3. Unicité et définition de la réussite
    ''')
    for d,u in unique.items():add(f"- **{d}** : {u['counts'][0]} puzzle sans solution, {u['counts'][1]} à solution unique, {u['counts'][2]} avec au moins deux solutions selon Algorithm X (comptage plafonné à 2).")
    add('''La métrique principale reste la fidélité stricte à la référence, et la validité Sudoku avec respect des indices est mesurée séparément. Une solution valide alternative doit être interprétée comme telle ; elle n'est pas assimilable à une violation des contraintes. L'unicité des corpus officiels n'était pas supposée.

    ## 4. Audit du tokenizer : utilisation, complémentarité et redondance — items 1–3

    Cet audit encode les solutions connues. Il mesure la capacité de représentation et de reconstruction ; il ne mesure pas une résolution à partir des indices.
    ''')
    for d in rep:
        enc=tok[d]['encoded']; rec=tok[d]['reconstruction']; gen=tok[d]['generated']; last=rec[-1]
        add(f"- **{d}** : utilisation moyenne encodée {pct(np.mean([r['utilization'] for r in enc]))}, générée {pct(np.mean([r['utilization'] for r in gen]))} ; entrées non observées dans les codebooks encodés {sum(len(r['unobserved']) for r in enc)}/192. Reconstruction finale : {pct(last['token_accuracy'])} des tokens, {pct(last['exact_rate'])} des grilles. Perplexité marginale par niveau : {', '.join(f'{r["perplexity"]:.2f}' for r in enc)}.")
    reconstruction_errors=read('tokenizer_errors')
    for d,err in reconstruction_errors.items():
        add(f"Audit exact des erreurs de reconstruction **{d}** : {err['token_errors']} token(s) erroné(s) sur {err['total_tokens']}, touchant {err['grid_errors']} grille(s) sur {err['total_grids']}. Les cas non nuls ont été rejoués dans le tokenizer amont intact avec codes et sorties identiques ; détails dans results/tokenizer_errors.json. L'arrondi du taux token à deux décimales peut masquer une erreur isolée ; il ne faut pas le lire comme une garantie d'exactitude de toutes les grilles.")
    for d in rep:
        constant=[r for r in tok[d]['encoded'] if r['entropy_nats']<1e-10]
        sparse=[r for r in tok[d]['encoded'] if r['utilization']<=.125]
        if sparse:
            add(f"**Niveaux peu diversifiés ({d})** : "+', '.join(f"niveau {r['level']}, {len(np.flatnonzero(r['counts']))}/16 codes observés, H={abs(r['entropy_nats']):.4f} nat" for r in sparse)+'.')
        if constant:
            add("Un niveau entièrement constant n'encode aucune variation entre les solutions observées ; son résidu appris peut cependant agir comme un décalage commun utile au décodeur. Ce constat ne démontre ni inutilité causale du niveau ni effondrement global du tokenizer. Il motive un test dédié de suppression ou de remplacement de ce niveau, distinct de l'ablation de confiance réalisée ici.")
    add('''Les histogrammes, entropies en nats, perplexités, indices non observés et couvertures à 16/64/256/1 000/2 000 solutions sont disponibles dans results/tokenizer.json. L'entropie conditionnelle à la position distingue la variabilité entre puzzles d'une simple hétérogénéité entre positions. Les estimations restent empiriques ; l'absence d'un code dans ce corpus n'établit pas qu'il serait impossible à activer ailleurs.

    La reconstruction cumulative quantifie ce qu'apportent les niveaux au décodage observé. Une entropie élevée n'est pas une preuve d'utilité, une faible entropie n'est pas une preuve d'inutilité, et une reconstruction partielle qui se dégrade après ajout d'un niveau ne suffit pas à conclure à une hiérarchie défectueuse.

    ![Audit tokenizer](figures/01_tokenizer.png)

    Les distances d'embeddings ont été complétées par un test fonctionnel : pour 32 solutions hard de train, remplacer le premier emplacement de chaque niveau par chacun des 16 codes, puis comparer les 120 paires dans les mêmes contextes, les autres codes restant fixés. Le test compare les cases argmax et la variation totale des distributions finales. Il ne teste qu'un emplacement par niveau et peut créer des combinaisons hors distribution.
    ''')
    for r in fun['levels']:add(f"- Niveau {r['level']} : désaccord de cases min/médian/max = {pct(r['min_cell_disagreement'])} / {pct(r['median_cell_disagreement'])} / {pct(r['max_cell_disagreement'])} ; {r['zero_disagreement_pairs']}/120 paires sans désaccord argmax dans ces contextes ; corrélation de Spearman distance/effet = {('indéfinie (effet constant)' if r['embedding_effect_spearman'] is None else f(r['embedding_effect_spearman']))}.")
    add('''Les codes fins restent ici ceux de la solution connue : ce test ne régénère pas les niveaux suivants après substitution. Il peut donc masquer un rôle des codes précoces dans le conditionnement du générateur. Une paire sans changement argmax peut conserver des probabilités différentes. Ni ce test ni une petite distance géométrique ne prouvent une interchangeabilité globale. Les matrices complètes sont conservées dans results/functional_substitutions.json.
    ''')
    constant=read('constant_level'); cc=constant['conditions']['remove_level5_residual']
    add(f"**Complément post-hoc sur le niveau constant** : suppression de la contribution résiduelle du niveau 5 hard, puis régénération des niveaux suivants, sur 128 puzzles test. Réussite {pct(cc['metrics']['solved_rate'])}, variation {pct(cc['delta_solved'])}, IC {interval(cc['delta_ci95'],True)}, {cc['changed_boards']} grilles modifiées ; sham identique à la génération native. Le niveau constant peut agir comme un décalage commun : ce test n'établit pas qu'il encode des différences entre puzzles. Il a été ajouté après observation de la faible diversité et reste exploratoire. Si aucun succès ne change de statut, l'IC bootstrap nul est dégénéré et ne prouve pas un effet populationnel nul.")
    if cc['success_status_changes']==0:
        add(f"Aucun changement de statut succès/échec sur 128 puzzles : borne supérieure Wilson à 95 % du taux de changements de statut {pct(cc['success_status_change_wilson95'][1])}. Cette borne complète l'IC bootstrap dégénéré [0,0] ; elle n'établit pas une équivalence de performance à toute précision.")
    add('## 5. Déterminisme et dépendance au batch — items 5–6')
    for name,r in stability['contrasts'].items():add(f"- {name} : écart maximal d'état {r['hidden_max_abs']:.6g}, écart relatif L2 {r['hidden_relative_l2']:.3g}, codes divergents {r['code_disagreements']}, tokens finaux divergents {r['output_disagreements']}.")
    add('''Les répétitions testent 32 puzzles hard, trois fois, à poids et entrées constants. Les comparaisons de tailles 1/8/32, ordre inversé et composition élargie à 64 séparent l'arrondi FP32 d'un changement discret de décision. Ces résultats sont locaux au CPU et ne garantissent pas un comportement identique sur CUDA ou MPS.

    ## 6. Symétries et stabilité du score — item 4
    ''')
    for name,r in sym['variants'].items():
        met=r['metrics']; eq=r['equivariance_whole_board_agreement']; conf=trans['symmetry_traces_'+name].get('confidence_vs_identity')
        add(f"- **{name}**, n=128 : réussite {pct(met['solved_rate'])}"+(f", accord de grille après réalignement {pct(eq)}" if eq is not None else '')+(f", variation absolue moyenne de confiance {conf['mean_abs_change']:.3f}, IC {interval(conf['mean_abs_change_ci95'])}; différence de réussite vs identité {pct(conf['delta_success'])}, IC {interval(conf['delta_success_ci95'],True)}." if conf else '.'))
    add(f"Les ratios de variance intra/inter-puzzles des résumés, niveaux 1 à 12, valent : {', '.join(f'{v:.2f}' for v in sym['intra_inter_ratio'])}. Ces coordonnées ne sont pas garanties invariantes aux transformations. Une forte variance ne constitue donc pas une preuve de bruit ; l'équivalence des réponses réalignées et le comportement de la sonde gelée sont plus directement interprétables. Une seule permutation des chiffres et deux transformations spatiales sont testées ; les 128 variantes partagent leurs puzzles sources, et les IC les rééchantillonnent par source.")
    add('''## 7. Généralisation et difficulté — items 7–8

    Le croisement checkpoint/corpus utilise les mêmes 256 puzzles par corpus pour tous les checkpoints. Il sépare ainsi le changement de difficulté évaluée du changement des poids, dans les limites de cet échantillon.

    ![Modèles et corpus](figures/05_cross_models.png)
    ''')
    for model in rep:add('- **'+model+'** : '+', '.join(d+' '+pct(cross[model+'_on_'+d]['solved_rate']) for d in rep)+'.')
    cp=np.load(ROOT/'results/cross_predictions.npz'); split=read('splits'); contrasts={}
    for corpus in rep:
        _,refs=boards(corpus); refs=refs[np.array(split['cross'][corpus])]; hard_y=np.all(cp['hard_on_'+corpus]==refs,axis=1)
        for other in ['easy','medium']:
            other_y=np.all(cp[other+'_on_'+corpus]==refs,axis=1); delta=hard_y.astype(float)-other_y
            rng=np.random.default_rng(P['seed']); boots=delta[rng.integers(0,len(delta),(P['bootstrap'],len(delta)))].mean(1)
            contrasts['hard_minus_'+other+'_on_'+corpus]=dict(n=len(delta),delta=float(delta.mean()),paired_ci95=np.quantile(boots,[.025,.975]).tolist())
    write(ROOT/'results/cross_model_contrasts.json',contrasts)
    contrast=contrasts['hard_minus_easy_on_hard']
    add(f"Sur le corpus hard commun, l'écart hard–easy est de {pct(contrast['delta'])}, IC bootstrap apparié {interval(contrast['paired_ci95'],True)}. Ces comparaisons secondaires sont exploratoires, sans correction de multiplicité ; près du plafond sur easy, les petits écarts demandent davantage de puzzles pour être interprétés. Le profil observé ne permet pas d'assimiler toutes les difficultés à un même modèle.")
    met=ext['metrics']; add(f"Un corpus indépendant de {met['n']} puzzles à 30 indices a été produit par backtracking aléatoire puis retrait glouton d'indices. Chaque puzzle possède une solution unique selon backtracking MRV et Algorithm X. Les solutions équivalentes aux solutions officielles selon les symétries testées ont été exclues. Réussite hard : **{met['solved']}/{met['n']} = {pct(met['solved_rate'])}**, IC {interval(met['solved_wilson95'],True)}, contre {pct(rep['hard']['metrics']['solved_rate'])} sur le corpus officiel complet.")
    official_array=np.load(ROOT/'results/hard.npz'); external_array=np.load(ROOT/'results/external.npz')
    official_diff=official_array['baseline'][:,0,:6]; external_diff=external_array['baseline'][:,0,:6]
    pool=np.sqrt((official_diff.var(0)+external_diff.var(0))/2); smd=(external_diff.mean(0)-official_diff.mean(0))/np.maximum(pool,1e-12)
    add(f"Nombre moyen de candidats initiaux par case vide : officiel {official_diff[:,1].mean():.3f}, nouveau {external_diff[:,1].mean():.3f}. Proportion initiale de célibataires nus : officiel {official_diff[:,3].mean():.3f}, nouveau {external_diff[:,3].mean():.3f}. Différences de moyennes standardisées des six proxies : {', '.join(f'{v:.3f}' for v in smd)} ; la variance nulle de la proportion de cases vides (30 indices partout) ne porte pas d'information de difficulté.")
    add('''Ce corpus n'est pas uniforme sur les Sudokus et le retrait glouton modifie la distribution. Un écart de performance mesure un décalage de distribution combinant structure, difficulté et mécanisme de génération ; il ne permet pas de distinguer mémorisation et généralisation sans accès à l'entraînement. La contamination reste indéterminée.

    Les proxies de difficulté sont la proportion de cases vides, moyenne et dispersion du nombre initial de candidats, proportion de célibataires nus, proportion de cases remplies par leur propagation et nombre de tours de propagation. Ils ne représentent pas toute la difficulté logique ou humaine. La baseline les inclut ; le contrôle difficulté seule et la stratification par quartiles train apportent un diagnostic supplémentaire, sans ajustement causal exhaustif.
    ''')
    for r in probes['difficulty_strata']:add(f"- Strate {r['bin']}, n={r['n']}, réussite {pct(r['success_rate'])} : AUROC états {f(r['hidden_auc'])}, baseline {f(r['baseline_auc'])}, combinaison {f(r['combined_auc'])}.")
    add('''## 8. Discrimination, calibration et potentiel de monitoring — items 9–10

    Toutes les probabilités ci-dessous sont des sorties de sondes externes. Une AUROC élevée signifie que le score ordonne succès et erreurs ; elle n'implique pas calibration ni mécanisme métacognitif natif. Un prior constant peut être raisonnablement calibré et possède une AUROC de 0,5.
    ''')
    for name,m in probes['levels'][k]['models'].items():add(f"- **{name}**, niveau 6 : AUROC {f(m['auc'])}, Brier {m['brier']:.4f}, ECE à 10 intervalles {m['ece10']:.4f}, erreurs test {m['failures']}/{m['n']}.")
    prior_m=probes['levels'][k]['models']['prior']; combined_m=probes['levels'][k]['models']['combined']
    add(f"Exemple de dissociation : le prior constant présente une ECE de {prior_m['ece10']:.3f}, contre {combined_m['ece10']:.3f} pour la combinaison, tout en ayant une AUROC de 0,5 contre {combined_m['auc']:.3f}. Dans cette population où les succès dominent, une calibration moyenne favorable ne garantit donc pas une capacité à distinguer les erreurs.")
    add('''Le Brier combine calibration et résolution ; l'ECE dépend du découpage et de l'effectif par intervalle. Les courbes de risque/couverture conservent tous les scores ex aequo ensemble. Ici, les 6 000 puzzles officiels sont uniques : validité et respect des indices suffisent à caractériser la réussite. L'AUROC de la baseline atteint 1 au dernier niveau hard ; ce résultat est compatible avec l'accès au vérificateur externe, sans supposer un monitoring interne. Les derniers niveaux peuvent révéler la validité de la réponse déjà produite : cela ne constitue pas une anticipation de l'erreur.

    ![Sondes par niveau](figures/03_probes.png)

    ![Calibration et sélection](figures/04_calibration.png)

    Les réplications suivantes ajustent séparément les sondes sur chaque checkpoint, sur des familles de solutions partagées entre difficultés, avec la même procédure et des partitions de 1 000/400/600. Le faible nombre d'erreurs, surtout sur easy, limite fortement la précision de l'AUROC et interdit d'en tirer une hiérarchie générale entre modèles.
    ''')
    for d,r in repl.items():add(f"- **{d}** : erreurs test {r['baseline']['failures']}/600 ; AUROC baseline {f(r['baseline']['auc'])}, états {f(r['hidden']['auc'])}, combinaison {f(r['combined']['auc'])}."+(f" Gain combiné–baseline {r['contrast']['delta']:.4f}, IC {interval(r['contrast']['delta_ci95'])}." if r['contrast'] else ' Contraste non défini : une seule classe.'))
    transfer=trans['external'];add(f"Sur le corpus nouveau, les sondes hard gelées au niveau 6 donnent : AUROC baseline {f(transfer['baseline']['auc'])}, états {f(transfer['hidden']['auc'])}, combinaison {f(transfer['combined']['auc'])}. Les métriques complètes, calibration comprise, figurent dans results/transfer_confidence.json. Le transfert direct entre checkpoints est conservé séparément et limité aux 600 familles de solutions du test hard, exclues de l'entraînement et de la validation des sondes hard ; leurs coordonnées cachées ne sont pas alignées, aussi son échec éventuel ne prouve pas l'absence d'information décodable.")
    add(f'''## 9. Sonde de contrainte naked single — item 11, exploratoire

    Une sonde locale utilise les 512 activations des positions correspondant aux cases au dernier niveau. La cible indique si une case initialement vide possédait exactement un candidat à partir des seuls indices. Entraînement : 512 puzzles de train ; validation : 128 ; test : {naked['test_puzzles']} puzzles, {naked['test_cells']} cases vides, dont {naked['positives']} positives. Les puzzles, et non les cases, définissent les partitions et les unités bootstrap.

    AUROC états : **{f(naked['hidden']['auc'])}** ; labels d'entraînement mélangés : **{f(naked['shuffled_labels']['auc'])}** ; contraste IC {interval(naked['contrast']['delta_ci95'])}. Le calcul explicite du nombre de candidats atteint par construction une AUROC de 1 pour cette cible.

    Ce résultat concerne la décodabilité d'une contrainte initiale dans un état final. Il ne localise pas un circuit monosémantique, ne montre pas que la règle est calculée à cet endroit ni qu'elle est utilisée causalement. Le candidat explicite est un contrôle sémantique de la cible, pas un modèle appris concurrent. Une recherche de neurones spécifiques et des patchings de règles sélectifs restent nécessaires avant toute affirmation de circuit.

    ## 10. Ablation causale — item 12, exploratoire

    Une sonde sur les 512 moyennes d'activations du niveau 6 fournit une direction linéaire, apprise sur train et sélectionnée sur validation. L'intervention retire la projection des activations centrées sur cette direction, à la couche 8 du niveau 6 uniquement ; le reste de la génération se poursuit normalement. Le centrage utilise les moyennes train. Les 600 puzzles test sont les seules entrées utilisées pour mesurer les effets causaux.

    Cinq directions aléatoires unitaires reçoivent, pour chaque token et puzzle, exactement la norme absolue de projection de l'intervention ciblée ; leur signe suit leur propre projection. Ce contrôle apparie la taille de la perturbation, sans prétendre apparier toute sa géométrie. Le sham ne modifie aucun état.
    ''')
    for name,r in conditions.items():add(f"- **{name}** : réussite {pct(r['metrics']['solved_rate'])}, variation vs génération native {pct(r['delta_solved'])}, IC {interval(r['delta_ci95'],True)} ; erreurs introduites {r['introduced_errors']}, erreurs corrigées {r['rescued']}, grilles modifiées {r['changed_boards']}.")
    random_delta=[r['delta_solved'] for n,r in conditions.items() if n.startswith('random_')]
    add(f"L'effet ponctuel ciblé vaut {pct(conditions['confidence']['delta_solved'])}, contre une plage de {pct(min(random_delta))} à {pct(max(random_delta))} pour les cinq directions aléatoires. L'intervention ciblée n'apporte donc pas, à elle seule, une preuve de spécificité métacognitive ; cette comparaison est descriptive et ne transforme pas cinq contrôles en test exhaustif de directions.")
    add('''![Interventions causales](figures/06_causal.png)

    Ces interventions testent l'influence d'une direction sur la résolution. Une baisse de performance peut refléter une information utile à la tâche ou une perturbation générique. Elle n'établit pas un contrôle métacognitif natif. Cinq contrôles aléatoires ne permettent pas une estimation précise des queues de leur distribution. Un effet nul n'exclut pas redondance, compensation ou mauvaise localisation de la direction.

    ## 11. Progression de la représentation — item 13, exploratoire

    Une projection gaussienne fixe réduit les résumés de 1 024 à 128 dimensions. Une régression ridge (alpha=100, sans sélection sur test) prédit les 81 distributions de chiffres à partir des états de chaque niveau ; les métriques portent sur les cases initialement vides. Un contrôle utilise les seuls indices, encodés one-hot puis projetés en 128 dimensions, avec la même ridge. Les cibles ne sont utilisées qu'à l'entraînement ou pour le calcul final des métriques.

    Les états sont observés avant la tête qui prédit les codes du niveau ; les décodages intermédiaires sont calculés après ajout des codes de ce niveau. Les deux courbes ne représentent donc pas exactement le même instant du calcul.

    ![Décodabilité et révisions](figures/07_hierarchy.png)

    Le profil est descriptif : aucun test de rupture, analyse de taille finie ou réplication par graines ne justifie de parler de transition de phase. Les révisions correct→incorrect montrent directement que l'affinement n'est pas obligatoirement monotone. Une solution décodable n'implique ni engagement irréversible ni utilisation causale de cette lecture linéaire.

    ## 12. Limites et portée des conclusions

    - Trois checkpoints Sudoku, une version publiée et un backend CPU ; aucune conclusion sur toute la famille NCM ou sur les modèles de texte.
    - Un découpage de données et une graine d'entraînement des sondes ; bootstrap conditionnel aux sondes ajustées. Les performances de train ou validation ne servent pas de preuves finales.
    - Nombre d'erreurs limité ; particulièrement peu d'erreurs dans les réplications easy. Les métriques par strate et par niveau sont exploratoires.
    - Symétries testées partielles ; absence de doublons dans cet audit ne prouve pas absence de contamination d'entraînement.
    - Le corpus nouveau ajoute un décalage de générateur et ne définit pas un test de mémorisation.
    - La baseline ne contient pas toutes les informations possibles des sorties ; un gain de la sonde combinée serait relatif à cette baseline précise.
    - Entropie des codes, reconstruction et décodabilité ne prouvent pas une représentation conceptuelle humaine ni une conscience de sa fiabilité.
    - Le test naked single ne caractérise pas de circuit monosémantique. L'ablation de confiance ne démontre pas le contrôle métacognitif.
    - Aucun entraînement de NCM, aucune étude longitudinale de checkpoints et aucune mesure comparative de coût GPU ne sont réalisés.

    ## 13. Sources, preuves et reproduction

    Les expériences sont exécutées depuis ce dossier. Pour reconstruire toutes les sorties (opération longue) :

    ```sh
    ./run.sh
    ```

    Pour vérifier les artefacts existants et rejouer les sondes sauvegardées :

    ```sh
    PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" .venv/bin/python scripts/verify.py
    ```

    Pour recréer l'environnement local : Python 3.13, venv et installation de requirements.txt. Le fichier sources/requirements-lock.txt est un instantané de l'environnement complet, incluant des paquets système non utilisés. L'environnement de campagne utilise des paquets système en lecture ; le verrou enregistre les versions réellement résolues. Les variables de caches de run.sh doivent être conservées pour confiner les écritures. Les résultats bit à bit ne sont garantis ni entre versions BLAS ni entre architectures.

    - **Rapport et méthode** : RAPPORT.md, TODO.md, sources/TODO_protocol_v1.md, protocol.json et JOURNAL.md.
    - **Code exécuté** : scripts/common.py, prepare.py, extract.py, analyze.py, solvers.py, experiments.py, upstream_parity.py, report.py et verify.py ; smoke_checks.py, recover_summaries.py et continue.py assurent contrôles, récupération des seuls résumés et ordonnancement ; run.sh orchestre une reproduction séquentielle. La campagne initiale exécute certaines étapes indépendantes simultanément, avec un thread de calcul par processus.
    - **Code des modèles** : vendor/ncm_mac et vendor/upstream-ncm ; ce dernier reste intact. vendor/reference_metacognition est une copie de référence non importée par les expériences ; son inclusion ne signifie pas que ses expériences ont été exécutées ici.
    - **Entrées exactes** : artifacts/sudoku, data/sudoku et data/external.json. Les poids ne sont pas modifiés. Les fichiers pickle officiels sont chargés par un lecteur n'acceptant que les données simples.
    - **Preuves par puzzle** : results/hard.npz, easy.npz et medium.npz ; préfixes, références, codes encodés/générés, décodages intermédiaires, états résumés et statistiques sans référence. hard.npz contient aussi les activations locales finales.
    - **Sondes** : results/probe_models.pkl, naked_single_probe.pkl, probe_predictions.npz et causal_direction.npz. Ne charger les pickles de sondes qu'en provenance de ce dossier de confiance.
    - **Interventions** : results/causal_predictions.npz ; résultats de symétries, substitutions, croisement et transfert dans les fichiers JSON/NPZ correspondants.
    - **Inspection simple** : results/summary.csv, probes.csv et predictions.csv ; figures en PNG et SVG.
    - **Intégrité** : results/input_manifest.json, final_manifest.json et verification.json ; le dernier indique les contrôles effectivement passés. Les logs sont dans logs/.
    - **Archive de sources** : sources_utilisees.zip contient les scripts, les deux implémentations des modèles, les configurations et le protocole ; les gros poids et les traces restent disponibles dans les dossiers ci-dessus.

    Références primaires : [article NCM](https://arxiv.org/html/2610.14162v1), [dépôt officiel](https://github.com/HazyResearch/ncm), [checkpoints publiés](https://huggingface.co/hazyresearch/ncm-checkpoints), [corpus publiés](https://huggingface.co/datasets/hazyresearch/ncm-datasets). Les copies de l'article consultées sont dans sources/ et leurs empreintes sont conservées. Les résultats numériques de ce rapport sont calculés localement, non repris de ces pages.
    ''')
    (ROOT/'RAPPORT.md').write_text('\n'.join(text))
    # Preserve the original frozen checklist and mark only the executed protocol complete.
    todo=(ROOT/'sources/TODO_protocol_v1.md').read_text().replace('- [ ]','- [x]')
    todo+='\nStatut : protocole expérimental sélectionné exécuté ; aucun circuit monosémantique ni contrôle métacognitif natif établi. Les limites et études non entreprises figurent au rapport.\n'
    (ROOT/'TODO.md').write_text(todo)
    with zipfile.ZipFile(ROOT/'sources_utilisees.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for sub in ['scripts','vendor/ncm_mac','vendor/upstream-ncm']:
            for path in (ROOT/sub).rglob('*'):
                if path.is_file() and '__pycache__' not in str(path) and '.git/' not in str(path) and (path.suffix in ['.py','.yaml','.txt','.md'] or path.name=='LICENSE'):
                    archive.write(path,str(path.relative_to(ROOT)))
        for path in ['run.sh','requirements.txt','README.md','protocol.json','TODO.md','JOURNAL.md','sources/TODO_protocol_v1.md','sources/requirements-lock.txt','results/protocol_frozen.json','results/input_manifest.json']:
            archive.write(ROOT/path,path)
    progress('report_complete',report='RAPPORT.md',figures=7)

if __name__=='__main__': main()
