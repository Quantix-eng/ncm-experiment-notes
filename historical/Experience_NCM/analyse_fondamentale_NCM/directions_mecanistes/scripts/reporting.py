"""Export figures, an autonomous French report and all study sources."""
from base import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zipfile,csv
FIG=D/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})

def read(n):return json.loads((D/f'results/{n}.json').read_text())
def pct(x):return f'{100*x:.2f} %'
def pp(x):return f'{100*x:.2f} points'
def ci(v,percent=False):return '['+', '.join(pct(x) if percent else f'{x:.4f}' for x in v)+']'
def save(fig,name):
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(FIG/(name+'.png'),bbox_inches='tight');fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close(fig)

def figures(diag,algebra,drop,first,conditional,external,steers):
    ds=['hard','easy','medium'];fig,axes=plt.subplots(1,2,figsize=(11,4.5));widths=[8,9,10,18]
    for d in ds:
        axes[0].plot(widths,[algebra['checkpoint_variances'][d][str(w)]['raw_embedding_inter_puzzle_variance'] for w in widths],'.-',label=d)
        axes[1].plot(range(1,13),[v['pooling_inter_variance_ratio'] for v in diag[d]['levels']],'.-',label=d)
    axes[0].set(xlabel='Nombre de positions après pooling',ylabel='Variance inter-solutions des embeddings',yscale='log',title='Avant GroupNorm : invariance à 9 positions')
    axes[1].set(xlabel='Niveau du tokenizer',ylabel='Variance après pooling / variance avant',yscale='log',title='Atténuation du résidu natif par le pooling')
    for a in axes:a.legend();a.grid(alpha=.2)
    fig.suptitle('Source : 256 solutions train par checkpoint ; variances moyennées sur positions et canaux');save(fig,'01_pooling')
    controls=diag['controls_hard'];names=list(controls);labels=['Solutions','Chiffres permutés','Ordre dans lignes','Ordre global','Chiffres i.i.d.']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));axes[0].bar(range(len(names)),[controls[n]['codes5']['used'] for n in names],color='steelblue')
    axes[0].set(xticks=range(len(names)),xticklabels=labels,ylabel='Codes VQ observés (sur 16)',title='Niveau 5 : contrôles de structure des lignes',ylim=(0,17));axes[0].tick_params(axis='x',labelrotation=25)
    for d in ds:axes[1].plot(range(16),diag[d]['levels'][4]['embedding_norms'],'.-',label=d)
    axes[1].set(xlabel='Identifiant du code VQ du niveau 5',ylabel='Norme euclidienne du vecteur',title='Un code proche de zéro et des codes plus éloignés');axes[1].legend()
    fig.suptitle('Source : 64 entrées par contrôle, hard ; les trois derniers contrôles sont des entrées invalides');save(fig,'02_controls')
    models=['baseline','state','combined','shuffled_states'];labels=['Sorties','États','Combinaison','États permutés'];xx=np.arange(4)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));rr=first['results']['next'];cr=conditional['results']
    axes[0].bar(xx-.18,[rr[n]['all']['accuracy']*100 for n in models],.36,label='Toutes les cases vides')
    axes[0].bar(xx+.18,[rr[n]['changed_only']['accuracy']*100 for n in models],.36,label='Cases effectivement changées')
    axes[0].set(title='v1 : entraînement du prochain token sur toutes les cases',ylabel='Token du niveau 9 prédit correctement (%)',ylim=(0,70));axes[0].legend(fontsize=8)
    axes[1].bar(xx,[cr[n]['changed_only']['accuracy']*100 for n in models],color='steelblue',label='Sonde conditionnelle')
    axes[1].axhline(conditional['contrasts']['second_accuracy']*100,color='black',ls='--',label='Deuxième choix du décodeur 8')
    axes[1].set(title='Complément : apprentissage des valeurs de remplacement',ylabel='Token prédit correctement parmi les changements (%)',ylim=(0,70));axes[1].legend(fontsize=8)
    for a in axes:a.set_xticks(xx,labels);a.tick_params(axis='x',labelrotation=15)
    fig.suptitle('Source : test hard, 600 puzzles ; le sous-groupe changé est défini après observation du futur');save(fig,'03_next_token')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5));cr=conditional['results']
    for offset,name,label in [(-.15,'baseline','Sorties observables'),(.15,'combined','Sorties + états')]:
        axes[0].bar(np.arange(2)+offset,[cr[name]['changed_only']['accuracy']*100,external[name]['changed_only']['accuracy']*100],.3,label=label)
        axes[1].bar(np.arange(2)+offset,[cr[name]['gate_auc'],external[name]['gate_auc']],.3,label=label)
    for a in axes:a.set_xticks([0,1],['Test officiel (600)','Externe (256)']);a.legend(fontsize=8)
    axes[0].set(ylabel='Valeurs de remplacement correctes (%)',ylim=(0,65),title='Prévision conditionnelle du prochain token')
    axes[1].set(ylabel='AUROC du changement',ylim=(0,1),title='Porte de changement sans information future')
    fig.suptitle('Source : complément exploratoire ; aucune sonde ajustée sur le corpus externe');save(fig,'04_external')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));names=['positive','negative','random0','random1','random2']
    for ax,(tag,r) in zip(axes,steers.items()):
        for offset,key,label in [(-.15,'full_logodds_gain','Intervention complète'),(0,'indirect_logodds_gain','Nouveaux codes 9 seuls'),(.15,'direct_only_logodds_gain','Lecture directe, codes fixes')]:
            vals=[r['conditions']['0.25_'+n][key] for n in names];means=np.array([v['mean'] for v in vals]);bounds=np.array([v['ci95'] for v in vals])
            ax.errorbar(np.arange(5)+offset,means,yerr=np.stack([means-bounds[:,0],bounds[:,1]-means]),fmt='o',capsize=2,label=label,ms=4)
        ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(5),['Positive','Négative','Aléatoire 1','Aléatoire 2','Aléatoire 3']);ax.tick_params(axis='x',labelrotation=20)
        ax.set(ylabel='Gain du log-rapport chiffre demandé / actuel (nats)',title='Direction '+tag);ax.legend(fontsize=7)
    fig.suptitle('Source : 96 puzzles test ; dose 0,25 ; IC 95 % appariés par puzzle, 2 000 tirages');save(fig,'05_steering')
    fig,ax=plt.subplots(figsize=(8,4));xx=np.arange(3)
    native=[drop[d]['native']['perplexity'] for d in ds];runs=np.array([[v['codes']['perplexity'] for v in drop[d]['dropout_runs']] for d in ds])
    ax.bar(xx-.18,native,.36,label='Encodeur en évaluation');m=runs.mean(1);ax.bar(xx+.18,m,.36,label='Dropout encodeur actif',yerr=np.stack([m-runs.min(1),runs.max(1)-m]),capsize=3)
    ax.set(xticks=xx,xticklabels=ds,xlabel='Checkpoint du tokenizer',ylabel='Perplexité empirique des codes, niveau 5',ylim=(0,3),title='Même checkpoint : régime déterministe et dropout configuré');ax.legend()
    fig.suptitle('Source : 64 solutions train par checkpoint ; 4 graines ; barres d’erreur = min/max des graines');save(fig,'06_dropout')

def main():
    diag=read('level5_diagnostic');algebra=read('pooling_algebra');drop=read('dropout_control');l5=read('level5_interventions')
    first=read('probes');firstext=read('external_probes');cond=read('conditional_probes');external=read('conditional_external');steers={'v1':read('steering'),'conditionnelle':read('steering_conditional')}
    figures(diag,algebra,drop,first,cond,external,steers)
    text=[]
    def add(s):text.append('\n'.join(line[4:] if line.startswith('    ') else line for line in s.strip('\n').splitlines()).strip()+'\n')
    primary=cond['contrasts']['combined_minus_fitted_baseline'];extprimary=external['contrasts']['combined_minus_fitted_baseline'];h=diag['hard']['levels'][4]
    add(f'''# NCM : niveau 5 peu diversifié et révisions au niveau 9

    Rapport de la seconde campagne — 24 octobre 2026. Analyse exploratoire des checkpoints Sudoku publiés, principalement hard. Le [rapport précédent](../RAPPORT.md) et ses résultats restent inchangés.

    ## Résultats principaux

    **Niveau 5 : une contrainte structurelle de représentation est identifiée.** À cette échelle, le pooling du tokenizer regroupe chaque ligne complète du Sudoku. Toutes les solutions ont les mêmes histogrammes de chiffres par ligne ; avant GroupNorm et soustraction des contributions précédentes, les moyennes d'embeddings sont donc invariantes, à l'arrondi près. Cela explique un déficit de variation lié au couple données/architecture, sans démontrer à lui seul la cause historique du codebook presque constant.

    **Une prédiction peut être évitée sur les données testées.** Forcer le code VQ 11 au niveau 5 hard, tout en conservant sa contribution, reproduit les codes et les réponses natives sur les 600 puzzles test et les 256 externes : 11 passes de prédiction des niveaux au lieu de 12. Il s'agit d'une équivalence sur ces corpus ; aucun gain de vitesse universel n'est revendiqué.

    **La valeur de remplacement est partiellement prévisible depuis les sorties observables.** La sonde conditionnelle combinée atteint {pct(cond['results']['combined']['changed_only']['accuracy'])}, contre {pct(cond['results']['baseline']['changed_only']['accuracy'])} pour la baseline observable ajustée. Gain {pp(primary['delta'])}, IC 95 % {ci(primary['ci95'],True)}. Sur le corpus externe : gain {pp(extprimary['delta'])}, IC {ci(extprimary['ci95'],True)}. L'ajout des états internes n'apporte donc pas de gain établi dans ce protocole conditionnel.

    **Les interventions ne démontrent pas une direction de révision spécifique utilisée par le générateur.** Les contrastes principaux via les codes 9 recouvrent zéro dans les deux séries. Les différences de lecture directe ne doivent pas être prises pour un mécanisme causal de génération.

    ## 1. Objectifs, protocole et portée

    L'étude suit [OBJECTIFS_ET_PROTOCOLE.md](OBJECTIFS_ET_PROTOCOLE.md), les paramètres [protocol.json](protocol.json) et le gel [results/protocol_frozen.json](results/protocol_frozen.json). Les amendements [conditionnel](AMENDEMENT_CONDITIONNEL.md) et [dropout](AMENDEMENT_DROPOUT.md) sont explicitement post-hoc ; ils ne remplacent pas les résultats v1. Le [journal](JOURNAL.md) indique leur motivation et l'ordre des analyses.

    Les partitions de la campagne précédente sont conservées : 1 000 puzzles train, 400 validation, 600 test, groupés par familles de solutions. Les trois difficultés partagent les mêmes familles ; leurs diagnostics ne sont pas trois réplications indépendantes des données. Le corpus externe contient 256 puzzles uniques de la campagne précédente et ne sert à aucun ajustement. Il avait déjà été examiné pour d'autres questions ; après consultation de v1, il ne constitue plus une confirmation intacte pour les compléments.

    Les poids NCM sont gelés, CPU FP32, un thread par processus, batch 16, graine {Q['seed']}. Les intervalles prédictifs rééchantillonnent les puzzles et les comptes de cellules par puzzle, jamais des cellules supposées indépendantes. Les intervalles causaux sont appariés par puzzle (2 000 tirages). Les sondes ne sont pas réajustées dans le bootstrap ; les IC sont conditionnels à ce découpage et à ces ajustements. Les comparaisons exploratoires multiples ne font pas l'objet d'une correction de multiplicité.

    Le runner explicite conserve les masques, le cache de préfixe, le calcul des logits et l'accumulation du port CPU. Avant toute interprétation, il reproduit bit à bit les codes, les logits finaux et les états z8/z9 sur 32 puzzles, deux lots distincts. À l'extraction, les codes et décodages 8/9 sont comparés aux traces précédentes sur la totalité des 2 256 puzzles hard et externes.

    ## 2. Ce que signifie ici un niveau « mort »

    Le niveau 5 hard encode et génère un seul identifiant VQ sur les solutions/puzzles officiels examinés. Cette constance ne signifie ni activations neuronales nulles, ni contribution exactement nulle, ni inutilité dans tout régime d'entrée. Les codebooks easy et medium sont également peu diversifiés à ce niveau, avec une diversité différente de hard.

    Le diagnostic utilise les 256 premières familles de train et 256 de test, sans entraîner NCM : variances à position fixée puis moyennées sur canaux/positions, distances aux codes, marges du plus proche voisin, MSE des résidus et statistiques EMA sauvegardées. L'accumulation explicite doit donner les mêmes codes que l'encodeur. Un calcul direct des distances en double précision est comparé au calcul rapide sur 256 vecteurs par niveau et checkpoint.
    ''')
    for d in ['hard','easy','medium']:
        v=diag[d];k=v['levels'][4];err=sum(x['direct_float64_assignment_disagreements'] for x in v['levels'])
        add(f"- **{d}** : {k['codes']['used']}/16 codes observés au niveau 5 sur l'échantillon train ; variance inter-puzzles de son entrée {k['input']['inter_puzzle_variance']:.6g}, soit {pct(k['pooling_inter_variance_ratio'])} de la variance du résidu pleine résolution ; {err} désaccord de sélection entre distances rapides et directes sur les échantillons vérifiés. EMA positives pour {k['ema_active']}/16 codes.")
    add('''Les EMA sont des moyennes mobiles sauvegardées, pas des comptages de ce corpus ni un journal temporel. Leur positivité est incompatible avec l'interprétation simpliste « ces codes n'ont jamais été utilisés », mais elle ne permet pas de reconstruire quand et pourquoi ils l'ont été. Les tests de distances ne révèlent pas de bug de sélection dans les cas vérifiés ; un problème d'entraînement original reste une possibilité non démontrée.

    ## 3. Pourquoi neuf positions constituent une échelle particulière

    Les solutions sont sérialisées en 89 tokens : neuf chiffres par ligne et huit séparateurs. Le pooling adaptatif à neuf positions utilise les fenêtres floor(j × 89/9) : ceil((j+1) × 89/9). Chacune contient les neuf chiffres d'une ligne, plus un séparateur aux extrémités ou deux pour les lignes intérieures.

    L'encodeur configuré ici est une somme d'embeddings de tokens et de positions, sans blocs contextuels. Chaque ligne contient une fois chacun des chiffres 1 à 9 ; les embeddings positionnels sont fixes. La moyenne d'embeddings de chaque fenêtre avant normalisation est ainsi indépendante de l'ordre des chiffres dans la ligne. Les histogrammes de tokens sont exactement identiques entre les 256 solutions vérifiées ; la propriété découle également de leur validité Sudoku.
    ''')
    for d in ['hard','easy','medium']:
        vv=algebra['checkpoint_variances'][d]
        add(f"- **{d}**, variance des embeddings moyennés avant GroupNorm : pooling 8 = {vv['8']['raw_embedding_inter_puzzle_variance']:.5g}, pooling 9 = {vv['9']['raw_embedding_inter_puzzle_variance']:.5g}, pooling 10 = {vv['10']['raw_embedding_inter_puzzle_variance']:.5g}. Après GroupNorm, pooling 9 = {vv['9']['normalized_embedding_inter_puzzle_variance']:.5g}.")
    add('''![Pooling et variance](figures/01_pooling.png)

    La GroupNorm peut réintroduire une petite variation. Surtout, le niveau 5 quantifie un résidu après les quatre contributions précédentes : l'invariance des embeddings d'origine n'impose donc pas mathématiquement un code résiduel unique. Le constat établit un mécanisme d'atténuation de la variation, pas un théorème de collapse du codebook ni une propriété générale du coarse-to-fine.

    Les contrôles hard conservent les mêmes 64 solutions sources. La permutation des chiffres est une symétrie valide ; les permutations internes à chaque ligne, le mélange global des cases et les chiffres i.i.d. sont des entrées diagnostiques généralement invalides, et ne servent pas à mesurer une performance Sudoku.
    ''')
    for name,v in diag['controls_hard'].items():add(f"- **{name}** : {v['codes5']['used']}/16 codes utilisés ; variance de l'entrée résiduelle du niveau 5 {v['u5']['inter_puzzle_variance']:.6g}.")
    add('''![Contrôles du niveau 5](figures/02_controls.png)

    Le maintien d'un seul code après permutation interne aux lignes et l'activation de nombreux codes après mélange global soutiennent une sensibilité à la structure des moyennes par ligne. Le codebook n'est pas incapable d'activer d'autres entrées ; celles-ci peuvent toutefois correspondre à des entrées hors distribution.

    ## 4. Quantification restante et régime de dropout

    Un k-means à 16 centres est ajusté aux seules entrées résiduelles du niveau 5 des 256 solutions de train. Les distances sont évaluées sur les 256 familles test. Le décodeur et le générateur restent inchangés ; il ne s'agit pas de remplacer le tokenizer puis de revendiquer une amélioration de résolution.
    ''')
    for d in ['hard','easy','medium']:
        k=diag[d]['quantizer_refit'];add(f"- **{d}** : MSE test du codebook original {k['test_mse_original']:.6g}, k-means {k['test_mse_kmeans']:.6g}, réduction relative {pct(k['relative_mse_reduction'])}, {k['test_usage']['used']}/16 centres utilisés sur test.")
    add('''Il reste donc de la variation géométriquement exploitable après pooling, malgré la faible diversité du codebook original en évaluation. Sa meilleure quantification n'est pas une preuve d'information utile au Sudoku : elle peut notamment refléter les contributions résiduelles précédentes.

    Le contrôle de dropout intervient uniquement sur encoder.drop, avec sa probabilité configurée. Les autres modules restent en évaluation et l'encodeur quantifie par le chemin sans mise à jour EMA. Quatre graines sont appliquées aux mêmes 64 solutions de train ; les state_dict sont comparés avant/après. Ce contrôle teste un régime d'activation sur poids gelés, sans reproduire l'histoire d'entraînement.
    ''')
    for d in ['hard','easy','medium']:
        r=drop[d];vals=[v['codes']['perplexity'] for v in r['dropout_runs']]
        add(f"- **{d}**, dropout p={r['dropout_probability']:.2f} : évaluation {r['native']['used']}/16 codes, perplexité {r['native']['perplexity']:.3f} ; dropout actif, codes cumulés {r['dropout_pooled_codes']['used']}/16, perplexité moyenne par répétition {np.mean(vals):.3f} (min–max {min(vals):.3f}–{max(vals):.3f}). Poids et EMA identiques : {r['state_dict_before']==r['state_dict_after']}.")
    add('''![Dropout et diversité](figures/06_dropout.png)

    L'interprétation doit distinguer invariance des entrées propres, usage du codebook sous bruit et causalité historique de l'optimisation. Le dropout augmente effectivement la diversité sur les trois checkpoints, tout en conservant une forte dominance des codes natifs. La désactivation du dropout restaure les régularités des moyennes de lignes ; son activation les perturbe. Cela ne prouve pas que le dropout est l'unique cause de l'apprentissage observé. Les trajectoires et graines d'entraînement originales restent indisponibles.

    ## 5. Contribution du niveau 5 et prédiction évitable

    Deux interventions sont comparées aux prédictions natives sauvegardées : (a) produire le code 11 déterminé sur train, sans exécuter la prédiction Transformer du niveau 5, puis conserver sa contribution ; (b) effectuer la prédiction mais annuler la contribution résiduelle du niveau 5, puis régénérer les niveaux suivants. Les positions et les autres niveaux conservent leur configuration.
    ''')
    for corpus,conditions in l5.items():
        for name,v in conditions.items():
            add(f"- **{corpus}, {name}** : {v['metrics']['solved']}/{v['metrics']['n']} grilles exactes ; {v['changed_grids']} grilles et {v['changed_codes']} codes différents du natif ; {v['introduced_errors']} erreurs introduites, {v['rescued']} corrigées ; variation de réussite {pp(v['delta_success']['mean'])}, IC {ci(v['delta_success']['ci95'],True)} ; {v['generator_calls_per_batch']} prédictions de niveaux par batch. Borne Wilson supérieure du taux de changement de statut succès/échec : {pct(v['success_status_change_wilson95'][1])}.")
    add('''Le remplacement par le code fixe est exactement équivalent sur les corpus testés. La suppression du résidu modifie certaines réponses tout en laissant les succès/échecs inchangés. L'IC bootstrap [0,0] dans ce cas est dégénéré : il ne démontre ni absence d'effet populationnel ni équivalence à toute précision. Les bornes Wilson donnent un repère sur les discordances non observées.

    Onze passes au lieu de douze ne représentent pas automatiquement 8,3 % de temps gagné : les niveaux ont des tailles différentes et le préfixe, le cache et le décodeur ont un coût. Les durées brutes sont sauvegardées à titre descriptif, avec conditions non randomisées et calculs parfois concurrents ; aucun benchmark de vitesse n'est revendiqué.

    ## 6. Prévoir la révision sans lire le futur

    Les observations sont l'état cumulatif z8 (89 positions × 256 canaux), le résumé de la dernière couche au niveau 8 (1 024 valeurs), les probabilités du décodeur au niveau 8, les indices et des statistiques observables. La projection gaussienne des résumés globaux vers 64 dimensions est fixée par la graine. Les positions de z8 correspondent aux tokens ; les cases sont repérées par positions fixes, séparateurs exclus. Les 26 positions de génération du niveau 8 ne sont pas assimilées directement à 81 cellules.

    La baseline contient 71 valeurs locales/globales : 12 probabilités, 12 indicatrices du token actuel, 9 candidats autorisés par les indices, 18 indicatrices ligne/colonne, 20 statistiques globales sans référence. La combinaison y ajoute 256 canaux de z8 local et 64 projections globales. Les états seuls incluent aussi les coordonnées. Le contrôle permute les états de train entre puzzles, en conservant les sorties observables et les cibles initiales.

    Les sondes sont des régressions ridge multi-sorties, standardisées sur train seulement. Alpha dans {10,100,1000} et température softmax dans {0.1,0.25,0.5,1} sont choisis par log-loss validation. Les cibles sont (i) le token prédit au niveau 9, avec 12 classes pour conserver les tokens spéciaux éventuels ; (ii) inchangé, corrigé, dégradé ou erreur remplacée par une autre erreur. Les références ne servent qu'à cette seconde cible et aux métriques. Aucune observation du niveau 9 n'entre dans les prédicteurs.

    Les métriques portent sur les cases initialement vides. Le sous-groupe des cases effectivement changées est défini avec le futur pour l'évaluation ; il ne constitue jamais un oracle disponible à la sonde.

    ## 7. Résultat v1 : gain réel contre une baseline apprise, mais objectif dominé par la persistance
    ''')
    cached=np.load(ROOT/'results/hard.npz');dd=cached['decoded'][:,:,CELL];rr=cached['reference'][:,CELL]
    before=dd[:,7]==rr;after=dd[:,8]==rr;change=dd[:,7]!=dd[:,8]
    counts=dict(corrected=int((~before&after).sum()),degraded=int((before&~after).sum()),wrong_to_wrong=int((~before&~after&change).sum()))
    write(D/'results/revision_transition_counts.json',dict(n_puzzles=2000,n_cells=162000,scope='All 81 cells per official hard puzzle, including givens; descriptive motivation, not probe test metrics',**counts))
    add(f"Le pic initial correspond précisément à **{counts['corrected']:,} cases incorrectes devenant correctes** entre les niveaux 8 et 9 sur les 2 000 puzzles hard (toutes les cases, indices compris). Sur le même passage, {counts['degraded']:,} cases correctes deviennent incorrectes et {counts['wrong_to_wrong']:,} erreurs changent de valeur sans devenir correctes. Ces comptes décrivent les décodages du tokenizer ; ils ne localisent pas un instant où le modèle déciderait consciemment ni ne démontrent une révision explicite de raisonnement. Les évaluations de sondes ci-dessous concernent uniquement les cases initialement vides du test réservé.")
    for name,v in first['results']['next'].items():
        if isinstance(v,dict) and 'all' in v:add(f"- **{name}** : exactitude sur toutes les cases vides {pct(v['all']['accuracy'])}, sur les cases effectivement changées {pct(v['changed_only']['accuracy'])}, log-loss globale {v['all']['logloss']:.4f}.")
    r=first['primary'];second=first['results']['next']['second_decoder_choice_changed_accuracy']
    add(f"Le contraste principal v1 combinaison–baseline sur changements vaut {pp(r['delta'])}, IC {ci(r['ci95'],True)}. Le deuxième choix du décodeur 8 prédit pourtant {pct(second)} des remplacements, bien au-delà des sondes entraînées sur toutes les cases. Le gain v1 ne suffit donc pas à établir une anticipation substantielle des remplacements. Les résultats initiaux sont conservés, sans les remplacer par le complément suivant.")
    for name,v in first['results']['quality'].items():
        add(f"- **Qualité du changement, {name}** : F1 macro {v['all']['macro_f1']:.4f}, AUROC « case corrigée » {v['all']['corrected_auc']:.4f}, log-loss {v['all']['logloss']:.4f}.")
    add('''La classification de la qualité reste limitée et ne démontre pas une direction précise de correction. La F1 macro du prochain token est calculée sur les 12 classes, y compris celles absentes des cibles, et doit être interprétée avec cette convention.

    ![Prédiction du prochain token](figures/03_next_token.png)

    ## 8. Complément conditionnel : remplacement et porte de changement

    Après v1, un amendement exploratoire ajoute six statistiques locales accessibles au niveau 8 : entropie, maximum, deuxième probabilité, marge, probabilité du token actuel et nombre de candidats normalisé. La sonde de remplacement est ajustée uniquement sur les changements de train, et sélectionnée sur les changements de validation. Le token actuel est exclu de la distribution alternative. Une porte binaire prédit séparément le changement sur toutes les cases de train ; sa probabilité combine persistance et distribution alternative pour produire une prévision jointe sans oracle.
    ''')
    for name,v in cond['results'].items():add(f"- **{name}**, test officiel : remplacement conditionnel correct {pct(v['changed_only']['accuracy'])}, prévision jointe toutes cases {pct(v['joint_all']['accuracy'])}, AUROC de la porte {v['gate_auc']:.4f}.")
    for name in ['baseline','state','combined','shuffled_states']:
        v=external[name];add(f"- **{name}**, externe : remplacement conditionnel correct {pct(v['changed_only']['accuracy'])}, prévision jointe toutes cases {pct(v['joint_all']['accuracy'])}, AUROC de la porte {v['gate_auc']:.4f}.")
    for label,r in [('Officiel',cond['contrasts']),('Externe',external['contrasts'])]:
        c=r['combined_minus_fitted_baseline'];s=r['combined_minus_decoder_second']
        add(f"**{label}** : combinaison–baseline ajustée {pp(c['delta'])}, IC {ci(c['ci95'],True)} ; combinaison–deuxième choix brut {pp(s['delta'])}, IC {ci(s['ci95'],True)}.")
    add('''![Évaluation externe](figures/04_external.png)

    La valeur de remplacement est partiellement prévisible et une baseline observable apprise dépasse l'heuristique du deuxième choix brut. En revanche, l'ajout de z8 et des résumés internes n'apporte pas de gain établi face à cette baseline. L'exactitude jointe et l'AUROC de la porte rappellent que connaître le bon remplacement conditionnellement au changement ne revient pas à savoir quelle case changera.

    Il s'agit d'un complément décidé après observation de v1. La baseline a aussi été enrichie : la différence entre les scores v1 et conditionnels ne peut donc pas être attribuée au seul changement du masque d'entraînement. L'évaluation externe n'a servi à aucun ajustement, mais elle avait déjà été consultée et n'est pas une nouvelle confirmation demeurée intacte.

    ## 9. Intervention locale avant le niveau 9

    Sur les 96 premiers puzzles du test fixé, une case initialement vide et un chiffre alternatif sont choisis depuis les seuls prédicteurs pré-9 et la sonde gelée. Pour v1, le score est probabilité du chiffre alternatif moins celle du chiffre actuel. Pour la série conditionnelle, le score est probabilité prédite de changement × probabilité conditionnelle du chiffre alternatif. Ni référence ni transition réelle ne participent à ce choix.

    La direction est la différence des poids ridge du chiffre demandé et du chiffre actuel, restreinte aux 256 canaux z8 locaux et ramenée aux coordonnées non standardisées. Il s'agit d'une direction partielle de la sonde : les autres variables, notamment les probabilités de lecture, sont tenues fixes dans cette construction. La perturbation ajoutée a une norme égale à 0,25 puis 0,5 fois la norme de z8 à la case ciblée ; le complément conditionnel utilise uniquement 0,25. Les directions négatives et trois directions aléatoires par puzzle ont exactement la même norme. Le reste de la génération est poursuivi normalement.

    Pour séparer les chemins, trois lectures du niveau 9 sont comparées :

    1. **Complète** : décodeur(z8 + delta + contribution des nouveaux codes 9).
    2. **Codes seuls** : décodeur(z8 + contribution des nouveaux codes 9), sans delta au décodeur.
    3. **Directe** : décodeur(z8 + delta + contribution des codes 9 natifs), sans régénération de ces codes.

    La deuxième lecture est reconstruite explicitement pour éviter les erreurs d'arrondi d'une soustraction de delta. Elle est un contrôle d'intervention, pas l'identification d'une médiation naturelle unique dans un réseau non linéaire. Les niveaux suivants de la vraie branche conservent la perturbation.
    ''')
    for tag,r in steers.items():
        c=r['primary_indirect_positive_vs_random'];signed=r['secondary_positive_vs_negative']
        add(f"**Direction {tag}** : contraste principal des log-rapports demandé/actuel via les codes seuls, positif–moyenne des aléatoires = {c['mean']:.5f} nat, IC {ci(c['ci95'])}. Positif–négatif = {signed['mean']:.5f}, IC {ci(signed['ci95'])}.")
        for name,v in r['conditions'].items():
            add(f"- {name} : adoption demandée complète {pct(v['full_adoption'])}, codes seuls {pct(v['indirect_adoption'])}, directe {pct(v['direct_only_adoption'])} ; {v['changed_level9_codes']} codes 9 modifiés sur 3 552 ; {v['mean_offtarget_changed_cells']:.3f} autres cases changées en moyenne ; réussite finale {v['final_metrics']['solved']}/96, erreurs introduites {v['introduced_errors']}, corrigées {v['rescued']}.")
    add('''![Contrôles d'intervention](figures/05_steering.png)

    Les intervalles des contrastes principaux recouvrent zéro. Dans la série conditionnelle, l'effet ponctuel de la direction positive sur le log-rapport complet est très proche de celui de la lecture directe, tandis que le gain via les nouveaux codes seuls reste proche de zéro. Cela ne fournit pas de preuve d'une direction de révision spécifique exploitée par le générateur. L'adoption d'un chiffre et l'amplitude de son log-rapport ne sont pas des métriques identiques : l'une est discrète, l'autre peut varier sans franchir l'argmax.

    Le nombre de changements des codes 9 est faible ; les quantificateurs peuvent être insensibles aux petites perturbations. Les succès finaux restent inchangés sur ces 96 puzzles, mais cela ne démontre ni absence de rôle causal de toute représentation de révision ni équivalence universelle. Les effets peuvent être distribués, non linéaires ou mal alignés avec la direction linéaire choisie. La dose 0,5 et les comparaisons secondaires ne constituent pas des recherches corrigées d'une dose optimale.

    ## 10. Ce que nous avons établi et ce qui reste ouvert

    - Une invariance des moyennes d'embeddings à neuf positions est expliquée par la sérialisation des lignes complètes de Sudoku. Elle se retrouve avec les trois tokenizers examinés et ne constitue pas une propriété universelle de NCM.
    - Le résidu avant quantification du niveau 5 reste variable, et un k-means local en exploite davantage. Ce gain géométrique n'établit pas un gain de résolution après modification des codebooks.
    - L'usage du niveau doit être distingué entre entrées propres et régime de dropout, sans assimiler EMA positive et fréquence actuelle ni en déduire une trajectoire d'apprentissage.
    - Sur les 856 puzzles hard testés, la prédiction du niveau 5 peut être remplacée par un code constant en conservant exactement les réponses. Cela n'autorise pas à supprimer sans effet sa contribution ni à annoncer un gain de vitesse proportionnel au nombre de niveaux.
    - Les probabilités déjà lisibles au niveau 8 prédisent une partie des valeurs de remplacement au niveau 9. Le bénéfice supplémentaire des états examinés n'est pas établi contre la baseline ajustée.
    - Aucune direction causale spécifique de révision ni contrôle métacognitif natif n'est démontré. Les résultats négatifs portent sur les sondes, positions, doses et contrôles effectivement testés.

    Pour expliquer historiquement pourquoi les codebooks ont convergé ainsi, il manque des checkpoints d'entraînement, les logs et une réplication fidèle de la recette complète. Pour renforcer l'étude des révisions, il faudrait des données réellement nouvelles préservées avant choix de la méthode, davantage de graines, des lectures plus structurées ou non linéaires et des interventions localisant le chemin de génération. Aucun petit entraînement substitutif ni benchmark GPU n'a été réalisé ici.

    ## 11. Reproduction et preuves

    Depuis ce sous-dossier, reproduction complète (longue) :

    ```sh
    ./run.sh
    ```

    Vérification des artefacts sauvegardés :

    ```sh
    PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" ../.venv/bin/python scripts/verify.py
    ```

    L'environnement Python et les poids sont ceux de la campagne précédente ; [requirements.txt](../requirements.txt) et [verrou complet](../sources/requirements-lock.txt) documentent les versions. La reproduction suppose de conserver l'arborescence avec ../artifacts, ../data, ../vendor, ../scripts et ../results. Les caches sont confinés dans ce sous-dossier. Les calculs concurrents initiaux ne servent pas à mesurer la vitesse.

    **Sources** : scripts/base.py ; runtime.py ; diagnose_level5.py ; pooling_algebra.py ; dropout_control.py ; archive_centers.py ; level5_interventions.py ; extract_revision.py ; probes.py ; conditional.py ; steer_revision.py ; report.py/reporting.py ; verify.py ; continue.py ; run.sh. Le runner d'origine importé est ../vendor/ncm_mac et les utilitaires communs sont ../scripts/common.py. Tous les scripts de cette campagne et les dépendances de modèle copiées sont inclus dans sources_directions_mecanistes.zip ; les gros poids et traces restent dans leurs dossiers.

    **Artefacts inspectables** : results/level5_diagnostic.json et level5_geometry.npz ; pooling_algebra.json ; dropout_control.json/.npz ; kmeans_centers.npz ; level5_interventions.json/.npz ; revision_official.npz et revision_external.npz ; modèles et prédictions de probes et conditional ; steering.json/.npz et steering_conditional.json/.npz. Les CSV d'inspection et les figures PNG/SVG sont générés avec le rapport. Les références utilisées comme cibles sont conservées séparément des variables pré-9.

    **Traçabilité** : results/protocol_frozen.json ; final_manifest.json et verification.json ; JOURNAL.md et logs/. L'assemblage automatique initial a précédé la création du script de rendu et a échoué ; le rendu final est exécuté après tous les compléments, sans perdre ni réajuster les résultats v1.

    Référence scientifique : [article NCM](https://arxiv.org/html/2610.14162v1), §3 pour le tokenizer résiduel et le générateur ; les résultats et contrôles rapportés ici sont calculés localement. Les observations de cette étude concernent les poids publiés effectivement copiés, pas toutes les variantes NCM.
    ''')
    (D/'RAPPORT.md').write_text('\n'.join(text))
    # Flat summaries for review without loading model pickles.
    with (D/'results/conditional_summary.csv').open('w') as out:
        w=csv.writer(out);w.writerow(['corpus','features','changed_accuracy','joint_accuracy','gate_auc'])
        for corpus,rs in [('official',cond['results']),('external',external)]:
            for name in ['baseline','state','combined','shuffled_states']:
                r=rs[name];w.writerow([corpus,name,r['changed_only']['accuracy'],r['joint_all']['accuracy'],r['gate_auc']])
    with (D/'results/steering_summary.csv').open('w') as out:
        w=csv.writer(out);w.writerow(['series','condition','full_adoption','indirect_adoption','direct_adoption','final_solved','changed_codes9'])
        for tag,r in steers.items():
            for name,v in r['conditions'].items():w.writerow([tag,name,v['full_adoption'],v['indirect_adoption'],v['direct_only_adoption'],v['final_metrics']['solved'],v['changed_level9_codes']])
    with zipfile.ZipFile(D/'sources_directions_mecanistes.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in (D/'scripts').glob('*.py'):z.write(path,str(path.relative_to(ROOT)))
        for path in D.glob('*.md'):z.write(path,str(path.relative_to(ROOT)))
        for name in ['protocol.json','run.sh']:z.write(D/name,str((D/name).relative_to(ROOT)))
        z.write(ROOT/'scripts/common.py','scripts/common.py');z.write(ROOT/'requirements.txt','requirements.txt')
        for name in ['protocol.json','results/splits.json','sources/requirements-lock.txt']:
            z.write(ROOT/name,name)
        for path in (ROOT/'vendor/ncm_mac').rglob('*'):
            if path.is_file() and '__pycache__' not in str(path):z.write(path,str(path.relative_to(ROOT)))
        for path in (ROOT/'artifacts/sudoku').glob('*/config.yaml'):z.write(path,str(path.relative_to(ROOT)))
    log('report_complete',words=len((D/'RAPPORT.md').read_text().split()),figures=6)

if __name__=='__main__':main()
