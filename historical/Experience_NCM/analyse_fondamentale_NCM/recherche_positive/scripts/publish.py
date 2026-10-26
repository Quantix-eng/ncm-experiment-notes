"""Publish measured results without fitting, selecting, or changing the frozen policy."""
from base import *
import subprocess,zipfile

def read(n):return json.loads((D/'results'/n).read_text())
def pct(x):return f'{100*x:.2f}'
def interval(x,percent=True):
    return '['+', '.join(pct(v) if percent else f'{v:.4f}' for v in x)+']'

def pilot_format_audit():
    a=np.load(D/'results/discovery_pilot.npz');old=np.load(ROOT/'results/hard.npz');pred=a['terminal_only'];ref=old['reference'][a['indices']]
    write(D/'results/pilot_format_audit.json',dict(status='post_hoc_metric_clarification',n=len(pred),exact_89_tokens=int(np.all(pred==ref,1).sum()),exact_81_cells=int(np.all(pred[:,CELL]==ref[:,CELL],1).sum()),exact_8_separators=int(np.all(pred[:,SEP]==ref[:,SEP],1).sum()),reference_cell_accuracy=float((pred[:,CELL]==ref[:,CELL]).mean()),explanation='The frozen pilot metric counted all 89 tokens. Zero complete outputs does not mean zero grids with all 81 digits correct. The pilot and primary protocol hashes are unchanged.'))

def figures(t,w,rr,b):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','svg','pdf']:fig.savefig(D/'figures'/f'{name}.{ext}',bbox_inches='tight')
        plt.close(fig)
    cnt=np.array(t['hard']['train_counts'])[:,1:10];prob=cnt/np.maximum(cnt.sum(1)[:,None],1)
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(10,4.0),gridspec_kw={'width_ratios':[3,1]})
    im=ax.imshow(prob,aspect='auto',vmin=0,vmax=1,cmap='Blues');ax.set(xticks=np.arange(9),xticklabels=np.arange(1,10),yticks=np.arange(16),yticklabels=[f'{i}  (n={n})' for i,n in enumerate(cnt.sum(1))],xlabel='Native decoded digit',ylabel='Terminal code ID (training support)',title='Train-only terminal readout')
    fig.colorbar(im,ax=ax,label='Conditional frequency',fraction=.035)
    ax2.axis('off');ax2.text(.05,.95,'Writing alphabet\n(digit : code ID)\n\n'+'\n'.join(f'{i+1} : {v}' for i,v in enumerate(t['hard']['alphabet_by_digit'])),va='top',fontsize=11)
    save(fig,'01_terminal_alphabet')
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(10,3.0),gridspec_kw={'width_ratios':[2,1]})
    x=np.arange(9);r=w['per_digit'];p=np.array([v['hits']/v['trials'] for v in r]);rp=np.array([v['random_hits']/v['trials'] for v in r])
    ax.bar(x-.18,100*p,.36,label='Code substitution',color='#2166ac');ax.bar(x+.18,100*rp,.36,label='Norm-matched Gaussian',color='#d95f02');ax.set(xticks=x,xticklabels=np.arange(1,10),ylim=(0,105),xlabel='Requested digit',ylabel='Target writing (%)');ax.legend(fontsize=8,loc='upper left')
    for i,v in enumerate(r):ax.text(i,100*p[i]+2,f"{v['hits']}/{v['trials']}",ha='center',fontsize=7)
    ax2.axis('off');ax2.text(0,.95,f"Reserved test: 128 puzzles\n784 / 1024 target writes\n0 / 1024 random writes\n\nPaired advantage:\n{pct(w['paired_advantage']['mean'])} percentage points\n97.5% CI {interval(w['paired_advantage']['ci_familywise'])}\n\nOther-cell changes:\n0 observed after substitutions",va='top',fontsize=10)
    save(fig,'02_causal_writing')
    fig,(ax,ax2)=plt.subplots(1,2,figsize=(10,3.0));names=['hard','easy','medium'];x=np.arange(3)
    v=np.array([rr[n]['coverage']['mean'] for n in names]);ci=np.array([rr[n]['coverage']['ci95'] for n in names]);ax.bar(x,100*v,color=['#2166ac','#67a9cf','#92c5de']);ax.errorbar(x,100*v,yerr=100*np.array([v-ci[:,0],ci[:,1]-v]),fmt='none',color='black',capsize=3);ax.axhline(30,color='gray',ls='--',label='Hard: development-selected 30% minimum');ax.set(xticks=x,xticklabels=['hard (1024)','easy (256)','medium (256)'],ylim=(0,70),ylabel='Certified decoder bypass (%)');ax.legend(fontsize=7)
    a=np.array([rr[n]['native']['solved_rate'] for n in names]);h=np.array([rr[n]['hybrid']['solved_rate'] for n in names]);ax2.bar(x-.18,100*a,.36,label='Native',color='#999999');ax2.bar(x+.18,100*h,.36,label='Certified hybrid',color='#2166ac');ax2.set(xticks=x,xticklabels=names,ylim=(0,100),ylabel='Exact success (%)');ax2.legend(fontsize=8);ax2.set_title('All checkpoints evaluated on new 30-clue puzzles',fontsize=9)
    save(fig,'03_certified_readout')
    fig,axes=plt.subplots(1,2,figsize=(10,2.8))
    for ax,name,title in zip(axes,['end_to_end','readout_only'],['Full pipeline: all 12 generator levels','Readout only: cached codes and states']):
        values=[b[name]['mean_native_ms'],b[name]['mean_certified_ms']]
        ax.bar([0,1],values,color=['#999999','#2166ac'],width=.55)
        ax.set(xticks=[0,1],xticklabels=['Native','Certified hybrid'],ylabel='Arithmetic mean latency (ms)',title=title,ylim=(0,max(values)*1.2))
        for i,value in enumerate(values):ax.text(i,value+max(values)*.025,f'{value:.3f} ms',ha='center',fontsize=10)
    save(fig,'04_timing')

def report(t,w,rr,b):
    h=rr['hard'];e=rr['easy'];m=rr['medium'];speed=b['end_to_end'];stage=b['readout_only'];raw=read('benchmark_raw.json');q=read('discovery_pilot.json')
    fr=f'''# Un alphabet terminal partiel et une lecture certifiée du NCM Sudoku

Étude et révision critique du 24 octobre 2026. Trois checkpoints Sudoku publiés, CPU FP32, un thread. [Article anglais](PAPER.md), [PDF](PAPER.pdf), [protocole local conservé](PROTOCOLE_GELE.md), [journal](JOURNAL.md), [note de révision](REVISION_EDITORIALE.md). Les sept campagnes antérieures restent intactes. La présente révision modifie la présentation, sans nouvelles expériences ni changement des données, métriques ou critères initiaux.

## Observations établies et portée des résultats

Sur 128 nouveaux puzzles, une substitution d'un code terminal écrit le chiffre demandé dans **784/1 024 essais = 76,56 %**, contre **0/1 024** pour des directions gaussiennes de même norme à chaque position latente. L'avantage apparié vaut **76,56 points**, IC 97,5 % **{interval(w['paired_advantage']['ci_familywise'])}**. Aucun argmax des 80 autres cases ne change dans ces essais ; ce critère ne mesure pas l'invariance des logits ou des séparateurs. Il s'agit d'un effet d'édition pour les codes, contextes et doses testés. Le contrôle gaussien ne suffit pas à établir une spécificité conceptuelle ou un mécanisme nouveau du NCM.

Une table de lecture apprise sur les anciens codes et sorties natives de train produit une grille certifiée valide pour **552/1 024 nouveaux puzzles = 53,91 %**, IC 97,5 % **{interval(h['coverage']['ci_familywise'])}**. Le décodeur final peut être évité pour ces entrées ; si le certificat échoue, la politique appelle exactement le décodeur natif. La garantie de non-dégradation dépend de l'unicité et du vérificateur externe ; elle n'est pas une propriété apprise découverte du NCM.

Les deux règles de décision annoncées sont satisfaites, mais elles ne constituent pas un verdict sur l'importance ou la nouveauté scientifique. Le seuil de couverture de 30 % a été choisi après une validation favorable (201/400 = 50,25 %), sans justification indépendante d'utilité minimale. Le bilan défendable est une **édition terminale mesurée et une couverture de lecture certifiée sur ces checkpoints**. Une amélioration populationnelle de résolution, une accélération globale, une explication des révisions intermédiaires et une nouveauté mécaniste ne sont pas établies.

## 1. Pourquoi cette voie, et ce qui a été fixé

Les analyses de développement conservées utilisent les anciens train (1 000 puzzles) et validation (400). Pour chaque identifiant parmi 16, nous comptons les chiffres natifs décodés à la même position au dernier niveau, sur les 81 cases. La table associe l'identifiant au chiffre 1..9 le plus fréquent. L'alphabet d'écriture choisit, pour chaque chiffre, l'identifiant ayant le plus grand nombre d'occurrences de ce chiffre sur train. Aucun poids NCM n'est entraîné ; les références de solution ne servent pas à ajuster ces tables.

La validation hard montre 99,38 % d'accord par case, mais seulement 201/400 lectures certifiables. Le pilote causal sur 32 anciennes entrées validation réussit 208/256 écritures (81,25 %). Le décodage du **seul résidu terminal** conserve {pct(q['terminal_only_cell_agreement'])} % d'accord avec les chiffres natifs mais **0/32 grilles exactes**. Ces résultats ont motivé la notion d'alphabet **partiel**, avant les nouveaux tests. Le dernier niveau ne constitue pas une solution complète autonome.

Le gel intervient le **24 octobre 2026 à 21:53:58 UTC**. `protocol.json`, `PROTOCOLE_GELE.md`, les tables et le pilote sont épinglés par SHA-256 dans `results/protocol_frozen.json`. La direction est sélectionnée après exploration ; les nouveaux tests évaluent une hypothèse et une règle figées, sans rendre cette sélection exploratoire confirmatoire rétrospectivement.

### Sélection de la question et limites du gel

Cette direction a été choisie dans une recherche explicitement motivée par l'obtention d'un résultat positif, après plusieurs campagnes peu favorables à nos hypothèses initiales. Le pilote révélait déjà des écritures fréquentes et environ 50 % de lectures certifiables. La sélection d'une piste prometteuse sur développement, suivie d'un nouveau test dont les règles restent fixes, n'invalide pas à elle seule ce test. Elle doit cependant être rendue visible ; satisfaire des critères choisis dans ce contexte ne démontre pas une avancée fondamentale.

Il s'agit d'un gel local, **ni prérégistration indépendante ni revue externe**. Les scripts expérimentaux n'étaient pas inclus dans les empreintes initiales. Il n'existe pas non plus de registre externe, complet et non réinscriptible de toutes les commandes exploratoires. Les règles appliquées correspondent au protocole conservé et les journaux placent son gel avant la génération des nouveaux puzzles ; ces fichiers ne prouvent pas à eux seuls toute la chronologie ni l'absence absolue de p-hacking. Les empreintes actuelles attestent l'intégrité des fichiers présents, pas leur validité scientifique. Le protocole historique est conservé sans réécriture rétroactive.

## 2. Données, méthode et inférence

1 024 solutions sont générées après le gel par backtracking aléatoire, puis retrait glouton jusqu'à 30 indices. L'unicité est contrôlée par MRV et Algorithm X. Les clés de solutions normalisées par chiffres et huit symétries D4 sont exclues contre les corpus officiels, les deux corpus externes antérieurs et les nouvelles familles déjà retenues. Cet audit est incomplet sur les isomorphies Sudoku ; l'entraînement inconnu peut contenir des parents. Le générateur de puzzles n'est pas uniforme et 30 indices ne définissent pas à eux seuls une difficulté humaine.

Les deux critères sont l'avantage moyen d'écriture par puzzle au-dessus de zéro et la couverture certifiée hard au-dessus de 30 %. Bootstrap percentile de 4 000 tirages par puzzle, IC à **97,5 %** pour les deux critères (Bonferroni), à 95 % pour les descriptions. Les huit cibles éligibles, les contrôles et les répétitions restent groupés dans chaque puzzle. Ces IC restent approximatifs et conditionnels aux poids et à la table fixés ; ils ne couvrent pas la variation d'un réentraînement.

Les autres résultats, dont les différences de réussite, les chiffres particuliers, easy/medium et les temps, sont secondaires. L'effectif est un budget de précision fixé ; il n'y a pas de recherche séquentielle d'un résultat favorable sur le nouveau corpus. Toutes les erreurs sont conservées.

## 3. Édition terminale : association, intervention et limites du contrôle

Le dernier niveau comporte 89 positions, dont 81 cases et huit séparateurs. Nous exploitons cet alignement terminal ; il n'existe pas pour les 37 positions du niveau 9. Sur les nouveaux puzzles hard, la lecture par table reproduit **{pct(h['cell_agreement_lookup'])} % des chiffres du décodeur natif**. Il s'agit d'un accord avec la sortie du modèle, pas d'une exactitude envers la solution : la lecture seule atteint {pct(h['lookup_only']['cell_accuracy'])} % de cases correctes.

![Association identifiants/chiffres sur train ; les effectifs évitent de surinterpréter les codes rares.](figures/01_terminal_alphabet.png)

Pour le test causal, une case initialement vide est tirée uniformément dans chacun des 128 premiers nouveaux puzzles, indépendamment de la prédiction et des effets. Nous testons tous les chiffres, sauf celui lu avant intervention : 8 cibles par source, soit 1 024 essais. Tous les niveaux et codes autres que le code terminal sélectionné restent fixes. L'état testé est `z_final + R12(c12 modifié) - R12(c12 natif)` ; aucune continuation n'est requise après le dernier niveau. Le sham reproduit la lecture native.

La transformation résiduelle terminale inclut une convolution de largeur trois. La substitution modifie donc jusqu'à trois positions latentes. Le contrôle aléatoire utilise exactement ces positions et, à chacune, une direction gaussienne indépendante de même norme L2. Les normes et le support sont sauvegardés et vérifiés. Ce contrôle apparie l'amplitude locale ; il ne reproduit pas la distribution des différences entre embeddings de codebook.

![Écritures causales sur les nouvelles données ; les résultats par chiffre sont descriptifs.](figures/02_causal_writing.png)

Le contraste mesure l'efficacité de substitutions choisies par leur association au chiffre demandé, face à un bruit isotrope de même amplitude locale. Les directions apprises du codebook et les directions gaussiennes n'ont pas la même géométrie ; le témoin reste donc limité. Un avantage peut refléter une lecture attendue d'embeddings entraînés, sans établir un circuit de raisonnement ou une propriété spécifique à NCM. Des contrôles par codes valides avec étiquettes permutées, d'autres positions et un codec témoin seraient nécessaires pour tester ces interprétations ; ils ne font pas partie des critères prospectifs de cette campagne.

Les 76,56 % d'écritures et les {pct(h['cell_agreement_lookup'])} % d'accord prédictif portent sur des interventions et des populations de positions différentes : leur écart n'est pas une estimation contrôlée de « l'effet du contexte ». Les écritures ratées montrent néanmoins que la règle testée « un identifiant suffit à imposer le chiffre » n'est pas exacte. Les 240 échecs sont conservés. Leur réanalyse descriptive, post-hoc, distingue **237 demandes laissant la case inchangée et trois donnant un autre chiffre**. Ce résultat ne localise pas la dépendance au contexte dans le décodeur, les voisins ou les niveaux antérieurs. Les perturbations aléatoires n'écrivent jamais la cible ; elles modifient toutefois en moyenne 0,0703 autre case par essai éligible.

Zéro dommage collatéral observé ne prouve pas un taux populationnel nul. Le bootstrap de ce nombre nul serait dégénéré ; 0/128 sources avec au moins un dommage correspond à une borne Wilson supérieure descriptive à 95 % d'environ 2,91 %. Les 1 024 essais ne sont pas 1 024 sources indépendantes. Modifier un chiffre pour en demander un autre rend généralement le Sudoku faux : **ce test d'écriture n'est pas une méthode d'amélioration de la résolution**.

## 4. Application : lecture directe avec certificat et fallback exact

La politique n'accède qu'aux codes générés, au puzzle et à la table train. Elle construit 81 chiffres, fixe les séparateurs, puis vérifie les 27 lignes/colonnes/blocs, les indices et le format. Si tout passe, elle retourne la grille ; sinon elle appelle le décodeur natif sur le même latent singleton. Le solveur d'unicité et la référence ne sont **jamais** appelés par cette politique. Le code déployable est `scripts/policy.py` ; les deux solveurs servent seulement à préparer et contrôler les données.

**Garantie conditionnelle.** Soit p un puzzle à solution unique s et V un vérificateur sain. Toute grille g acceptée par V est s. Si la sortie native vaut s, la politique retourne soit une grille certifiée, donc s, soit exactement la sortie native s. Le succès exact ne peut donc pas diminuer point par point sous ces préconditions. Cette propriété ne dépend pas de la précision de la table. Si plusieurs solutions sont possibles, elle préserve la validité avec indices, pas nécessairement une référence particulière. L'unicité n'est pas gratuite à certifier à l'exécution : ici, c'est une précondition des données et de la garantie.

Cette démonstration concerne la politique de vérification et de repli. Elle s'appliquerait à un autre générateur de candidats sous les mêmes préconditions ; elle n'apporte donc pas, à elle seule, une preuve sur le fonctionnement interne du NCM. Les sorties rejetées par le certificat peuvent rester incorrectes après le repli natif.

Sur les 1 024 puzzles hard : **883 grilles exactes nativement, 885 avec la politique**, zéro échec introduit, deux corrigés. Les 552 certificats sont tous corrects. Le gain secondaire de 0,195 point, IC 95 % **{interval(h['quality_delta']['ci95'])}**, n'établit pas une amélioration populationnelle stricte. Le résultat pratique principal est la couverture de 53,91 %, adossée à la garantie conditionnelle, et non ce petit gain.

![Couverture et qualité des trois checkpoints sur les mêmes nouvelles familles de puzzles.](figures/03_certified_readout.png)

Sur les 256 premières familles communes, le checkpoint easy certifie **{e['certified']}/256 = {pct(e['coverage']['mean'])} %**, IC 95 % {interval(e['coverage']['ci95'])}, et passe de {e['native']['solved']} à {e['hybrid']['solved']} grilles exactes. Medium certifie **{m['certified']}/256 = {pct(m['coverage']['mean'])} %**, IC 95 % {interval(m['coverage']['ci95'])}, et passe de {m['native']['solved']} à {m['hybrid']['solved']}. Aucune régression dans les deux cas. Ce sont trois poids distincts et trois tables distinctes ; les données sont corrélées, et ces puzzles à 30 indices ne sont pas les corpus easy/medium officiels. Les taux ne doivent pas être présentés comme leurs performances usuelles sur leurs difficultés d'origine.

## 5. Coût réellement mesuré

Le benchmark utilise les **64 premiers puzzles**, dont {pct(raw['acceptance_first64'])} % sont certifiés, sans sélection sur acceptation. CPU FP32, un thread, quatre puzzles de chauffe exclus, trois paires complètes par puzzle, ordres AB/BA et ordre des puzzles randomisés. Aucune autre inférence de campagne ne s'exécute pendant cette mesure. Les deux bras exécutent les douze niveaux du même générateur natif ; seul le décodage peut être évité. Les sorties sont contrôlées dans les deux bras.

- **Pipeline complet** : moyenne arithmétique **{speed['mean_native_ms']:.3f} ms native, {speed['mean_certified_ms']:.3f} ms hybride**. Le contraste géométrique apparié annoncé vaut {speed['geometric_speed_ratio']:.4f}, IC 95 % {interval(speed['geometric_speed_ratio_ci95'],False)} : l'analyse prévue n'établit pas une accélération globale.
- **Lecture seule**, seize répétitions internes : moyenne arithmétique **{stage['mean_native_ms']:.3f} ms native, {stage['mean_certified_ms']:.3f} ms hybride**, soit un rapport des moyennes de **{stage['mean_native_ms']/stage['mean_certified_ms']:.2f}** et une réduction descriptive de **{pct(1-stage['mean_certified_ms']/stage['mean_native_ms'])} %** pour cette étape et cet échantillon. Aucun IC de différence des moyennes arithmétiques n'a été annoncé ou estimé ici.

L'inférence temporelle initiale rééchantillonne les puzzles après moyenne des log-rapports des répétitions. Pour la lecture seule, ce rapport géométrique vaut 9,2117, IC 95 % [5,8356 ; 14,4797] ; il reste conservé comme statistique complémentaire. Il ne signifie ni une division par neuf du coût moyen de lecture, ni une accélération par neuf du NCM complet. Une branche très rapide sur les entrées certifiées et légèrement plus lente en fallback explique l'écart avec le quotient des moyennes arithmétiques. Les distributions et mesures brutes sont conservées. Les IC sont conditionnels à cette séance et ce matériel ; ils n'incluent ni variation entre machines ni séances indépendantes. Le taux de suppression des appels au décodeur n'est pas le taux d'économie du calcul complet. Aucun gain GPU, de batching, de mémoire ou d'énergie n'est établi.

![Moyennes arithmétiques observées, avec des axes de latence distincts : pipeline complet et lecture seule. Cette figure n'ajoute pas d'IC aux moyennes ; l'analyse appariée annoncée est décrite dans le texte.](figures/04_timing.png)

## 6. Apport, antécédents et limites

L'étude caractérise une association terminale et des interventions locales sur un checkpoint, puis une politique de lecture pour trois checkpoints. La nouveauté scientifique de cette observation n'est pas établie. Un codebook terminal aligné avec les positions et entraîné à reconstruire des chiffres pourrait présenter ce comportement sans mécanisme nouveau ou spécifique à NCM ; aucune comparaison d'architectures ou de codecs témoins ne teste cette possibilité ici.

Le principe général de sélection/fallback et celui du calcul adaptatif ne sont pas nouveaux ; [SelectiveNet](https://arxiv.org/abs/1901.09192) et [CALM](https://arxiv.org/abs/2207.07061) fournissent des antécédents distincts. Notre politique utilise une vérification externe exacte pour cette tâche et conserve tous les niveaux du générateur. Nous ne revendiquons ni première mondiale ni nouvelle théorie de la vérification.

Les résultats antérieurs ne sont pas remplacés : le codec suffit à produire des reconstructions non monotones ; la compensation conditionnelle spécifique à l'omission 9 n'a pas été établie. L'édition terminale n'explique pas pourquoi les révisions intermédiaires se produisent. Elle ne localise ni têtes d'attention ni règles Sudoku humaines, et ne démontre aucune métacognition native. Une réplication indépendante avec code et protocole préalablement scellés, contrôles de codes et comparaison à un autre codec serait nécessaire pour renforcer ces affirmations ; ces expériences restent à faire, sur des données non encore examinées.

Limites : un checkpoint pour la causalité, trois pour la lecture ; aucune réplication texte ; table fixe et une graine de génération des nouveaux puzzles ; contrôle gaussien qui ne reproduit pas la distribution des codes ; interventions potentiellement hors distribution ; une seule case tirée par source ; aucun test de substitution de plusieurs cases ni de séparateurs ; unicité nécessaire à la garantie de fidélité à la référence ; cadence et mémoire dépendantes du backend. Les gains de réussite restent petits et non confirmés, même si la non-dégradation est démontrable sous les préconditions du certificat.

## 7. Preuves et reproduction

Depuis ce dossier, `./run.sh` régénère les documents et vérifie les artefacts conservés, avec rejeu, sans remplacer les mesures historiques. Pour recalculer la campagne complète dans un **nouveau dossier frère**, utiliser `PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" ../.venv/bin/python scripts/reproduce.py --destination ../reproduction_alphabet` ; le script emploie les sources historiques conservées et les mêmes entrées. Ce recalcul n'est pas une réplication indépendante ; ses documents reprennent la rédaction antérieure, signalée comme telle, et ses temps peuvent varier. L'interprétation révisée de référence reste celle du présent dossier. Pour auditer seulement les artefacts et rejouer 16 puzzles : `PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" ../.venv/bin/python scripts/verify.py`. L'environnement est décrit dans `sources/environment.json` et le verrou antérieur inclus dans l'archive. Poids et données antérieures restent en lecture dans le parent.

Le compteur de `results/verification.json` recense des assertions logicielles et des vérifications de données : **4 096 correspondent à quatre propriétés contrôlées pour chacun des 1 024 puzzles**. Les autres contrôles portent notamment sur les statistiques, des rejeux et l'intégrité. Le script est distinct du runner, mais écrit et exécuté par le même agent de recherche. Il ne s'agit ni de milliers de réplications scientifiques ni d'une revue indépendante. Des tests réussis ne prouvent pas l'absence de biais, de p-hacking ou d'erreur d'interprétation.

La révision critique conserve les versions précédentes dans `sources/avant_revision_editoriale_20261024.zip`. `results/editorial_revision_baseline.json` épingle les fichiers scientifiques avant correction du texte ; `results/editorial_revision_audit.json` vérifie leur invariance. Les fichiers de publication et l'archive de sources sont actualisés, pas le protocole historique ou les observations.

`data/fresh.json` conserve les puzzles et solutions, `results/*_readout.npz` les codes, latents, sorties et certificats ; `writing_interventions.npz` les neuf cibles, témoins, normes locales et dommages ; `benchmark_raw.json` chaque durée et chaque ordre. Les scripts importés, configurations et protocoles sont dans `sources_recherche_positive.zip`, sans gros poids. Les empreintes des entrées et des sept manifestes précédents, le gel et le manifeste final relient les preuves. `results/verification.json` décrit les contrôles réellement passés ; `results/pdf_inspection.json` consigne le rendu. Toutes les figures sont exportées en PNG, SVG et PDF.

Référence du modèle : [NCM, article et code publiés](https://arxiv.org/html/2610.14162v1), [dépôt officiel](https://github.com/HazyResearch/ncm). Les chiffres de ce rapport proviennent des expériences locales.
'''
    fr=fr.replace("Le taux de suppression des appels au décodeur n'est pas le taux d'économie du calcul complet.","L'IC du pipeline complet recouvre 1 : une accélération globale n'est donc pas établie par cette séance. Le taux de suppression des appels au décodeur n'est pas le taux d'économie du calcul complet.")
    numeric=read('numerical_support_audit.json')
    fr=fr.replace("Ce contrôle apparie l'amplitude locale ; il ne reproduit pas la distribution des différences entre embeddings de codebook.",f"Ce contrôle apparie l'amplitude locale ; il ne reproduit pas la distribution des différences entre embeddings de codebook. L'audit a détecté des queues numériques FP32 hors du voisinage mathématique de trois positions : norme maximale par position {numeric['maximum_tail_position_norm']:.3g}. Les témoins apparient aussi ces queues. Un contrôle post-hoc calcule directement la différence affine en FP64 puis revient en FP32 pour le décodeur : aucune lecture complète ne change parmi les 1 152 demandes (chiffre natif compris) dans aucun bras. Le test principal gelé n'est pas remplacé par ce diagnostic ; détails dans `results/numerical_support_audit.json`.")
    fr=fr.replace("Ces résultats ont motivé la notion d'alphabet **partiel**, avant les nouveaux tests. Le dernier niveau ne constitue pas une solution complète autonome.","Le zéro du pilote porte sur les **89 tokens, séparateurs compris**. Un audit post-hoc distingue **10/32 grilles ayant leurs 81 chiffres exacts**, mais **0/32 ayant leurs huit séparateurs exacts** ; exactitude des cases envers la référence : 96,18 %. Le résidu terminal ne constitue donc pas un lecteur autonome fiable ; le zéro global n'était pas une preuve d'incapacité à reconstruire les chiffres de toute grille. La métrique pilote et les critères réservés restent gelés, avec cette clarification dans `results/pilot_format_audit.json`.")
    (D/'RAPPORT.md').write_text(fr)
    return h,e,m,speed,stage,raw

def paper(t,w,rr,b,h,e,m,speed,stage,raw):
    text=f'''---
title: "A Partial Terminal Alphabet and Certified Decoder Bypass in Sudoku NCM"
date: "Experimental report - 24 October 2026"
fontsize: 10pt
geometry: margin=22mm
colorlinks: true
header-includes:
  - \\usepackage{{float}}
  - \\floatplacement{{figure}}{{H}}
  - \\setlength{{\\parskip}}{{3pt plus 1pt minus 1pt}}
---

# Abstract

We report terminal-code interventions and a certified readout policy for frozen Sudoku Next-Concept Models. A favorable development pilot guided selection of the question and a local protocol; this was not an independent preregistration. On 128 reserved puzzles, replacing one terminal code writes a requested different digit in 784/1,024 trials (76.56%), versus 0/1,024 locally norm-matched Gaussian controls; the source-clustered 97.5% confidence interval for the paired advantage is {interval(w['paired_advantage']['ci_familywise'])} percentage points. No other cell argmax changes in these substitutions. Gaussian directions do not match codebook geometry, limiting mechanistic interpretation. On 1,024 new unique-solution puzzles, train-only lookup yields 552 certified grids (53.91%; 97.5% CI {interval(h['coverage']['ci_familywise'])}). The 30% coverage criterion was chosen after 50.25% development coverage and lacks an independently justified utility threshold. Both announced criteria are met, but that is not a verdict on scientific importance. The fallback guarantee follows from external verification and uniqueness, independently of NCM internals. Native/hybrid success is 883/885 out of 1,024, without an established population accuracy gain. Mean readout latency falls from 1.091 to 0.493 ms in one CPU session; end-to-end acceleration remains uncertain. Novelty, architecture specificity and an explanation of intermediate revisions remain unestablished.

# Question and contribution

Earlier campaigns found non-monotone reconstructions even when the codec directly encoded correct solutions, refuting a strict decoded-value lock at level 9. A crossed omission experiment did not establish useful, perturbation-specific compensation by later levels. These results motivate a different mechanistic question: what information is present in the final discrete codes, and can that information be manipulated and used without changing model weights?

The final Sudoku level has 89 positions: 81 cells and eight separators. Unlike the 37-position ninth level, it is directly aligned with serialized cells. We distinguish three claims: a code predicts a digit observationally; substituting the code controls the digit causally; and a policy exploiting that association has useful coverage without loss of correctness. None follows automatically from another. We test them separately and retain failed terminal-only reconstruction and uncertain full-pipeline timing.

We report a local empirical characterization of a digit association, code substitution and certified readout for these published Sudoku models. A trained terminal codebook might be expected to carry digit information. No comparison establishes that the observed behavior is novel or specific to NCM. It does not explain intermediate revisions, locate a reasoning circuit or establish an architecture-wide law. This critical editorial revision preserves all original data, numerical outcomes and announced rules; prior versions and an integrity audit are retained.

# Development, freeze and reserved data

For each checkpoint, we use only the original 1,000 training-partition puzzles to count pairs of generated terminal code IDs and same-position native decoded digits. For each of 16 IDs, the readout table selects its most frequent digit among 1 through 9. For each digit, the writing alphabet selects the ID with the largest training count for that digit. References are not used to fit either table. Ties use the first ID/digit in numerical order. Each checkpoint has a separate table.

Old validation data guide direction selection. Hard validation cell agreement is 99.38%, with 201/400 certifiable lookup grids. A 32-puzzle writing pilot succeeds in 208/256 trials (81.25%). Decoding the terminal residual alone agrees with native digits on 98.19% of cells but solves 0/32 complete grids. The latter result argues against interpreting the last residual as an independently complete answer.

The local freeze is dated 2026-10-24 21:53:58 UTC. Protocol, tables and pilot are hashed before recorded reserved data generation. The two announced co-primary criteria are mean source-level writing advantage above zero and hard-checkpoint certificate coverage above 30%. The 30% minimum was selected after observing 201/400 (50.25%) validation coverage. It is a favorable development-selected convention, without an independent justification of minimal practical utility. Its satisfaction should not be equated with an important scientific discovery. We use 4,000 paired/source-puzzle bootstrap draws, 97.5% percentile intervals for both criteria with Bonferroni adjustment, and descriptive 95% intervals elsewhere. This adjustment concerns these two criteria, not the earlier exploratory results. The approximate inference is conditional on the fixed models and tables; it does not represent training-seed uncertainty.

## Selection and limits of the local freeze

This direction was chosen during an explicit search for a positive result after earlier hypotheses were not supported. The development pilot already showed substantial writing and coverage. Selecting a promising direction on development data and then testing fixed rules on new data does not itself invalidate the new test. It does require an explicit account of selection, modest interpretation and preservation of failures.

This local record is not an independent preregistration or external review. The experimental scripts were not included in the initial freeze hashes, and there is no external, complete, append-only log of every exploratory command. Current code and recorded rules agree; local logs place the freeze before generation. These artifacts alone cannot establish the entire chronology or certify the absolute absence of p-hacking. Current hashes establish file integrity, not scientific validity. The historical protocol is preserved without retroactive rewriting.

We then generate 1,024 random-backtracking solution grids and greedily remove clues while maintaining uniqueness until 30 remain. MRV backtracking and an independent Algorithm X solver verify each puzzle. Digit-normalized D4 solution keys are excluded against the published corpora, both earlier external corpora and previously accepted new solutions. This is not a complete Sudoku-isomorphism audit, a uniform generator or proof against unknown training contamination. No puzzle is removed on the basis of NCM outputs. All new data are used for hard readout; the first 128 sources are used for writing and the first 256 for secondary easy/medium evaluation.

![Train-only terminal associations. Row labels include support, preventing rare IDs from appearing equally well established. The writing alphabet selects one ID per digit.](figures/01_terminal_alphabet.png){{width=92%}}

# Terminal editing: association, intervention and control limitations

## Intervention and matched control

Let the final cumulative state be `z12` and the terminal contribution be `R12(c12)`. Choose one initially blank cell uniformly in each reserved source, using the frozen seed and source index independently of prediction and effect. For each requested digit different from the pre-intervention native digit, replace only that cell's terminal ID with the frozen writing ID. Decode `z12 + R12(c12_modified) - R12(c12_native)` with every other code and level fixed. This gives eight eligible requests per source, or 1,024 trials. The sham reproduces native output. No later generator stage remains to be adapted.

The terminal residual transform contains a width-three convolution, so one ID replacement can change up to three full-resolution latent positions. At every affected position, the control uses an independent Gaussian direction scaled to exactly the substitution's local L2 norm, with the same support. We save and verify all norms. This controls local perturbation amplitude and support; it is not a control matched to the distribution of codebook differences. Both sets of mixed states can be off distribution.

## Reserved result

Substitutions write the requested digit in **784/1,024 trials (76.56%)**, versus **0/1,024 controls**. The paired source-level advantage is 76.56 percentage points, with adjusted 97.5% CI **{interval(w['paired_advantage']['ci_familywise'])}**, satisfying the first frozen criterion. Per-digit outcomes are reported without additional confirmatory claims. All 240 unsuccessful writes remain in the archive.

Across eligible substitutions, **zero of the other 80 cells change**. Controls change an average of 0.0703 other cells per request. This is observed selectivity, not universal spatial independence: sources, one-cell selection, doses and IDs are limited. A zero-count bootstrap would be degenerate. Treating each puzzle as the independent source, zero of 128 sources with any collateral change has a descriptive 95% Wilson upper bound of approximately 2.91%. The 1,024 requests are not 1,024 independent sources.

The learned codebook differences and Gaussian directions have different geometry. Beating isotropic noise may reflect a trained decoder reading trained embeddings, without a novel or NCM-specific reasoning mechanism. Valid-code substitutions with permuted labels, other-position controls and a comparator codec would test stronger interpretations; these are not prospective criteria or completed replications in this campaign. Collateral-change counts concern the argmax of the other 80 cells, not invariance of logits or separators.

The lookup agrees with the native hard decoder on **{pct(h['cell_agreement_lookup'])}%** of reserved cells, above the 76.56% intervention success. These figures use different position populations and interventions; their difference is not a controlled estimate of context dependence. Unsuccessful writes do establish that the tested ID substitution does not invariably impose its associated digit. A descriptive post-hoc decomposition of the 240 failures finds **237 unchanged cells and three producing a different digit**. It does not localize the dependence to earlier residuals, neighboring codes or particular decoder modules. Editing a valid digit into a different one usually violates Sudoku constraints; this is not an accuracy-enhancing edit.

![Reserved causal writing by requested digit and aggregate contrast. Trial counts vary because requests equal to the native digit are excluded before intervention. Other-cell changes concern Sudoku cells, not a claim about every logit or separator.](figures/02_causal_writing.png){{width=100%}}

# Certified direct readout

## Policy and conditional guarantee

From generated terminal IDs, the policy constructs 81 lookup digits and fixed separators. It checks that each of the 27 row, column and block units contains digits 1 through 9, every given clue is preserved, and separators are valid. If all checks pass, it returns the candidate without calling the decoder. Otherwise it invokes the unchanged native decoder on the identical singleton latent state. The policy uses no reference, uniqueness solver or NCM retraining. A singleton baseline and fallback avoid floating-point changes caused by regrouping decoder batches; batched native agreement is also recorded.

**Proposition.** For a unique-solution puzzle with solution s, a sound certificate and exact native fallback, replacing the native output by a certified candidate cannot reduce exact-reference success on that puzzle.

**Proof.** Any accepted candidate satisfies the puzzle and hence equals s by uniqueness. When the candidate is rejected, the hybrid output is exactly the native output. In particular, whenever native output equals s, either branch of the hybrid policy also equals s. This argument does not require an accurate lookup table. For multiple-solution puzzles, it guarantees validity with clues, not agreement with an arbitrary selected reference. Uniqueness is a data assumption here, not established at policy runtime.

This is a standard verification-and-fallback argument, not a new principle of certification. It would apply to another candidate generator under the same conditions and is not evidence about NCM internals. Lookup-only accuracy could be poor and the argument would still hold, with lower coverage. On rejected inputs, native fallback can still produce incorrect answers.

## Reserved coverage and secondary quality

On hard, **552/1,024 candidates pass (53.91%)**, adjusted 97.5% CI **{interval(h['coverage']['ci_familywise'])}**, exceeding the frozen 30% criterion. Every accepted candidate equals its reference. The policy solves **885/1,024**, versus **883/1,024** natively, with zero correct-to-incorrect changes and two repairs. The descriptive exact-success difference is 0.195 percentage points, 95% CI {interval(h['quality_delta']['ci95'])}; it does not establish a strictly positive population accuracy effect. Lookup-only reading solves 552/1,024, so the fallback is material.

Separate easy and medium tables certify **{e['certified']}/256 ({pct(e['coverage']['mean'])}%)** and **{m['certified']}/256 ({pct(m['coverage']['mean'])}%)**, respectively. Native/hybrid successes are {e['native']['solved']}/{e['hybrid']['solved']} and {m['native']['solved']}/{m['hybrid']['solved']}, with zero regressions. These are secondary checkpoint replications on the same new 30-clue families, not independent data replications or evaluations on the official easy/medium distributions. No writing intervention is run on those checkpoints.

![Certified bypass and exact success. Hard uses 1,024 new puzzles; easy and medium share its first 256. Coverage error bars are descriptive 95% intervals; the hard co-primary interval is given in the text.](figures/03_certified_readout.png){{width=100%}}

# Timing: saved decoder calls versus full computation

All timings use CPU FP32, one thread, and the first 64 new puzzles, of which {pct(raw['acceptance_first64'])}% are certified. Four warm-up inputs per arm are excluded. Full-pipeline timing has three paired repeats per puzzle, randomized puzzle order and randomized AB/BA arm order. Both arms run the same native cached generator through all twelve levels; only readout differs. Outputs are checked. Separate readout timing uses sixteen internal repetitions per arm and cached states. No other campaign inference runs during measurement.

Full native/hybrid arithmetic mean latencies are **{speed['mean_native_ms']:.3f}/{speed['mean_certified_ms']:.3f} ms**. The geometric native-to-hybrid ratio is **{speed['geometric_speed_ratio']:.4f}**, 95% CI **{interval(speed['geometric_speed_ratio_ci95'],False)}**. Because this interval includes one, **end-to-end acceleration is not established**.

Readout-only arithmetic means are **{stage['mean_native_ms']:.3f}/{stage['mean_certified_ms']:.3f} ms**, corresponding to a ratio of **{stage['mean_native_ms']/stage['mean_certified_ms']:.2f}** and a descriptive reduction of **{pct(1-stage['mean_certified_ms']/stage['mean_native_ms'])}%** for this step and sample. No arithmetic-mean difference interval was announced or estimated. This does not imply the same reduction for the full model.

The announced geometric readout ratio, retained as a complementary statistic, is {stage['geometric_speed_ratio']:.4f}, 95% CI {interval(stage['geometric_speed_ratio_ci95'],False)}. It summarizes paired log ratios and is not the quotient of arithmetic means. Its magnitude reflects very fast accepted inputs alongside somewhat slower fallback inputs. It must not be read as a ninefold reduction in average readout cost or a ninefold acceleration of the complete model.

Intervals bootstrap source puzzles after averaging repeated log ratios. Raw durations and orders are retained. Uncertainty is conditional on this session, CPU and implementation, without between-session or between-machine replication. We make no GPU, throughput, energy or memory claim.

![Observed arithmetic mean latencies for the full pipeline and readout alone, with different y-axis scales. No arithmetic-mean confidence intervals are added; the announced paired analysis and its uncertainty are described in the text.](figures/04_timing.png){{width=100%}}

# Relation to prior work and interpretation

NCM [1] generates hierarchical discrete codes with a learned codec. The terminal digit association and controlled substitution characterize these checkpoints; their novelty should not be inflated into a claim that residual quantization or explicit output representations are new. Our earlier failures to establish strict decoded locking or omission-specific compensation remain unchanged. The present result concerns the final representation and its readout, not a replacement explanation for all intermediate oscillations.

Selective prediction with a reject option predates this work, including SelectiveNet [2]. Adaptive language-model computation, including CALM [3], also predates this work. Our policy retains every generator level and uses a task-specific exact external checker; it does not learn an abstention head or establish an earlier internal stopping rule. The empirical contribution is the causal alphabet characterization, its train-only implementation and the measured certified coverage for frozen Sudoku NCM.

Terminal IDs are associated with cell values and can often be used to edit the selected cell under this protocol. The novelty and scientific importance of that behavior remain open. A full-resolution codebook trained to reconstruct digit sequences could show this association without a new mechanism; a controlled codec comparison is needed. The terminal-only pilot includes both digit and separator failures and must not be reduced to an inability ever to reconstruct all 81 digits. A verified lookup has measured coverage because soundness is supplied by puzzle constraints, independently of any claim of perfect code semantics. None of this explains the earlier intermediate oscillations.

# Limitations and reproducibility

The causal study covers one hard checkpoint, 128 sources and one selected blank cell per source. Single-ID writes do not establish multi-cell compositional control, separator control, arbitrary perturbation locality or a human-concept basis. The Gaussian control matches local norm and support but not codebook geometry. There is no text replication of this mechanism. Three separately trained Sudoku checkpoints do not establish an architecture-wide law. New puzzles share one procedural generator and seed, with incomplete isomorphism and contamination audits. The conditional guarantee needs a sound checker, unique solution and unchanged fallback. The empirical absence of regressions alone would not prove that guarantee outside its assumptions.

The full experiment, frozen protocol and tables, all readouts and intervention predictions, norm arrays, source-level metrics and raw timing pairs are retained. A separate verification implementation recomputes counts and intervals, rechecks the 1,024 puzzles with two solvers, tests the certificate against invalid inputs and a set-based implementation, and replays sixteen fresh native trajectories and all nine writing targets with their random controls. Hooks verify zero decoder calls on accepted policy inputs and one on fallback. Seven prior manifest hashes are checked without modifying those campaigns.

Verification counts are software and data assertions: **4,096 check four properties for each of 1,024 puzzles**. These are not thousands of scientific replications. The verifier was written and run by the same research agent, so it is not an independent review and passing it does not establish absence of bias or p-hacking. An independent replication with code and protocol sealed in advance, stronger code controls and a comparator codec remains necessary for stronger claims; the current test data have now been examined and cannot serve as a new untouched test.

The prior publication is preserved in `sources/avant_revision_editoriale_20261024.zip`. An editorial baseline and audit pin the unchanged scientific files and record this change in interpretation. No original data, numerical outcomes, sampling seeds or decision criteria are changed. A clear effect under this protocol remains an empirical observation; novelty, general mechanism and end-to-end benefit remain unestablished.

Run `./run.sh` here to rebuild the documents and verify retained evidence without replacing historical measurements. For a full computational rerun in a new sibling folder, use the parent virtual environment to execute `scripts/reproduce.py --destination ../reproduction_alphabet`. It uses the preserved historical sources and same inputs; this is not independent confirmation, and its regenerated documents carry the earlier wording with an explicit notice directing readers to the current revised interpretation. Timing may vary. `sources_recherche_positive.zip` contains scripts, imported local model sources, configurations, split/protocol files and environment records; large weights and traces remain in the parent study or local result folders with hashes. `results/verification.json` records checks actually passed, and the final manifest pins deliverables. Scientific outcomes, including all failures and uncertain timing, are retained.

# References

1. *Next-Concept Models*, published article and implementation: [arXiv:2610.14162](https://arxiv.org/html/2610.14162v1), [official repository](https://github.com/HazyResearch/ncm). Local frozen copies and model-input hashes are inherited from the audited original study.
2. Geifman and El-Yaniv. *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*. ICML 2019. [Primary paper](https://arxiv.org/abs/1901.09192).
3. Schuster et al. *Confident Adaptive Language Modeling*. NeurIPS 2022. [Primary paper](https://arxiv.org/abs/2207.07061).
'''
    numeric=read('numerical_support_audit.json')
    text=text.replace('The latter result argues against interpreting the last residual as an independently complete answer.',"The zero counts all 89 tokens, including separators. A post-hoc format audit finds 10/32 grids with all 81 digits correct but 0/32 with all eight separators correct; reference digit accuracy is 96.18%. This is not evidence that terminal-only input can never reconstruct grid digits. It fails as a reliable standalone readout, with format and digit errors distinguished. The frozen pilot metric and reserved criteria are unchanged.")
    text=text.replace('Both sets of mixed states can be off distribution.',f"Both sets of mixed states can be off distribution. A numerical audit found FP32 convolution/subtraction tails outside the mathematical three-position support (maximum position norm {numeric['maximum_tail_position_norm']:.3g}). Controls match the actual norms including these tails. A post-hoc direct FP64 affine contraction, cast back to FP32 before decoding, removes the tails and changes none of the complete readouts across all 1,152 requests (including native-digit requests) in either arm. Original frozen interventions and primary counts are retained.")
    (D/'PAPER.md').write_text(text)

def build_pdf():
    build=D/'.cache/tex-build';build.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(TMPDIR=str(D/'.cache/tmp'),TEXMFVAR=str(D/'.cache/tex-var'),TEXMFCONFIG=str(D/'.cache/tex-config'))
    Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
    cmd=['/opt/homebrew/bin/pandoc','PAPER.md','--standalone','--number-sections','-o','PAPER.tex']
    subprocess.run(cmd,cwd=D,check=True,env=env,capture_output=True)
    with (D/'logs/latex.log').open('w') as out:
        for _ in range(2):subprocess.run(['/Library/TeX/texbin/pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory',str(build),'PAPER.tex'],cwd=D,check=True,env=env,stdout=out,stderr=subprocess.STDOUT)
    (D/'PAPER.pdf').write_bytes((build/'PAPER.pdf').read_bytes())
    (D/'README.md').write_text('''# Alphabet terminal et lecture certifiée

Commencer par [RAPPORT.md](RAPPORT.md) pour les observations, la méthode et les limites de leur interprétation. [PAPER.pdf](PAPER.pdf) est la version anglaise mise en page. La [note de révision](REVISION_EDITORIALE.md) explique les corrections de portée ; les versions précédentes, le gel et tous les résultats sont conservés.

Le protocole local et les tables étaient fixés avant les nouveaux tests, après un pilote favorable. Ce gel n'était pas une prérégistration indépendante et n'incluait pas les scripts. La nouveauté, un mécanisme général et un gain global ne sont pas établis. Une réplication indépendante avec code scellé, contrôles de codes et codec témoin reste à faire. Les assertions de vérification sont des contrôles logiciels, pas des milliers de réplications scientifiques.

`./run.sh` régénère les documents et vérifie les résultats conservés, sans réécrire les observations historiques. Pour recalculer toutes les expériences dans un nouveau dossier frère : `PYTHONPYCACHEPREFIX="$PWD/.cache/pycache" ../.venv/bin/python scripts/reproduce.py --destination ../reproduction_alphabet`. Ce recalcul des mêmes entrées utilise les sources historiques ; ce n'est pas une réplication indépendante. Les rapports régénérés dans ce dossier frère ont la rédaction antérieure, accompagnée d'une notice ; l'interprétation révisée de référence reste ici. Les anciens dossiers sont lus sans écriture.

Lancer `scripts/verify.py` avec le Python du parent pour un audit sans régénération des documents (deux solveurs, statistiques, rejeu natif et causal sur 16 puzzles, archive et publication). Les commandes et variables de cache sont dans le rapport et `run.sh`.

La politique utilisable est `scripts/policy.py` : lookup, vérification, sinon décodeur natif. Elle n'utilise ni référence ni solveur. L'unicité est une précondition de la garantie de fidélité à la référence, pas une vérification gratuite effectuée par la politique.

Les données nouvelles ont une solution unique et 30 indices. Les réplications easy/medium partagent les premières familles ; les temps ne démontrent pas une accélération globale. Ne pas présenter les substitutions volontaires de chiffres comme une amélioration de résolution.
''')

def archive():
    with zipfile.ZipFile(D/'sources_recherche_positive.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((D/'scripts').glob('*.py')):z.write(p,'scripts/'+p.name)
        for p in [D/'run.sh',D/'protocol.json',D/'PROTOCOLE_GELE.md',D/'FEUILLE_DE_ROUTE.md',D/'JOURNAL.md',D/'REVISION_EDITORIALE.md',D/'results/editorial_revision_baseline.json',D/'sources/avant_revision_editoriale_20261024.zip',D/'results/protocol_frozen.json',D/'results/discovery_tables.json',D/'results/discovery_pilot.json']:
            z.write(p,str(p.relative_to(D)))
        for p in (D/'sources').glob('*.json'):z.write(p,'sources/'+p.name)
        for p in [ROOT/'scripts/common.py',ROOT/'scripts/solvers.py',ROOT/'results/splits.json',ROOT/'sources/requirements-lock.txt',ROOT/'results/input_manifest.json']:
            z.write(p,'parent/'+str(p.relative_to(ROOT)))
        for p in sorted((ROOT/'vendor/ncm_mac').rglob('*')):
            if p.is_file() and p.suffix in ['.py','.yaml','.yml','.json','.md'] and '__pycache__' not in str(p):z.write(p,'parent/'+str(p.relative_to(ROOT)))
        for task in ['hard','easy','medium']:z.write(ROOT/f'artifacts/sudoku/{task}/config.yaml',f'parent/artifacts/sudoku/{task}/config.yaml')

def main():
    t=read('discovery_tables.json');w=read('writing_interventions.json');rr={k:read(k+'_readout.json') for k in ['hard','easy','medium']};b=read('benchmark.json')
    write(D/'results/primary_decision.json',dict(protocol_sha256=sha(D/'protocol.json'),interval_level=.975,adjustment='Bonferroni for two co-primary effects; source-puzzle percentile bootstrap',writing=dict(threshold=0.,estimate=w['paired_advantage']['mean'],interval=w['paired_advantage']['ci_familywise'],criterion_satisfied=w['paired_advantage']['ci_familywise'][0]>0),coverage=dict(threshold=.30,estimate=rr['hard']['coverage']['mean'],interval=rr['hard']['coverage']['ci_familywise'],criterion_satisfied=rr['hard']['coverage']['ci_familywise'][0]>.30),both_co_primary_criteria_satisfied=bool(w['paired_advantage']['ci_familywise'][0]>0 and rr['hard']['coverage']['ci_familywise'][0]>.30)))
    pilot_format_audit();figures(t,w,rr,b);h,e,m,speed,stage,raw=report(t,w,rr,b)
    paper(t,w,rr,b,h,e,m,speed,stage,raw)
    build_pdf();archive();log('positive_publication_complete')

if __name__=='__main__':main()
