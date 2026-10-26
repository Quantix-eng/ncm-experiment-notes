"""Derive the concise GPU report directly from the measured JSON artifacts."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = {8: json.loads((ROOT / "audit/sudoku-mps.json").read_text()),
               32: json.loads((ROOT / "audit/sudoku-mps-b32.json").read_text())}
    summary = dict(ground_truth_commit="254bb49", results={})
    for batch, report in reports.items():
        assert set(report["tasks"]) == {"easy","medium","hard"}
        rows = [x for t in report["tasks"].values() for x in t["probability_comparisons"]
                if x["level"] != "decoder_free_running"]
        assert all(math.isfinite(x[k]) for x in rows for k in
                   ["max_abs_probability","max_total_variation","max_kl_cpu_to_mps","max_abs_logit"])
        maxima = {k:max(x[k] for x in rows) for k in
                  ["max_abs_probability","max_total_variation","max_kl_cpu_to_mps","max_abs_logit"]}
        assert all(math.isfinite(x) for x in maxima.values())
        summary["results"][str(batch)] = dict(
            tested_boards=sum(t["cpu_metrics"]["n"] for t in report["tasks"].values()),
            changed_answers=sum(t["changed_answer_boards"] for t in report["tasks"].values()),
            changed_code_boards=sum(t["changed_code_boards"] for t in report["tasks"].values()),
            probability_maxima=maxima,
            probability_gate_passed=(maxima["max_abs_probability"]<=1e-5 and
                                     maxima["max_total_variation"]<=1e-4 and maxima["max_kl_cpu_to_mps"]<=1e-6))
    summary["decision"] = "CPU remains default; MPS stays experimental"
    (ROOT / "audit/sudoku-mps-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines = ["# GPU Sudoku — 22 octobre 2026", "",
             "**MPS fonctionne sans repli CPU ; la version CPU `254bb49` reste la référence et le choix par défaut.**", "",
             "Interpolation adaptative remplacée par sélection/somme/division sur Metal ; cache du contexte conservé. FP32, poids et règles de décodage inchangés. Aucun module CPU de référence modifié.", "", "## Fidélité", ""]
    for batch,x in summary["results"].items():
        m=x["probability_maxima"]
        lines.append(f"- Batch {batch} : {x['tested_boards']} cas, {x['changed_answers']} réponses différentes et {x['changed_code_boards']} grilles avec codes différents. Écart maximal de probabilité à conditionnement identique : {m['max_abs_probability']:.3g} ; KL maximale {m['max_kl_cpu_to_mps']:.3g}.")
    lines += ["", "Les mêmes 128 indices par difficulté sont testés aux deux batchs : **384 grilles distinctes**, pas 768. Diagnostic : softmax des logits à T=1, mesuré en FP64 ; l'inférence reste FP32 avec argmax à T=0. Le seuil exploratoire d'écart absolu `1e-5` est dépassé aux deux batchs et n'est pas assoupli après observation. Identité des réponses observée ne signifie pas identité des distributions.", "", "## Vitesse et mémoire", "", "Médianes de cinq répétitions après chauffe ; ratios CPU optimisé / MPS (>1 favorise MPS).", ""]
    for difficulty,t in reports[8]["tasks"].items():
        text=" ; ".join(f"batch {b} : {v['cpu']['median_seconds']/v['mps']['median_seconds']:.2f}×" for b,v in t["benchmark"].items())
        lines.append(f"- {difficulty} : {text}.")
    lines += ["", "Ces mesures sont sensibles à l'état de la machine. Les séries CPU puis GPU ne sont pas appariées ; un contrôle medium sur trois paires alternées, libération des caches incluse, est conservé séparément. Les allocations du pilote sont relevées en fin de série, pas un pic mesuré.", ""]
    memory=ROOT / "audit/sudoku-mps-memory.json"
    if memory.exists():
        control=json.loads(memory.read_text())
        for b,v in control.items():
            lines.append(f"- Contrôle medium batch {b}, coût de libération inclus : CPU {v['cpu_median']:.2f} s, MPS {v['mps_median']:.2f} s ; pilote après libération ≤ {max(x['driver_after_clear'] for x in v['memory'])/1e9:.2f} Go.")
    control=json.loads((ROOT / "audit/sudoku-mps-pooling-control.json").read_text())
    pool_error=max(x["max_abs_error"] for x in control["pool_random_mps"])
    lines += ["",f"Contrôle du pooling sur entrée aléatoire : écart maximal MPS/CPU {pool_error:.3g}. Adaptateur seul sur CPU, 8 grilles par difficulté : aucune réponse différente, écart maximal de probabilité du générateur {max(x['generator_max_abs_probability'] for x in control['cpu_adapter_control'].values()):.3g}. Cela distingue une contribution du pooling des autres écarts de backend, sans établir un budget d'erreur exhaustif."]
    lines += ["", "**Décision :** conserver le CPU pour la publication actuelle. La piste Metal est reproductible mais reste expérimentale ; elle nécessite un bénéfice stable et une validation élargie de fidélité avant promotion. Le corpus complet de 6 000 grilles n'a pas été évalué ici.", "", "Preuves : [batch 8 et benchmarks](audit/sudoku-mps.json), [batch 32](audit/sudoku-mps-b32.json), [contrôle mémoire](audit/sudoku-mps-memory.json), [contrôle du pooling](audit/sudoku-mps-pooling-control.json), [protocole](experiments/README.md), [journal](FEUILLE_DE_ROUTE.md)."]
    (ROOT / "RAPPORT_GPU.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":main()
