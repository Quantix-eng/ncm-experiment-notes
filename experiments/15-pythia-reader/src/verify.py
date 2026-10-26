"""Independent scalar recomputation from saved residuals; no new model evaluation."""
from common import *
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['validation','test'],required=True)
a=parser.parse_args();check_freeze();setup();m=load_model()
ln=m.gpt_neox.final_layer_norm
unembedding_fp64=m.embed_out.weight.double()
states=np.load(D/f'results/{a.split}_states.npz')['states']
rows=json.loads((D/f'results/{a.split}_observations.json').read_text())
summary=json.loads((D/f'results/{a.split}_summary.json').read_text())
all_contrasts=0
for row in rows:
    z=row['loss']
    for key,value in dict(D=z[1]-z[0],R=z[1]-z[2],I=z[1]+z[3]-z[2]-z[0],S=z[4]-z[2]).items():
        assert abs(row[key]-value)<1e-12;all_contrasts+=1
    assert row['continuation_R']==row['continuation_I']==row['continuation_S']==0
assert np.array([x['correct'] for x in rows]).sum(axis=0).tolist()==summary['metrics']['correct_by_condition']
largest=0.;checked=0
with torch.inference_mode():
    for i in range(min(32,len(rows))):
        r0,rd,s0,sd,so=map(torch.from_numpy,states[i])
        observed=[]
        for r,s in [(r0,s0),(rd,s0),(rd,sd),(r0,sd),(rd,so)]:
            # Independent FP64 formula, rather than the runner's native-kernel ratio form.
            rd=r.double();sd=s.double()
            normalized=(rd-rd.mean())/sd*ln.weight.double()+ln.bias.double()
            logits=torch.nn.functional.linear(normalized,unembedding_fp64)
            loss=float(torch.logsumexp(logits,0)-logits[rows[i]['target']])
            observed.append(loss)
            largest=max(largest,abs(loss-rows[i]['loss'][len(observed)-1]));checked+=1
        assert abs((observed[1]-observed[2])-rows[i]['R'])<2e-4
    assert largest<2e-4,largest
write(f'results/{a.split}_verification.json',dict(scalar_loss_checks=checked,max_error=largest,
    scalar_contrast_checks=all_contrasts,correct_counts_verified=True,
    independent_fp64_formula=True,no_new_full_model_forward=True,independent_scientific_validation=False))
print(checked,largest)
