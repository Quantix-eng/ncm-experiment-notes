"""Post-hoc algebraic check of the predeclared row/pooling hypothesis.

Fixed row token bags imply fixed pooled sums before GroupNorm and residual
subtraction. This explains a representation constraint, not a training trajectory.
"""
from base import *
import math
import torch.nn.functional as F

@torch.inference_mode()
def main():
    ix=np.array(SPLIT['train'])[:256];_,s=boards('hard');ids=torch.tensor(s[ix]);one=F.one_hot(ids,12).float();report={}
    windows=[]
    for j in range(9):
        start=math.floor(j*89/9);stop=math.ceil((j+1)*89/9);counts=np.bincount(s[ix[0],start:stop],minlength=12)
        for row in s[ix]:assert np.array_equal(np.bincount(row[start:stop],minlength=12),counts)
        windows.append(dict(slot=j,start_token=start,stop_token_exclusive=stop,length=stop-start,token_counts=counts))
    for d in ['hard','easy','medium']:
        state=torch.load(ROOT/f'artifacts/sudoku/{d}/tokenizer.pt',map_location='cpu',weights_only=True,mmap=True)['model']
        e=F.embedding(ids,state['encoder.wte.weight'])+state['encoder.wpe.weight'][None]
        z=F.group_norm(e.transpose(1,2),8).transpose(1,2)
        vals={}
        for length in [8,9,10,18]:
            h=F.interpolate(one.transpose(1,2),size=length,mode='area').transpose(1,2)
            a=F.interpolate(e.transpose(1,2),size=length,mode='area').transpose(1,2)
            b=F.interpolate(z.transpose(1,2),size=length,mode='area').transpose(1,2)
            vals[str(length)]=dict(token_histogram_inter_puzzle_variance=float(h.double().var(0,correction=0).mean()),
                 raw_embedding_inter_puzzle_variance=float(a.double().var(0,correction=0).mean()),
                 normalized_embedding_inter_puzzle_variance=float(b.double().var(0,correction=0).mean()))
        report[d]=vals
    write(D/'results/pooling_algebra.json',dict(n=256,indices=ix,windows=windows,checkpoint_variances=report,
      derivation='Adaptive area pooling uses floor(j*89/9):ceil((j+1)*89/9). Each window contains all 9 digits of one row plus one or two separators. Their token histograms are identical across solved boards. Positional embeddings are fixed. Therefore pre-normalization pooled embeddings are invariant to within-row digit order, up to rounding. GroupNorm and previous residual contributions can reintroduce variation.',
      limitation='This does not force a single nearest residual code and does not establish the historical cause of codebook collapse.'))
    log('pooling_algebra_complete',variances=report)

if __name__=='__main__':main()
