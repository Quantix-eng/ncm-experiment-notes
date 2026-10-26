"""Audit shared solution families across difficulties and effective weight identity."""
from common import *
from prepare import symkey

def main():
    keys={};sols={};raw={};weights={};allp={}
    for d in ['easy','medium','hard']:
        p,s=boards(d);allp[d]=p;raw[d]=s;keys[d]=set(symkey(c) for c in p[:,1:-1][:,CELL]);sols[d]=[symkey(c) for c in s[:,CELL]]
        weights[d]={}
        for kind in ['tokenizer','generator']:
            path=ROOT/f'artifacts/sudoku/{d}/{kind}.pt';state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)['model']
            digest=hashlib.sha256()
            for key,t in sorted(state.items()):
                digest.update(key.encode());digest.update(str(t.dtype).encode());digest.update(str(tuple(t.shape)).encode());digest.update(t.contiguous().numpy().tobytes())
            weights[d][kind]=dict(file_sha256=sha(path),state_dict_sha256=digest.hexdigest())
    overlap={}
    for a,b in [('easy','medium'),('easy','hard'),('medium','hard')]:
        overlap[a+'_'+b]=dict(puzzle_symmetry_overlap=len(keys[a]&keys[b]),solution_symmetry_overlap=len(set(sols[a])&set(sols[b])),
            exact_solutions_same_order=bool(np.array_equal(raw[a],raw[b])),
            tokenizer_state_dict_equal=weights[a]['tokenizer']['state_dict_sha256']==weights[b]['tokenizer']['state_dict_sha256'],
            generator_state_dict_equal=weights[a]['generator']['state_dict_sha256']==weights[b]['generator']['state_dict_sha256'])
    hardmap={key:i for i,key in enumerate(sols['hard'])};mapping={d:[hardmap.get(key,-1) for key in sols[d]] for d in sols}
    write(ROOT/'results/cross_corpus_overlap.json',dict(overlap=overlap,weights=weights,solution_family_to_hard_index=mapping,
          grouping='D4 plus digit normalization, not exhaustive Sudoku isomorphism',
          implication='Cross-difficulty datasets share solution families; they are correlated replications. Transfer must hold out the same solution families.'))
    progress('cross_corpus_audit',overlap=overlap)

if __name__=='__main__':main()
