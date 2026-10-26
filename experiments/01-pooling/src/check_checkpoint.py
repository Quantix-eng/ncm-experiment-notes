"""New small inference check: force the recorded constant identifier at level 5.

This executes the head before replacing its logits. It checks observed parity;
it does not reproduce the historical prediction-bypass timing or save compute.
"""
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--asset-root',type=Path,default=ROOT/'assets')
    ap.add_argument('--n',type=int,default=16);ap.add_argument('--output',type=Path)
    a=ap.parse_args()
    if not 1<=a.n<=2000:raise ValueError('Use 1..2000 puzzles')
    if a.output and a.output.exists():raise FileExistsError(a.output)
    import torch
    sys.path.insert(0,str(ROOT/'common/backends/audit'))
    from ncm_mac.sudoku import SudokuModel,load_boards,verify_assets
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    verify_assets(a.asset_root,'hard',include_data=True)
    model=SudokuModel(a.asset_root/'artifacts/sudoku/hard')
    boards=load_boards(a.asset_root/'data/sudoku/hard/puzzles.pickle')[:a.n]
    equal={k:True for k in ['codes','reconstruction','logits','ids']};constant=True;calls=0
    def force(module,inputs,out):
        nonlocal calls
        calls+=1
        forced=torch.full_like(out,-torch.inf);forced[...,11]=0
        return forced
    for start in range(0,len(boards),8):
        prefix=model.prefix(boards[start:start+8])
        native=model.predict(prefix)
        offset=sum(model.net.levels[:4]);length=model.net.levels[4]
        constant=constant and bool((native.codes[:,offset:offset+length]==11).all())
        hook=model.net.lm_heads[4].register_forward_hook(force)
        try:patched=model.predict(prefix)
        finally:hook.remove()
        for key in equal:equal[key]=equal[key] and torch.equal(getattr(native,key),getattr(patched,key))
    report={'scope':'new smoke check on the first n official hard puzzles; not the historical held-out study',
            'n':a.n,'torch':torch.__version__,'backend':'cpu','dtype':'float32','threads':1,
            'level5_identifier_11_on_all_inputs':constant,'forced_head_calls':calls,'equal':equal,
            'passed':constant and all(equal.values())}
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
