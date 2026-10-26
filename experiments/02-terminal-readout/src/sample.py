"""Small portable readout/edit trial using the saved discovery table.

This uses the first n official puzzles, not the historical fresh test corpus.
It does not refit the table, run Gaussian controls or benchmark latency.
"""
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root',type=Path,default=REPO/'assets')
    parser.add_argument('--n',type=int,default=8)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 1<=args.n<=128:raise ValueError('Use 1..128 puzzles for this small trial.')
    if args.output and args.output.exists():raise FileExistsError(args.output)
    sys.path.insert(0,str(REPO/'common/backends/audit'))
    import numpy as np
    import torch
    from ncm_mac.sudoku import SudokuModel,load_boards,verify_assets
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    verify_assets(args.asset_root,'hard',include_data=True)
    table=json.loads((STUDY/'recorded/discovery_tables.json').read_text())['hard']
    mapping=np.array(table['mapping']);alphabet=table['alphabet_by_digit']
    puzzles=load_boards(args.asset_root/'data/sudoku/hard/puzzles.pickle')[:args.n]
    truth=np.array(load_boards(args.asset_root/'data/sudoku/hard/solutions.pickle')[:args.n])
    model=SudokuModel(args.asset_root/'artifacts/sudoku/hard')
    cells=np.array([i for i in range(89) if i%10!=9]);separators=np.array([i for i in range(89) if i%10==9])
    units=[list(range(r*9,r*9+9))for r in range(9)]+[list(range(c,81,9))for c in range(9)]
    units += [[(br+r)*9+bc+c for r in range(3)for c in range(3)]for br in (0,3,6)for bc in (0,3,6)]
    hits=trials=certified=native_solved=hybrid_solved=0
    with torch.inference_mode():
        for index,board in enumerate(puzzles):
            prefix=model.prefix([board]);pred=model.predict(prefix)
            native=pred.ids[0].numpy();candidate=np.full(89,10,dtype=np.int64)
            code=pred.codes[:,-89:];candidate[cells]=mapping[code[0].numpy()[cells]]
            given=np.array(board)[1:-1];values=candidate[cells]
            valid=bool(np.all((given[cells]==0)|(given[cells]==values)) and all(np.array_equal(np.sort(values[u]),np.arange(1,10))for u in units) and np.all(candidate[separators]==10))
            if valid:assert np.array_equal(candidate,truth[index])
            certified+=valid;native_solved+=int(np.array_equal(native,truth[index]))
            hybrid_solved+=int(np.array_equal(candidate if valid else native,truth[index]))
            token=int(cells[np.flatnonzero(given[cells]==0)[0]])
            vq=model.vqvae.vq
            contribution=vq._apply_quant_resi(vq.embeddings[11](code),11)
            for digit in range(1,10):
                if digit==int(native[token]):continue
                changed=code.clone();changed[0,token]=int(alphabet[digit-1])
                delta=vq._apply_quant_resi(vq.embeddings[11](changed),11)-contribution
                edited=model.vqvae.decoder(pred.reconstruction+delta).argmax(-1)
                hits+=int(edited[0,token]==digit);trials+=1
    assert hybrid_solved>=native_solved
    report={'passed':True,'scope':'fresh small trial on first official puzzles using an existing discovery table; not the historical evaluation',
            'n':args.n,'certified':certified,'native_solved':native_solved,'hybrid_solved':hybrid_solved,
            'requested_writes':trials,'successful_writes':hits,'writing_rate':hits/trials,
            'controls':'no Gaussian control or timing comparison in this small trial'}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
