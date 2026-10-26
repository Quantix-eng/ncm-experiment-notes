"""Post-hoc fixed fine-code control; no new tuning and no solving claim."""
from repair import *

@torch.inference_mode()
def main():
    selection=json.loads((D/'results/selection_frozen.json').read_text());name=selection['selected'];ar=np.load(D/'results/hard_test.npz');n=128;ref=ar['reference'][:n]
    native=ar['native_codes'][:n];new=ar[name+'_codes'][:n];hybrid=native.copy();hybrid[:,13:22]=new[:,13:22]
    model=SudokuModel(ROOT/'artifacts/sudoku/hard');arrays={};report={}
    for condition,codes in [('native',native),('reencoded',new),('hybrid',hybrid)]:
        ids=[];four=[];five=[]
        for lo in range(0,n,16):
            zhats,=model.vqvae.vq.idx_to_zhats(torch.tensor(codes[lo:lo+16]),return_zlevels=False)
            ids.append(model.vqvae.decoder(zhats[-1]).argmax(-1).numpy())
            if condition!='hybrid':
                four.append(model.vqvae.decoder(zhats[3]).argmax(-1).numpy());five.append(model.vqvae.decoder(zhats[4]).argmax(-1).numpy())
        ids=np.concatenate(ids);arrays[condition+'_ids']=ids;arrays[condition+'_codes']=codes
        report[condition]=dict(exact=int(np.all(ids==ref,axis=1).sum()),token_errors=int((ids!=ref).sum()),changed_grids_vs_native=int(np.any(ids!=ar['native_ids'][:n],axis=1).sum()))
        if condition!='hybrid':
            for k,parts in [(4,four),(5,five)]:
                dec=np.concatenate(parts);arrays[f'{condition}_decoded{k}']=dec;report[condition][f'cell_accuracy{k}']=float((dec[:,CELL]==ref[:,CELL]).mean())
    assert np.array_equal(arrays['native_ids'],ar['native_ids'][:n]) and np.array_equal(arrays['reencoded_ids'],ar[name+'_ids'][:n])
    old=(arrays['native_decoded5'][:,CELL]==ref[:,CELL]).mean(1);newacc=(arrays['reencoded_decoded5'][:,CELL]==ref[:,CELL]).mean(1)
    report['paired_delta_level5_cell_accuracy']=paired_mean_ci(newacc-old);report['changed_fine_codes']=int((new[:,22:]!=native[:,22:]).sum());report['amendment_sha256']=filehash(D/'AMENDEMENT_COMPENSATION.md')
    np.savez_compressed(D/'results/compensation.npz',reference=ref,indices=ar['indices'][:n],**arrays);write(D/'results/compensation.json',report);log('compensation_complete',**report)

if __name__=='__main__':main()
