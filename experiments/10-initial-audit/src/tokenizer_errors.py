"""Independently replay all final reconstruction errors in untouched upstream."""
from common import *
sys.path.insert(0,str(ROOT/'vendor/upstream-ncm'))
from models.vqvae import VQVAE
from omegaconf import OmegaConf

@torch.inference_mode()
def main():
    result={}
    for d in ['easy','medium','hard']:
        a=np.load(ROOT/f'results/{d}.npz'); rec=a['reconstructed'][:,-1];ref=a['reference'];errors=np.argwhere(rec!=ref)
        rows=np.unique(errors[:,0]) if len(errors) else np.array([],dtype=int); cases=[]
        if len(rows):
            cfg=OmegaConf.load(ROOT/f'artifacts/sudoku/{d}/config.yaml'); vq=VQVAE(**cfg.tokenizer.vqvae).eval()
            vq.load_state_dict(torch.load(ROOT/f'artifacts/sudoku/{d}/tokenizer.pt',map_location='cpu',weights_only=True,mmap=True)['model'],strict=True)
            idx=vq.encode(torch.tensor(ref[rows]));pred=vq.decoder(vq.vq.idx_to_z_q(idx)).argmax(-1).numpy()
            assert np.array_equal(idx.numpy(),a['encoded'][rows])
            assert np.array_equal(pred,rec[rows])
            for row in rows:
                bad=np.flatnonzero(rec[row]!=ref[row]);cases.append(dict(index=int(row),token_positions=bad,
                    expected=ref[row,bad],actual=rec[row,bad],upstream_identical=True,
                    generator_solves_puzzle=bool(np.array_equal(a['decoded'][row,-1],ref[row]))))
        result[d]=dict(token_errors=len(errors),grid_errors=len(rows),total_tokens=ref.size,total_grids=len(ref),cases=cases)
    write(ROOT/'results/tokenizer_errors.json',result);progress('tokenizer_error_verification',results=result)

if __name__=='__main__':main()
