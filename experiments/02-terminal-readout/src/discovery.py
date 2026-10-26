"""Development-only exploration: symbol tables, certification, and small causal pilot."""
from base import *
from runtime import Engine

def tables():
    tr=np.array(SPLIT['train']);va=np.array(SPLIT['validation']);out={}
    for task in ['hard','easy','medium']:
        a=np.load(ROOT/f'results/{task}.npz');c=a['codes'][:,-89:];target=a['decoded'][:,-1];cnt=np.zeros((16,12),int);np.add.at(cnt,(c[tr][:,CELL].ravel(),target[tr][:,CELL].ravel()),1);mapping=cnt[:,1:10].argmax(1)+1
        alphabet=np.array([cnt[:,j].argmax() for j in range(1,10)]);assert len(set(alphabet))==9
        pred=lookup_board(a['codes'][va],mapping);accept=certificate(pred,a['prefix'][va]);native=target[va];hybrid=np.where(accept[:,None],pred,native)
        out[task]=dict(mapping=mapping,alphabet_by_digit=alphabet,train_counts=cnt,validation=dict(n=len(va),cell_agreement=float((pred[:,CELL]==native[:,CELL]).mean()),exact_agreement=int(np.all(pred==native,1).sum()),certified=int(accept.sum()),certified_correct=int(np.all(pred==a['reference'][va],1)[accept].sum()),native_solved=int(np.all(native==a['reference'][va],1).sum()),hybrid_solved=int(np.all(hybrid==a['reference'][va],1).sum())))
    write(D/'results/discovery_tables.json',out);return out

@torch.inference_mode()
def pilot(model,tables):
    a=np.load(ROOT/'results/hard.npz');ix=np.array(SPLIT['validation'])[:32];codes=torch.tensor(a['codes'][ix]);ref=a['reference'][ix];alphabet=tables['hard']['alphabet_by_digit'];p=a['prefix'][ix];counts=[];arrays={}
    for lo in range(0,len(ix),16):
        sl=slice(lo,lo+16);c=codes[sl];snapshots,=model.vqvae.vq.idx_to_zhats(c,return_zlevels=False);z11=snapshots[10];zfinal=snapshots[11];native=model.vqvae.decoder(zfinal).argmax(-1);assert np.array_equal(native.numpy(),a['decoded'][ix[sl],-1]);prefix=p[sl]
        with Engine(model,prefix) as e:
            r12=e.contribution(11,c[:,-89:]);only=model.vqvae.decoder(r12).argmax(-1);arrays.setdefault('terminal_only',[]).append(only.numpy())
            positions=np.array([np.flatnonzero(prefix[j,1:-1][CELL]==0)[int(k)%np.sum(prefix[j,1:-1][CELL]==0)] for j,k in enumerate(ix[sl])]);tokenpos=CELL[positions];native_digit=native.numpy()[np.arange(len(c)),tokenpos];hits=[]
            for digit in range(1,10):
                patched=c[:,-89:].clone();patched[torch.arange(len(c)),torch.tensor(tokenpos)]=int(alphabet[digit-1]);delta=e.contribution(11,patched)-r12;pred=model.vqvae.decoder(zfinal+delta).argmax(-1);hits.append(pred.numpy()[np.arange(len(c)),tokenpos]==digit)
            counts.append(np.stack(hits,1));arrays.setdefault('native_digit',[]).append(native_digit)
    only=np.concatenate(arrays['terminal_only']);h=np.concatenate(counts);old=np.concatenate(arrays['native_digit']);eligible=np.arange(1,10)[None]!=old[:,None];result=dict(n=len(ix),terminal_only_cell_agreement=float((only[:,CELL]==a['decoded'][ix,-1][:,CELL]).mean()),terminal_only_solved=int(np.all(only==ref,1).sum()),editing_hits=int((h&eligible).sum()),editing_trials=int(eligible.sum()),editing_rate=float(h[eligible].mean()))
    write(D/'results/discovery_pilot.json',result);np.savez_compressed(D/'results/discovery_pilot.npz',indices=ix,hits=h,eligible=eligible,terminal_only=only,native_digit=old);log('discovery_complete',**result)

if __name__=='__main__':
    t=tables();pilot(SudokuModel(ROOT/'artifacts/sudoku/hard'),t)
