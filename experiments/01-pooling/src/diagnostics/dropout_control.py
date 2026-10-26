"""Toggle only the frozen encoder dropout; never update quantizer EMA state."""
from base import *
from diagnose_level5 import diagnose,variance,utilization

def state_hash(module):
    h=hashlib.sha256()
    for key,value in sorted(module.state_dict().items()):h.update(key.encode());h.update(value.detach().contiguous().numpy().tobytes())
    return h.hexdigest()

@torch.inference_mode()
def main():
    ix=np.array(SPLIT['train'])[:64];report={};saved={}
    for d in ['hard','easy','medium']:
        model=SudokuModel(ROOT/f'artifacts/sudoku/{d}');_,solutions=boards(d);before=state_hash(model.vqvae)
        _,native=diagnose(model,solutions[ix],full=False);nativecodes=native['codes5'];runs=[]
        assert not model.vqvae.training and not model.vqvae.vq.training
        model.vqvae.encoder.drop.train(True)
        for seed in range(Q['seed']+1,Q['seed']+5):
            torch.manual_seed(seed);_,state=diagnose(model,solutions[ix],full=False)
            runs.append(dict(seed=seed,codes=utilization(state['codes5']),u5=variance(state['u5']),
                  fraction_different_from_native=float((state['codes5']!=nativecodes).mean())))
            saved[f'{d}_{seed}_codes']=state['codes5'];log('encoder_dropout_control',checkpoint=d,seed=seed,used=runs[-1]['codes']['used'])
        model.vqvae.encoder.drop.train(False);after=state_hash(model.vqvae);assert before==after
        combined=np.concatenate([saved[f'{d}_{seed}_codes'] for seed in range(Q['seed']+1,Q['seed']+5)])
        report[d]=dict(indices=ix,dropout_probability=float(model.vqvae.encoder.drop.p),native=utilization(nativecodes),native_u5=variance(native['u5']),
          dropout_runs=runs,dropout_pooled_codes=utilization(combined),state_dict_before=before,state_dict_after=after,
          quantizer_remained_eval=not model.vqvae.vq.training)
        saved[d+'_native_codes']=nativecodes
    write(D/'results/dropout_control.json',report);np.savez_compressed(D/'results/dropout_control.npz',**saved)
    log('dropout_control_complete')
if __name__=='__main__':main()
