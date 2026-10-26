"""Compare the untouched upstream generator to the copied CPU runner."""
from common import *
sys.path.insert(0,str(ROOT/'vendor/upstream-ncm'))
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer
from omegaconf import OmegaConf

@torch.inference_mode()
def main():
    result={}
    for d in ['easy','medium','hard']:
        ck=ROOT/f'artifacts/sudoku/{d}'; cfg=OmegaConf.load(ck/'config.yaml')
        vq=VQVAE(**cfg.tokenizer.vqvae).eval()
        vq.load_state_dict(torch.load(ck/'tokenizer.pt',map_location='cpu',weights_only=True)['model'],strict=True)
        net=NextConceptTransformer(vq,cfg.model).eval()
        net.load_state_dict(torch.load(ck/'generator.pt',map_location='cpu',weights_only=True)['model'],strict=True)
        p,s=boards(d); prefix=torch.tensor(p[:16]); codes=net.generate(None,prefix,temperature=0.0,top_k=0,top_p=1.0,cfg_scale=0.0,cfg_null_eos=None)
        logits=vq.decoder(vq.vq.idx_to_z_q(codes)); ids=logits.argmax(-1)
        port=SudokuModel(ck); pred=port.predict(prefix)
        result[d]=dict(n=16,codes_equal=bool(torch.equal(codes,pred.codes)),outputs_equal=bool(torch.equal(ids,pred.ids)),
            logits_max_abs=float((logits-pred.logits).abs().max()),code_disagreements=int((codes!=pred.codes).sum()),
            output_disagreements=int((ids!=pred.ids).sum()))
        assert result[d]['codes_equal'] and result[d]['outputs_equal']
        progress('upstream_parity',checkpoint=d,**result[d])
    write(ROOT/'results/upstream_parity.json',result)

if __name__=='__main__': main()
