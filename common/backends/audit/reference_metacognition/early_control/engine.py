"""Isolated pause/retain/resume engine, batch8 pooling, unchanged public model."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from features import GROUPS,probability_features,scalar_summary,prompt_cells,build_features
from ncm_mac.sudoku import Prediction

LEVELS=[1,2,4,6,9,13,18,26,37,50,67,89]

def puzzle_features(prefix):
    given=prompt_cells(prefix)
    return np.column_stack([np.eye(10)[given].reshape(len(given),-1),
        np.column_stack([(given[:,u]!=0).sum(1)/9 for u in GROUPS]),
        np.column_stack([(given==d).sum(1)/30 for d in range(1,10)])])

def code_features(prefix,codes,probabilities):
    assert codes.shape[1]==116 and probabilities.shape[1:]==(116,16)
    pieces=[puzzle_features(prefix)];offset=0
    for n in LEVELS[:9]:
        p=probabilities[:,offset:offset+n];c=codes[:,offset:offset+n];offset+=n
        values=[*probability_features(p,16),np.log(np.maximum(np.take_along_axis(p,c[...,None],axis=2)[...,0],1e-12))]
        pieces.extend(scalar_summary(v)for v in values)
    return np.column_stack(pieces)

@torch.inference_mode()
def begin(model,prefix,mode='none'):
    net,vq=model.net,model.vqvae.vq;b,p=prefix.shape
    assert b==8 and list(net.levels)==LEVELS
    assert net.config.attn_pattern=='block_diagonal'and net.query_level0 and net.use_rope
    prefix_h=net.transformer.wte(model.vqvae.encoder.wte(prefix));positions=net.transformer.wle(net.lvl_1L)
    z=torch.zeros((b,89,256));net._set_kv_caching(True)
    capture_codes=mode in ['codes','codes_hidden','summary_archived'];capture_hidden=mode in ['hidden','codes_hidden','summary_archived']
    try:
        mask=torch.zeros((1,1,p,p));rope=(net.rope_cos_gen_0,net.rope_sin_gen_0)
        for block in net.transformer.blocks:prefix_h=block(prefix_h,attn_mask=mask,rope_cos_sin=rope)
        kv=[(block.attn.cached_k,block.attn.cached_v)for block in net.transformer.blocks]
        codes=[];probs=[];snapshots=[];summaries=[];offset=0
        for k,n in enumerate(LEVELS[:9]):
            if k==0:x=net.build_l0_query_input(b,prefix_h.dtype)
            else:
                e=vq.embeddings[k-1](codes[-1]);z,x=vq.get_next_autoregressive_input(k-1,z,e)
                if mode=='summary_archived':snapshots.append(z.clone())
            x=net.transformer.wte(x)+positions[:,offset:offset+n]
            mask=torch.zeros((1,1,n,p+n));rope=(getattr(net,f'rope_cos_gen_{k}'),getattr(net,f'rope_sin_gen_{k}'))
            for block,(pk,pv)in zip(net.transformer.blocks,kv):
                block.attn.cached_k,block.attn.cached_v=pk,pv;x=block(x,attn_mask=mask,rope_cos_sin=rope)
            if capture_hidden and (k==8 or mode=='summary_archived'):
                h=x.double();summaries.append(torch.cat([h.mean(1),h.std(1,correction=0)],dim=1).numpy())
            logits=net.lm_heads[k](net.transformer.ln_f(x));codes.append(logits.argmax(-1));offset+=n
            if capture_codes:probs.append(logits.double().softmax(-1).numpy())
        features={}
        if capture_hidden:features['hidden']=summaries[-1]
        if capture_codes:
            prob=np.concatenate(probs,axis=1);features['codes']=code_features(prefix.numpy(),torch.cat(codes,dim=1).numpy(),prob)
        if mode=='summary_archived':
            e=vq._up_to_full(vq.embeddings[8](codes[-1]),8,89)
            snapshots.append(z.clone().add_(vq._apply_quant_resi(e,8)))
            dl=[model.vqvae.decoder(s)for s in snapshots]
            trace=dict(codes=torch.cat(codes,dim=1).numpy(),code_probs=prob,
                decoder_probs=np.stack([d.double().softmax(-1).numpy()for d in dl],axis=1),
                decoded_ids=np.stack([d.argmax(-1).numpy()for d in dl],axis=1),
                hidden_summary=np.stack(summaries,axis=1))
            features['summary']=build_features(prefix.numpy(),trace,9,LEVELS)['summary']
        return dict(z=z,codes=codes,kv=kv,prefix=prefix),features
    finally:net._set_kv_caching(False)


def rows(state):
    return [dict(z=state['z'][i:i+1],codes=[c[i:i+1]for c in state['codes']],
        kv=[(k[i:i+1],v[i:i+1])for k,v in state['kv']],prefix=state['prefix'][i:i+1])for i in range(8)]


def pack(states):
    assert len(states)==8
    return dict(z=torch.cat([s['z']for s in states]),codes=[torch.cat([s['codes'][k]for s in states])for k in range(9)],
        kv=[tuple(torch.cat([s['kv'][k][j]for s in states])for j in range(2))for k in range(8)],
        prefix=torch.cat([s['prefix']for s in states]))

@torch.inference_mode()
def finish(model,state):
    net,vq=model.net,model.vqvae.vq;z=state['z'].clone();codes=list(state['codes']);kv=state['kv']
    b,p=state['prefix'].shape;assert b==8
    positions=net.transformer.wle(net.lvl_1L);offset=116;net._set_kv_caching(True)
    try:
        for k in range(9,12):
            n=LEVELS[k];e=vq.embeddings[k-1](codes[-1]);z,x=vq.get_next_autoregressive_input(k-1,z,e)
            x=net.transformer.wte(x)+positions[:,offset:offset+n]
            mask=torch.zeros((1,1,n,p+n));rope=(getattr(net,f'rope_cos_gen_{k}'),getattr(net,f'rope_sin_gen_{k}'))
            for block,(pk,pv)in zip(net.transformer.blocks,kv):
                block.attn.cached_k,block.attn.cached_v=pk,pv;x=block(x,attn_mask=mask,rope_cos_sin=rope)
            logits=net.lm_heads[k](net.transformer.ln_f(x));codes.append(logits.argmax(-1));offset+=n
        z.add_(vq._apply_quant_resi(vq.embeddings[11](codes[-1]),11));logits=model.vqvae.decoder(z)
        return Prediction(torch.cat(codes,dim=1),z,logits,logits.argmax(-1))
    finally:net._set_kv_caching(False)
