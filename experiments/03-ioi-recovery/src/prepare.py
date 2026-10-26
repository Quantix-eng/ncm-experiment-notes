"""Data-only selection: never inspect model predictions."""
from support import *

if __name__=='__main__':
    # Loading the tokenizer is not an execution of the neural model.
    tok = GPT2TokenizerFast.from_pretrained(str(ROOT/'artifacts/gpt2'), local_files_only=True)
    rng = np.random.default_rng(2026102517)
    pools = {s:read('data/official_'+s+'.json') for s in ('dev','test')}
    selected = {}; exclusions = {}
    # Reject exact source/variant overlap against development, before sampling test.
    dev_prompts = {v['prompt'] for row in pools['dev'] for v in
                  (row,row['s2_io_flip_counterfactual'],row['s1_io_flip_counterfactual'])}
    for split, count in [('dev',128),('test',512)]:
        eligible=[]; rejected=[]; seen=set()
        for index,row in enumerate(pools[split]):
            variants=[row,row['s2_io_flip_counterfactual'],row['s1_io_flip_counterfactual']]
            reason=None
            if [v['answerKey'] for v in variants]!=[0,1,0]: reason='unexpected_labels'
            if any(v['choices']!=row['choices'] for v in variants): reason='changed_candidate_order'
            choices=[tok.encode(' '+name,add_special_tokens=False) for name in row['choices']]
            if any(len(c)!=1 for c in choices): reason='multitoken_candidate'
            prompts=[v['prompt'] for v in variants]
            if split=='test' and set(prompts)&dev_prompts: reason='exact_overlap_dev'
            if set(prompts)&seen: reason='duplicate_source_or_variant'
            tokens=[[tok.bos_token_id]+tok.encode(p,add_special_tokens=False) for p in prompts]
            if len({len(t) for t in tokens})!=1: reason='variant_token_lengths'
            if reason: rejected.append(dict(index=index,reason=reason)); continue
            seen.update(prompts)
            eligible.append(dict(id=f'{split}-{index:05d}',index=index,template=row['template'],
                choices=[c[0] for c in choices],names=row['choices'],tokens=tokens,
                prompts=prompts,answer_keys=[0,1,0]))
        assert len(eligible)>=count, (split,len(eligible))
        indices=rng.choice(len(eligible),size=count,replace=False)
        selected[split]=[eligible[int(i)] for i in indices]
        exclusions[split]=dict(total=len(pools[split]),eligible=len(eligible),excluded=rejected,
            selected=count,templates={t:sum(r['template']==t for r in selected[split])
                                      for t in sorted({r['template'] for r in selected[split]})})
    write('data/selected.json',selected)
    write('data/data_audit.json',dict(exclusions=exclusions,bos=True,
        candidate_encoding='one token with leading space',variant_order=['native','s2_io_flip','s1_io_flip'],
        test_status='tokenization and metadata only, no model outputs'))
    print(json.dumps({s:{k:v for k,v in x.items() if k!='excluded'} for s,x in exclusions.items()},indent=2))
