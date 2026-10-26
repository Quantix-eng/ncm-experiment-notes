"""Disclosed post-hoc reporting of paired argmax changes; no model inference."""
from support import *
import importlib.metadata as metadata


if __name__ == '__main__':
    check_freeze()
    assert not (D/'results/paired_descriptive.json').exists()
    native = np.load(D/'results/native.npz')
    crossed = np.load(D/'results/crossed.npz')
    exchanges = np.load(D/'results/exchanges.npz')
    refs = native['references']
    ids = crossed['logits'].argmax(-1)
    correct = ids == refs[:,:2,None,None]
    pairs = {}
    for k,name in enumerate(('Y00','Y10','Y11','Y01','Y1wrong')):
        pairs[name] = dict(losses_vs_Y00=int((correct[...,0] & ~correct[...,k]).sum()),
                          gains_vs_Y00=int((~correct[...,0] & correct[...,k]).sum()),
                          argmax_changes_vs_Y00=int((ids[...,0] != ids[...,k]).sum()))
    gains,losses = [],[]
    logits = exchanges['logits']
    rows = read('data/families.json')['rows']
    for f in range(16):
        for recipient in range(2):
            donor = 1-recipient
            regen = int(logits[f,0,recipient,donor,2].argmax()) == refs[f,recipient]
            frozen = int(logits[f,0,recipient,donor,recipient].argmax()) == refs[f,recipient]
            if regen and not frozen:
                gains.append(dict(source_id=rows[f]['source_id'],recipient=recipient))
            if frozen and not regen:
                losses.append(dict(source_id=rows[f]['source_id'],recipient=recipient))
    write('results/paired_descriptive.json',dict(
        status='Post-hoc descriptive export added after primary results; no new model calls or label changes',
        C=pairs,E_regenerated_vs_frozen_recipient=dict(gains=gains,losses=losses,n=32)))
    write('results/environment_packages.json',dict(
        actual_numerical_versions=dict(numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__),
        isolated_target_distributions={dist.metadata['Name']:dist.version
                                      for dist in metadata.distributions(path=[str(D/'deps')])},
        note='NumPy and PyTorch are imported before target dependencies; installed target NumPy is not used'))
    print(json.dumps(read('results/paired_descriptive.json'),indent=2))
