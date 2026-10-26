"""Fresh structural check on randomly initialized toy models.

This intentionally needs no trained checkpoint: whole-memory donor parity is
an identity of the access contract. It is separate from the trained records.
"""
from model import Net, Q, torch, read, write
from runtime import native, exchange


def main():
    row=read('data/families.json')['rows'][0]
    checks=[]
    with torch.inference_mode():
        for architecture in Q['architectures']:
            torch.manual_seed(17)
            net=Net(architecture).eval()
            reference=native(net,row)
            for cut in Q['cuts']:
                for comparison in range(2):
                    result=exchange(net,reference,cut,comparison,'whole')['logits']
                    branches=(0,1) if comparison==0 else (0,2)
                    for recipient in range(2):
                        for donor in range(2):
                            assert torch.equal(result[recipient,donor,2],reference['logits'][branches[donor],-1])
                    checks.append({'architecture':architecture,'cut':cut,'comparison':comparison,'donor_parity':True})
    report={'passed':True,'scope':'fresh untrained-model structural control, not the historical trained evaluation','checks':checks}
    write('results/fresh_whole_memory.json',report)
    import json
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
