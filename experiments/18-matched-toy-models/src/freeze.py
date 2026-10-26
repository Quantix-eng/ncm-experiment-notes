from model import *


if __name__=='__main__':
    assert not (D/'results/protocol_frozen.json').exists()
    net=Net('ar')
    paths=set(D.glob('*.py'))|{D/'PROTOCOLE.md',D/'config.json'}|set((D/'data').glob('*'))
    for module in list(sys.modules.values()):
        name=getattr(module,'__name__','');p=getattr(module,'__file__',None)
        if p and name.startswith(('torch','numpy')) and Path(p).is_file():paths.add(Path(p).resolve())
    write('results/protocol_frozen.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          status='Relocated local freeze before this training run; prior research outcomes are known',
          versions=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__),
          parameters_per_model=sum(p.numel() for p in net.parameters()),
          files={str(p.resolve()):sha(p) for p in sorted(paths)}))
    print('Frozen',len(paths),'files; parameters per model',sum(p.numel() for p in net.parameters()))
