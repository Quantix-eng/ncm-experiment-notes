"""Freeze before inference, then collect all predeclared O/C/E outcomes."""
from support import *
from runtime import Engine


def freeze():
    assert not (D/'results/protocol_frozen.json').exists()
    # Load libraries/model without a forward pass so their imported sources are pinned.
    net = model()
    files = set(D.glob('*.py')) | {D/'PROTOCOLE.md', D/'protocol.json', D/'data/families.json'}
    files |= {p for p in (ROOT/'artifacts/gpt2').glob('*') if p.is_file()}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        name = getattr(module, '__name__', '')
        if path and name.startswith(('torch', 'numpy', 'transformers', 'tokenizers', 'safetensors', 'huggingface_hub')):
            p = Path(path).resolve()
            if p.is_file():
                files.add(p)
    for name in ('benchmark_ncm_reel/results/summary.json', 'benchmark_ncm_reel/results/output_quality.json',
                 'benchmark_ncm_reel/results/final_manifest.json', 'article_benchmark_coordination/results/final_manifest.json'):
        files.add(ROOT/'analyse_fondamentale_NCM'/name)
    write('results/protocol_frozen.json', dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          status='Local pre-output freeze; no external registration',
          model_parameters=sum(p.numel() for p in net.parameters()),
          versions=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, transformers=transformers.__version__),
          files={str(p.resolve()):sha(p) for p in sorted(files)}))
    print('Frozen',len(files),'files; no inference',flush=True)


@torch.inference_mode()
def run():
    check_freeze()
    assert not (D/'results/native.npz').exists(), 'Refusing to overwrite observations'
    setup()
    start = time.monotonic()
    net = model()
    engine = Engine(net)
    rows = read('data/families.json')['rows']
    contexts, native_logits = [], []
    for row in rows:
        context_row, logits_row = [], []
        for tokens in row['tokens']:
            context, logits = engine.native(tokens)
            context_row.append(context)
            logits_row.append(logits.numpy())
        contexts.append(context_row)
        native_logits.append(logits_row)
    np.savez_compressed(D/'results/native.npz', logits=np.asarray(native_logits),
                        full=np.asarray([[ctx['full'].numpy() for ctx in group] for group in contexts]),
                        tails=np.asarray([[ctx['tail'].numpy() for ctx in group] for group in contexts]),
                        tokens=np.asarray([r['tokens'] for r in rows]), references=np.asarray([r['reference'] for r in rows]))
    print('Native parity and shams: 48/48',flush=True)
    vocab = net.config.vocab_size
    deltas = np.zeros((16,2,4,768), dtype=np.float32)
    c_logits = np.zeros((16,2,4,5,vocab), dtype=np.float32)
    c_states = np.zeros((16,2,4,5,768), dtype=np.float32)
    c_tails = np.zeros((16,2,4,12,768), dtype=np.float32)
    rng = np.random.default_rng(read('protocol.json')['seed'])
    for f in range(16):
        for anchor in range(2):
            ctx = contexts[f][anchor]
            z = ctx['full'][6,0,-1]
            directions = []
            target = .1*float(torch.linalg.vector_norm(z.double()))
            for support in range(2):
                direction = np.zeros(768)
                direction[support*384:(support+1)*384] = rng.normal(size=384)
                direction *= target/np.linalg.norm(direction)
                value = torch.tensor(direction, dtype=torch.float32)
                assert abs(float(torch.linalg.vector_norm(value.double()))/target-1) < 2e-6
                directions.extend((value, -value))
            adapted = [engine.continue_from(ctx,z+delta) for delta in directions]
            for q, delta in enumerate(directions):
                tails = (ctx['tail'], ctx['tail'], adapted[q]['tail'], adapted[q]['tail'], adapted[q^2]['tail'])
                initial = (z,z+delta,z+delta,z,z+delta)
                for condition, (state, tail) in enumerate(zip(initial,tails)):
                    final = engine.frozen(state,tail)
                    c_states[f,anchor,q,condition] = final.numpy()
                    c_logits[f,anchor,q,condition] = engine.decode(ctx,final).numpy()
                assert np.array_equal(c_logits[f,anchor,q,0], native_logits[f][anchor][-1])
                deltas[f,anchor,q] = delta.numpy()
                c_tails[f,anchor,q] = adapted[q]['tail'].numpy()
        print('C',f+1,'/16',flush=True)
    np.savez_compressed(D/'results/crossed.npz', logits=c_logits, states=c_states, tails=c_tails, deltas=deltas)
    del c_logits
    e_logits = np.zeros((16,2,2,2,3,vocab), dtype=np.float32)
    e_states = np.zeros((16,2,2,2,3,768), dtype=np.float32)
    e_tails = np.zeros((16,2,2,2,12,768), dtype=np.float32)
    for f in range(16):
        for comparison, branches in enumerate(((0,1),(0,2))):
            for recipient in range(2):
                ctx = contexts[f][branches[recipient]]
                for donor in range(2):
                    z = contexts[f][branches[donor]]['full'][6,0,-1]
                    regenerated = engine.continue_from(ctx,z)
                    tails = (contexts[f][branches[0]]['tail'], contexts[f][branches[1]]['tail'], regenerated['tail'])
                    for policy, tail in enumerate(tails):
                        state = engine.frozen(z,tail)
                        e_states[f,comparison,recipient,donor,policy] = state.numpy()
                        e_logits[f,comparison,recipient,donor,policy] = engine.decode(ctx,state).numpy()
                    e_tails[f,comparison,recipient,donor] = regenerated['tail'].numpy()
                    if recipient == donor:
                        expected = native_logits[f][branches[recipient]][-1]
                        assert np.array_equal(e_logits[f,comparison,recipient,donor,2],expected)
        print('E',f+1,'/16',flush=True)
    np.savez_compressed(D/'results/exchanges.npz',logits=e_logits,states=e_states,tails=e_tails)
    engine.close()
    check_freeze()
    write('results/execution.json',dict(status='complete', native_parity=48, shams=48,
          C_trials=128, E_conditions=384, elapsed_seconds=time.monotonic()-start,
          cache_policy='No mutable KV cache; recipient earlier-token representations retained',
          scope='Next-token within-forward intervention, not emitted-token revision'))


if __name__ == '__main__':
    if sys.argv[1:] == ['--freeze']:
        freeze()
    else:
        assert not sys.argv[1:]
        run()
