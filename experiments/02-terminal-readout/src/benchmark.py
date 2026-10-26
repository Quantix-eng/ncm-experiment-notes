"""Paired AB/BA CPU timing after all other experiment processes have completed."""
from base import *
from policy import decode_native,decode_certified
from ncm_mac.cached import generate_cached

@torch.inference_mode()
def main():
    a=np.load(D/'results/hard_readout.npz');model=SudokuModel(ROOT/'artifacts/sudoku/hard');mapping=np.array(json.loads((D/'results/discovery_tables.json').read_text())['hard']['mapping']);N=Q['benchmark_n'];rng=np.random.default_rng(Q['seed']);raw=[];stage=[];checks=[]
    def full(p,certified):
        codes,z=generate_cached(model.net,torch.tensor(p))
        return decode_certified(model,z,codes,p,mapping) if certified else decode_native(model,z)
    # Warm both arms; none of these latencies is retained.
    for i in range(4):
        p=a['prefix'][i:i+1];full(p,False);full(p,True)
    for rep in range(Q['benchmark_repeats']):
        order=rng.permutation(N)
        for i in order:
            p=a['prefix'][i:i+1];ab=rng.permutation(2);times={};outputs={}
            for arm in ab:
                t=time.perf_counter_ns();outputs[int(arm)]=full(p,bool(arm));times[int(arm)]=time.perf_counter_ns()-t
            assert np.array_equal(outputs[0],a['native'][i:i+1]) and np.array_equal(outputs[1],a['hybrid'][i:i+1]);raw.append(dict(puzzle=int(i),repeat=rep,first_arm=int(ab[0]),native_ns=times[0],certified_ns=times[1]))
        log('benchmark_full',repeat=rep+1,repeats=Q['benchmark_repeats'])
    for i in rng.permutation(N):
        p=a['prefix'][i:i+1];z=torch.tensor(a['zfinal'][i:i+1]);codes=torch.tensor(a['codes'][i:i+1]);ab=rng.permutation(2);times={}
        for arm in ab:
            start=time.perf_counter_ns()
            for _ in range(Q['decoder_inner_repeats']):
                output=decode_certified(model,z,codes,p,mapping) if arm else decode_native(model,z)
            times[int(arm)]=(time.perf_counter_ns()-start)/Q['decoder_inner_repeats']
            assert np.array_equal(output,a['hybrid'][i:i+1] if arm else a['native'][i:i+1])
        stage.append(dict(puzzle=int(i),first_arm=int(ab[0]),native_ns=times[0],certified_ns=times[1]))
    report={}
    for name,rows in [('end_to_end',raw),('readout_only',stage)]:
        logs=np.array([np.mean([np.log(r['native_ns']/r['certified_ns']) for r in rows if r['puzzle']==i]) for i in range(N)]);b=bootstrap(logs);report[name]=dict(mean_native_ms=float(np.mean([r['native_ns'] for r in rows])/1e6),mean_certified_ms=float(np.mean([r['certified_ns'] for r in rows])/1e6),log_speed_ratio=b,geometric_speed_ratio=float(np.exp(b['mean'])),geometric_speed_ratio_ci95=np.exp(b['ci95']),geometric_latency_reduction=float(1-np.exp(-b['mean'])),n_source_puzzles=N)
    write(D/'results/benchmark_raw.json',dict(full=raw,stage=stage,clock='perf_counter_ns',threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),warmups=4,order='randomized AB/BA within pair; randomized puzzle order each repeat',acceptance_first64=float(a['accept'][:N].mean())))
    write(D/'results/benchmark.json',report);log('benchmark_complete',**report)
if __name__=='__main__':main()
