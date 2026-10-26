"""Replay the frozen decoder at level9 from prefix codes only, training only."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from features import CELL_POS, prompt_cells
from run_pilot import ROOT, SudokuModel, file_hash, load_boards, state_hash, write_json
from cell_features import visible_features


@torch.inference_mode()
def local_readout(model, codes, level=9):
    vq = model.vqvae.vq
    lengths = list(model.net.levels)
    assert codes.shape[1] == sum(lengths[:level]), 'Only current and past codes are allowed'
    running = torch.zeros((len(codes),89,256),dtype=torch.float32)
    offset = 0
    for k,length in enumerate(lengths[:level]):
        h = vq.embeddings[k](codes[:,offset:offset+length])
        h = vq._up_to_full(h,k,89)
        running.add_(vq._apply_quant_resi(h,k))
        offset += length
    captured = []
    handle = model.vqvae.decoder.output_layer.register_forward_pre_hook(lambda module,args: captured.append(args[0].detach().clone()))
    try: logits = model.vqvae.decoder(running)
    finally: handle.remove()
    assert len(captured)==1 and captured[0].shape==(len(codes),89,256)
    assert torch.equal(model.vqvae.decoder(running),logits), 'Hook changed decoder outputs'
    return captured[0].numpy(), logits.numpy()


def main():
    torch.set_num_threads(4)
    directory = ROOT/'audit/metacognition/correction-v0.1/training'
    directory.mkdir(exist_ok=False)
    protocol_path = Path(__file__).with_name('protocol-v0.1.json')
    split = json.loads((ROOT/'audit/metacognition/split-v0.2.json').read_text())
    selected = split['partition_indices']['train']
    source = ROOT/'audit/metacognition/train-v0.2'
    extraction = json.loads((source/'report.json').read_text())
    solutions = load_boards(ROOT/'data/sudoku/hard/solutions.pickle')
    solutions = {i: np.asarray(solutions[i])[CELL_POS] for i in selected}
    model = SudokuModel(ROOT/'artifacts/sudoku/hard')
    before = state_hash(model)
    parts, records = {}, []
    start = time.perf_counter()
    for number,batch in enumerate(extraction['batches']):
        path = source/batch['file']
        assert file_hash(path)==batch['sha256']
        with np.load(path,allow_pickle=False) as trace:
            hidden, logits = local_readout(model,torch.from_numpy(trace['codes'][:,:116].astype(np.int64)))
            probabilities = torch.from_numpy(logits).double().softmax(-1).numpy()
            assert np.array_equal(probabilities,trace['decoder_probs'][:,8])
            blank = prompt_cells(trace['prefix'])==0
            visible = visible_features(trace['prefix'],trace,logits)
            current = trace['decoded_ids'][:,8,CELL_POS]
            final = trace['decoded_ids'][:,11,CELL_POS]
            gold = np.stack([solutions[i] for i in trace['sample_indices']])
            ids = np.broadcast_to(trace['sample_indices'][:,None],blank.shape)
            cell = np.broadcast_to(np.arange(81),blank.shape)
            block = dict(visible=visible[blank],hidden=hidden[:,CELL_POS][blank],group=ids[blank],cell=cell[blank],
                         current_error=(current!=gold)[blank],persistent=(final!=gold)[blank])
            for key,values in block.items(): parts.setdefault(key,[]).append(values)
            records.append(dict(indices=trace['sample_indices'].tolist(),decoder_probabilities_bitwise_equal=True))
        if number%25==0: print(f'local training replay {number+1}/125',flush=True)
    assert state_hash(model)==before
    for key,values in parts.items(): np.save(directory/f'{key}.npy',np.concatenate(values))
    write_json(directory/'report.json',dict(status='complete',scope='training_only',protocol_sha256=file_hash(protocol_path),
        source_files={p.name:file_hash(p) for p in [Path(__file__),Path(__file__).with_name('cell_features.py')]},
        extraction_report_sha256=file_hash(source/'report.json'),batches=records,model_state_sha256_before_and_after=before,
        arrays={p.name:dict(sha256=file_hash(p),bytes=p.stat().st_size)for p in directory.glob('*.npy')},
        n_groups=1000,n_blank_cells=51000,wall_seconds=time.perf_counter()-start))
    print('Training local decoder readout complete',flush=True)

if __name__=='__main__':main()
