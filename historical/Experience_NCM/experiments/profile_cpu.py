import sys,json
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'src'))
import torch
from ncm_mac.sudoku import SudokuModel,load_boards
torch.set_num_threads(4)
m=SudokuModel(root/'artifacts/sudoku/easy');boards=load_boards(root/'data/sudoku/easy/puzzles.pickle')
for batch in [1,8]:
 p=m.prefix(boards[:batch]);m.predict(p)
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as prof:m.predict(p)
 print('BATCH',batch);print(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=18))
 rows=[dict(op=x.key,self_cpu_us=x.self_cpu_time_total,total_cpu_us=x.cpu_time_total,calls=x.count) for x in prof.key_averages()]
 (root/f'audit/cpu-profile-b{batch}.json').write_text(json.dumps(rows,indent=2)+'\n')
