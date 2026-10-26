"""Avoid concurrent Sudoku generation and the larger LM1B model in local RAM."""
from pathlib import Path
import time,sys,subprocess
D=Path(__file__).resolve().parents[1];deadline=time.monotonic()+3600
while not (D/'results/sudoku_test.npz').exists():
    if time.monotonic()>deadline:raise TimeoutError('Sudoku results not completed')
    time.sleep(5)
subprocess.run([sys.executable,'-u',str(D/'scripts/text_audit.py'),'--task','lm1b'],check=True)
