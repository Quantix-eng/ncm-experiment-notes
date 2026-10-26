"""Two independent exact solvers: MRV backtracking and Algorithm X exact cover."""
from common import *

def backtracking(board,limit=2,rng=None):
    board=list(board); found=[]
    if any(len([board[i] for i in u if board[i]])!=len({board[i] for i in u if board[i]}) for u in UNITS): return []
    def search():
        best=None; choices=None
        for i,v in enumerate(board):
            if v: continue
            can=sorted(set(range(1,10))-{board[j] for j in PEERS[i]})
            if not can: return
            if choices is None or len(can)<len(choices): best,choices=i,can
        if best is None: found.append(board.copy()); return
        if rng: rng.shuffle(choices)
        for v in choices:
            board[best]=v; search()
            if len(found)>=limit: break
        board[best]=0
    search(); return found

def exact_cover_count(board,limit=2):
    cols={i:set() for i in range(324)}; rowcols={}
    for r in range(9):
        for c in range(9):
            for d in ([int(board[r*9+c])] if board[r*9+c] else range(1,10)):
                row=(r*9+c)*9+d-1
                keys=(r*9+c,81+r*9+d-1,162+c*9+d-1,243+(r//3*3+c//3)*9+d-1)
                rowcols[row]=keys
                for key in keys: cols[key].add(row)
    count=0
    def search(active):
        nonlocal count
        if not active: count+=1; return
        key=min(active,key=lambda k:(len(active[k]),k))
        for row in sorted(active[key]):
            covered=rowcols[row]; conflicts=set().union(*(active[c] for c in covered))
            search({c:rows-conflicts for c,rows in active.items() if c not in covered})
            if count>=limit: return
    search(cols); return count
