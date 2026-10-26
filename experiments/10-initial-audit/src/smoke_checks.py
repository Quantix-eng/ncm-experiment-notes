"""Small pre-analysis checks of labels, metrics and instrumentation contracts."""
from common import *
from solvers import backtracking,exact_cover_count
from analyze import score_metrics,paired_auc,fit_probe,token_stats
from prepare import symkey

def main():
    checks=[]
    def ck(name,value):
        assert value,name; checks.append(name)
    solved=np.array([(r*3+r//3+c)%9+1 for r in range(9) for c in range(9)])
    ck('independent_exact_solver_valid_grid',exact_cover_count(solved)==1)
    ck('independent_backtracking_valid_grid',backtracking(solved)==[solved.tolist()])
    bad=solved.copy();bad[0]=bad[1]
    ck('duplicate_grid_rejected',exact_cover_count(bad)==0 and backtracking(bad)==[])
    puzzle=solved.copy();puzzle[0]=0
    ck('candidate_target_single',candidate_counts(puzzle)[0]==1)
    mapping=np.r_[0,np.arange(9,0,-1)]
    ck('group_key_digit_invariant',symkey(puzzle)==symkey(mapping[puzzle]))
    ck('group_key_rotation_invariant',symkey(puzzle)==symkey(np.rot90(puzzle.reshape(9,9)).ravel()))
    yy=np.array([0,0,1,1]);constant=np.full(4,.5)
    ck('constant_auc_half',score_metrics(yy,constant)['auc']==.5)
    ck('constant_brier',score_metrics(yy,constant)['brier']==.25)
    ck('tied_coverage_whole_group',len(score_metrics(yy,constant)['risk_coverage'])==1)
    stat=token_stats(np.tile(np.array([[0,1]]),(20,1)),[2])[0]
    ck('position_entropy_not_global_entropy',abs(stat['entropy_nats']-np.log(2))<1e-12 and stat['conditional_entropy_given_position_nats']==0)
    write(ROOT/'results/smoke_checks.json',dict(status='passed',checks=checks))
    print(json.dumps(dict(status='passed',checks=checks)))

if __name__=='__main__':main()
