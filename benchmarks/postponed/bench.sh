#!/bin/bash
# Sequential before/after benchmarks for the perf branches (base c97630d).
S=/private/tmp/claude-501/-Users-yannberthelot-Documents-Ajax/6affcbf6-0748-4e5f-ae30-aa95277a6f34/scratchpad; B=/private/tmp/claude-501/-Users-yannberthelot-Documents-Ajax/6affcbf6-0748-4e5f-ae30-aa95277a6f34/scratchpad/bench_runs; PY=$S/clone_BATCH/.venv/bin/python
export JAX_PLATFORMS=cpu AJAX_NO_COMPILE_CACHE=1
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $B/log.txt; }
ab() { # ab TAG DIR AGENT [extra args]
  local tag=$1 dir=$2 agent=$3; shift 3
  (cd $dir && PYTHONPATH=$dir/src:$dir $PY benchmarks/agent_bench.py --only $agent --tag $tag --trials 3 --out $B/$tag.jsonl "$@" >> $B/agent_bench.log 2>&1)
  log "done $tag"
}
log "start (load: $(uptime | sed 's/.*averages: //'))"
ab td3_base $S/bench_base TD3;            ab td3_new $S/b12_td3-q1 TD3
ab apg_base $S/bench_base APG;            ab apg_new $S/b12_model-ref APG
ab udrl_base $S/bench_base UDRL;          ab udrl_new $S/perf_udrl-stream UDRL
ab ppo_base $S/bench_base PPO;            ab ppo_new $S/perf_ppo-gae PPO
ab ppo32_base $S/bench_base PPO --n-envs 32; ab ppo32_new $S/perf_ppo-gae PPO --n-envs 32
ab apo_base $S/bench_base APO;            ab apo_new $S/perf_ppo-gae APO
ab pqn_base $S/bench_base PQN;            ab pqn_new $S/perf_ppo-gae PQN
ab tdmpc2_base $S/bench_base TDMPC2;      ab tdmpc2_new $S/b12_row-trace TDMPC2
ab redq_base $S/bench_base REDQ;          ab redq_new $S/perf_redq-subset REDQ
for v in base new; do d=$S/bench_base; [ $v = new ] && d=$S/b12_row-trace
  (cd $d && PYTHONPATH=$d/src:$d $PY $S/b12_meas/lower_tdmpc2.py with 2>&1 | grep -iv warn | tail -2 | sed "s/^/lower_$v: /" >> $B/log.txt); done
for v in base new; do d=$S/bench_base; [ $v = new ] && d=$S/b12_eval-while
  (cd $S/b12_eval-while && PYTHONPATH=$d/src:$S/b12_eval-while $PY $B/evalwhile_time.py 2>&1 | grep -iv warn | tail -1 | sed "s/^/evalwhile_$v: /" >> $B/log.txt); done
for n in SAC DreamerV3 TDMPC2; do (cd $S/b12_skeleton && PYTHONPATH=$S/b12_skeleton/src:$S/b12_skeleton $PY $S/b12_work/premise.py $n 2>&1 | grep -iv warn | tail -1 | sed "s/^/skeleton: /" >> $B/log.txt); done
for seeds in 1 8; do for v in base new; do d=$S/bench_base; [ $v = new ] && d=$S/perf_kv-cache
  (cd $d && PYTHONPATH=$d/src:$d $PY $B/kv_time.py $seeds 2>&1 | grep -iv warn | tail -1 | sed "s/^/kv_$v: /" >> $B/log.txt); done; done
for t in td3 apg udrl ppo ppo32 apo pqn tdmpc2 redq; do
  echo "== $t" >> $B/compare.txt
  $PY $S/bench_base/benchmarks/agent_bench.py --compare $B/${t}_base.jsonl $B/${t}_new.jsonl >> $B/compare.txt 2>&1
done
log "end (load: $(uptime | sed 's/.*averages: //'))"
