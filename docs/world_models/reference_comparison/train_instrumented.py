# fmt: off
# ruff: noqa: ARG001, ARG005, C901, E731, I001
#
# Derived from danijar/dreamerv3 at commit 29eb964
# (29eb964e2918a3f4db04086f7f51b60388e97f3d): a copy of the whole file
# embodied/run/train.py (lines 1-128), for the Ajax-vs-reference CartPole
# comparison (PROTOCOL.md section 3.2 and README.md in this directory).
#
# Modified. Every stock line is unchanged; the only additions are the blocks
# marked "# INSTRUMENTATION" (a module-level hook object, a per-iteration
# progress read, the record call at multiples of record_every steps, and the
# post-loop flush step + final file writes). The hook only reads state (agg,
# agent.updates, agent.params, driver.acts) and steps separate evaluation envs;
# see run_reference.py. The "fmt: off" and "ruff: noqa" lines above keep the
# stock text (two-space indentation, the lambda) as it is upstream, so that
# `diff` against the upstream file shows only the INSTRUMENTATION blocks.
#
# Copyright (c) 2023 Danijar Hafner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The license text is also in LICENSE.dreamerv3 in this directory.
import re
from collections import defaultdict
from functools import partial as bind

import embodied
import numpy as np


# INSTRUMENTATION: hook object set by run_reference.py before calling train().
# It must provide record_every, progress(step, agent), record(step, agent=,
# agg=, logger=), flush(driver) and finish().
INSTR = None


def train(make_agent, make_replay, make_env, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = embodied.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = embodied.Usage(**args.usage)
  agg = embodied.Agg()
  epstats = embodied.Agg()
  episodes = defaultdict(embodied.Agg)
  policy_fps = embodied.FPS()
  train_fps = embodied.FPS()

  batch_steps = args.batch_size * (args.batch_length - args.replay_context)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_eval = embodied.when.Clock(args.eval_every)
  should_save = embodied.when.Clock(args.save_every)

  @embodied.timer.section('log_step')
  def log_step(tran, worker):

    episode = episodes[worker]
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')

    if tran['is_first']:
      episode.reset()

    if worker < args.log_video_streams:
      for key in args.log_keys_video:
        if key in tran:
          episode.add(f'policy_{key}', tran[key], agg='stack')
    for key, value in tran.items():
      if re.match(args.log_keys_sum, key):
        episode.add(key, value, agg='sum')
      if re.match(args.log_keys_avg, key):
        episode.add(key, value, agg='avg')
      if re.match(args.log_keys_max, key):
        episode.add(key, value, agg='max')

    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.num_envs)]
  driver = embodied.Driver(fns, args.driver_parallel)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(log_step)

  dataset_train = iter(agent.dataset(bind(
      replay.dataset, args.batch_size, args.batch_length)))
  dataset_report = iter(agent.dataset(bind(
      replay.dataset, args.batch_size, args.batch_length_eval)))
  carry = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def train_step(tran, worker):
    if len(replay) < args.batch_size or step < args.train_fill:
      return
    for _ in range(should_train(step)):
      with embodied.timer.section('dataset_next'):
        batch = next(dataset_train)
      outs, carry[0], mets = agent.train(batch, carry[0])
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      agg.add(mets, prefix='train')
  driver.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.replay = replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we just saved.

  print('Start training loop')
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10)

    if should_eval(step) and len(replay):
      mets, _ = agent.report(next(dataset_report), carry_report)
      logger.add(mets, prefix='report')

    if should_log(step):
      logger.add(agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(embodied.timer.stats(), prefix='timer')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.write()

    if should_save(step):
      checkpoint.save()

    # INSTRUMENTATION (1): progress read every iteration (time, step, updates;
    # reads only) and record(N) after the vector step that brings step to a
    # multiple of record_every (training metrics window, evaluation, files).
    if INSTR is not None:
      INSTR.progress(int(step), agent)
      if int(step) % INSTR.record_every == 0:
        INSTR.record(int(step), agent=agent, agg=agg, logger=logger)

  # INSTRUMENTATION (2): flush env step (episodes whose is_last row is the
  # row the next vector step would emit; no policy call, no replay.add, no
  # training, no callbacks), then the final file writes.
  if INSTR is not None:
    INSTR.flush(driver)
    INSTR.finish()

  logger.close()
