# Round 2 (pre-registered 2026-10-04, before any round-2 result exists)

Question: is deviation D22 (two-hot expectation: Ajax's exact mirror-difference
form vs the reference's literal mirror-pair sum, which XLA contracts into FMAs
under jit) the whole difference between Ajax and the reference on the round-1
CartPole protocol (PROTOCOL.md; round-1 results in full/)?

Two like-for-like pairs, same protocol, seeds 0 1 2, 20000 rows, eval every
2000 rows with 10 episodes:

- Pair E (exact decode on both sides): reference + exact form
  (run_reference_exact.py, new, full2/ref_s*) vs Ajax round-1 (full/ajax,
  5cb0738, exact form natively, 3-seed vmap).
- Pair R (reference decode on both sides): Ajax 5cb0738 + reference decode
  (run_ajax_refdecode.py, new, full2/ajax, 3-seed vmap, from the frozen
  worktree wt-m7-frozen) vs reference round-1 (full/ref_s*).

Statistics and rule, unchanged from round 1 (PROTOCOL.md section 5): S1 = per
seed mean eval return at 16000/18000/20000 rows; S2 = Ajax-definition
training-episode statistic at 20000; LEARNS = mean Ajax S1 >= min reference
S1; WITHIN_RANGE = min <= mean Ajax S1 <= max. Reported for both pairs.

Added for S-B (entropy): for each 2000-row window from 4000 to 20000, whether
the Ajax and reference seed ranges of ent/action/mean overlap; round 1 had
non-overlapping ranges at 4000, 6000, 8000 and 12000.

Reading: D22 is "the whole difference" if, in both pairs, WITHIN_RANGE holds
or the S1 gap is smaller than in round 1, and the entropy ranges overlap in
most windows. Anything else means another difference remains and is
investigated before M7 is called faithful.
