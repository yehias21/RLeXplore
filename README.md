# RLeXplore

Exploration algorithms in reinforcement learning, implemented against the taxonomy of
Amin, Gomrokchi, Satija, van Hoof, Precup, *A Survey of Exploration Methods in
Reinforcement Learning* (2021). Value-based, discrete-action agents on MiniGrid.

![demo](dqn.gif)

## Layout

The project is organised around five SOLID concerns:

```
rlexplore/
  core/           Protocols (ActionSelector, IntrinsicRewardSource, ...) + plugin registry
  envs/           Environment adapters; preprocessing split from env creation (SRP)
  models/         Q-networks (MLP, NoisyMLP, Bootstrapped)
  memory/         Replay buffer
  exploration/    Strategies, one family per file
  agents/         DQN agent composes model + strategy + memory via DI
  training/       Trainer and Evaluator
  logging_/       Stdout + Comet loggers behind a common Protocol
  config.py       Declarative experiment config
  cli.py          rlexplore --config ... --mode {train,eval,both}
```

New strategies/envs/models register themselves (`@STRATEGIES.register("name")`)
and become addressable from config without edits to the trainer (OCP).

## Implemented strategies

Mapped to the survey's taxonomy (section references in parens):

| key | strategy | survey | reward? |
|---|---|---|---|
| `epsilon_greedy` | ε-greedy, exp. decay | §4.1 eq. 14 | extrinsic |
| `epsilon_first` | ε-first | §4.1 Tran-Thanh 2010 | extrinsic |
| `decaying_epsilon` | linear-decay ε | §4.1 Caelen 2007 | extrinsic |
| `ez_greedy` | εz-greedy (temporally extended) | §4.1 Dabney 2020 | extrinsic |
| `boltzmann` | softmax(Q/T) | §5.1 eq. 20 | extrinsic |
| `max_boltzmann` | ε-greedy + Boltzmann fallback | §5.1 Wiering 1999 | extrinsic |
| `vdbe` | state-dependent ε from TD-error | §5.1 Tokic 2010 | extrinsic |
| `vdbe_softmax` | VDBE + Boltzmann fallback | §5.1 Tokic & Palm 2011 | extrinsic |
| `pursuit` | pursuit algorithm | §5.1 Thathachar 1984 | extrinsic |
| `count_based` | β/√N(s) bonus | §6.2 | extrinsic + intrinsic |
| `hash_count` | SimHash pseudo-count | §6.2 Tang 2017 | extrinsic + intrinsic |
| `ucb_q` | UCB-H on Q | §6.1 Jin 2018 | extrinsic + intrinsic |
| `rnd` | Random Network Distillation | §6.3 Burda 2018b | extrinsic + intrinsic |
| `icm` | Intrinsic Curiosity Module | §6.3 Pathak 2017 | extrinsic + intrinsic |
| `noisy_net` | NoisyNets | §8 Fortunato 2018 | extrinsic |
| `bootstrapped_dqn` | Bootstrapped DQN (K heads) | §8 Osband 2016 | extrinsic |

Not included (need different agent/environment paradigms and out of scope here):
full Bayes-adaptive methods (BAMDP/BEETLE/PSRL, which are model-based and tabular),
meta-RL (MAML, MAESN, which require task distributions), policy-gradient exploration
methods (Gaussian/parameter-space noise, which need a PG agent), and
continuous-control-specific methods (MuJoCo, OU noise, PolyRL).

## Correctness notes on the originals

Three strategies existed in the previous `pyexplore` version. Issues found and
fixed in this rewrite:

- **ε-greedy**: matched eq. (14). The exponential decay schedule is a common
  practical variant; kept.
- **Boltzmann**: matched eq. (20); kept unchanged.
- **Count-based**: three problems:
  1. The bonus was added to Q-values at action-selection time. The survey
     (eq. 24) adds it to the reward, not to Q. Selection here is now greedy on
     Q, and the bonus is routed through the Trainer into the replay reward.
  2. `update_count` fired *before* `get_bonus`, so the first visit already got
     `β/√1` (partially self-cancelling). Now `observe` updates counts after
     the transition is emitted, giving genuinely novel states their full bonus.
  3. In the old `optimize_model`, `additional_reward` was a scalar broadcast
     across the whole batch using only the *current* state's count, which is
     incorrect. We
     now compute a per-transition bonus at collection time and store it in the
     replay buffer as part of the reward.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m rlexplore.cli --config examples/config.yaml --mode both
```

Switch strategies by editing `exploration.type` in the YAML. To run RND:

```yaml
exploration:
  type: rnd
  params:
    beta: 0.1
    hidden: 128
    out_size: 128
    lr: 0.0001
```

## Extending

```python
from rlexplore.core.registry import STRATEGIES
from rlexplore.exploration.base import ExplorationStrategy

@STRATEGIES.register("my_strategy")
class MyStrategy(ExplorationStrategy):
    def select_action(self, state, q_net, step):
        ...
```

Reference it from the config:

```yaml
exploration:
  type: my_strategy
  params: {...}
```

## License

See `LICENSE`.
