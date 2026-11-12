"""CLI entrypoint for training the ATCS D3QN reinforcement learning agent.

Delegates to `rl_model.train.main()`.
"""
import sys
from rl_model.train import main, train

__all__ = ["train", "main"]

if __name__ == '__main__':
    sys.exit(main())