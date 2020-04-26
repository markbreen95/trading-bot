from collections import namedtuple

Crossover = namedtuple("CrossoverStrategyConfig",
                       [
                           "short",
                           "long",
                           "buy_cutoff",
                           "sell_cutoff",
                        ])