#!/bin/python
# -*- coding: utf-8 -*-


def evolve_func(ser_algo_pop):
    # The evolve function that is actually run from the separate processes in the desert island
    import dill
    algo, pop = dill.loads(ser_algo_pop)
    new_pop = algo.evolve(pop)
    return dill.dumps((algo, new_pop))


class Desert_island(object):

    def __init__(self, pool_size=None):

        if pool_size == 1:

            desert_island.pool = None
            desert_island.pool_size = 1

            return

        import pathos

        if pool_size is None:
            pool_size = pathos.multiprocessing.cpu_count()

        desert_island.pool_size = pool_size
        # desert_island.pool = pathos.multiprocessing.Pool(pool_size)
        desert_island.pool = pathos.pools.ProcessPool(pool_size)
        desert_island.pool_size = pool_size

        return

    def run_evolve(self, algo, pop):

        print('called2')
        if desert_island.pool is None:

            new_pop = algo.evolve(pop)
            return algo, new_pop

        else:

            import dill

            ser_algo_pop = dill.dumps((algo, pop))
            res = desert_island.pool.pipe(evolve_func, (ser_algo_pop,))
            # res = desert_island.pool.pipe(evolve_func, (ser_algo_pop,))
            # res = desert_island.pool.apply_async(evolve_func, (ser_algo_pop,))

            print('ret')
            ret = dill.loads(res.get())
            print('got ret')
            return ret
            # return dill.loads(res.get())

    def get_name(self):
        return "Desert island (1987)"
