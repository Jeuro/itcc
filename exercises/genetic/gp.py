import operator
import math
import random
import numpy
import time
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

target = time.time()

def safeDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 0)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(operator.abs, 1)
pset.addEphemeralConstant("leaf1", lambda: random.randint(-100,100))
pset.addEphemeralConstant("leaf2", lambda: random.randint(-100,100))
pset.addEphemeralConstant("leaf3", lambda: random.randint(-100,100))
pset.addEphemeralConstant("leaf4", lambda: random.randint(-100,100))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evaluate(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    error = abs(func - target)
    return error,

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    p = 200
    g = 200
    c = 0.6
    m = 0.2
    pop = toolbox.population(n=p)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, c, m, g, stats=mstats,
                                   halloffame=hof, verbose=True)

    print(target, toolbox.compile(expr=hof[0]))
    f = evaluate(hof[0])
    print(f)
    # print log
    #return pop, log, hof

    nodes, edges, labels = gp.graph(hof[0])

    import pygraphviz as pgv

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.pdf")


if __name__ == "__main__":
    main()