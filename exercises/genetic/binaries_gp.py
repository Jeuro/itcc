import operator
import math
import random
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

target = [0, 1] * 200

# primitive set parameter is the index of the list
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.not_, 1)
pset.addPrimitive(operator.and_, 2)
pset.addPrimitive(operator.or_, 2)
pset.addPrimitive(operator.xor, 2)
pset.addEphemeralConstant("leaf1", lambda: random.randint(0,1))
pset.addEphemeralConstant("leaf2", lambda: random.randint(0,1))
pset.renameArguments(ARG0="i")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evaluate(individual):
    fit = 0
    # Transform the tree expression into a callable function
    f = toolbox.compile(expr=individual)
    l = [f(i) for i in range(len(target))]

    for j in range(len(l)):
        if l[j] == target[j]:
            fit += 1

    return fit,

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

#toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
#toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    p = 1
    g = 1
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

    f = toolbox.compile(expr=hof[0])
    l = [f(i) for i in range(len(target))]
    print(target, l)
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