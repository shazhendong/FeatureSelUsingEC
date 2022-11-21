#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import numpy
import mytoolbox
import sys, getopt
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from pandas_plink import read_plink
# from sklearn.impute import SimpleImputer


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def eval(individual, size, dataset_X, dataset_y):
    sizeLimit = size
    if sum(list(individual)) >= sizeLimit:
        individual = mytoolbox.shrink(individual, sizeLimit)
    if sum(list(individual)) == 0:
        i = random.randint(0, len(list(individual))-1)
        individual[i] = 1
    X,y = dataset_X, dataset_y
    if sum(list(individual))==0:
        return (0,)
    X = mytoolbox.selectFeatures(X, list(individual))
    
    foldsNum = 5
    kf = StratifiedKFold(n_splits=foldsNum, shuffle=True)

    arr_auc = []
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = LogisticRegression(solver="lbfgs", random_state=0,max_iter=10000).fit(X_train, y_train)
        arr_auc.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    return numpy.mean(arr_auc),



def main(ts,mp,cp,ng,ps,sl,data_X,data_y):
    '''
    ts: tournsize
    mp: mutation prob
    cp: crossover prob
    ng: number of generation
    ps: population size
    sl: size limit
    data: dataset
    '''
    #random.seed(64)



    pop = toolbox.population(n=ps)
    toolbox.register("evaluate", eval, size=sl, dataset_X=data_X, dataset_y=data_y)
    toolbox.register("mate", mytoolbox.cross_over)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=ts)
    
    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=numpy.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    #algorithms.eaMuPlusLambda(pop, toolbox, mu = p_mu, lambda_ = p_ld, cxpb = cp, mutpb = mp, ngen=ng, stats=stats, halloffame=hof, verbose=True)
    algorithms.eaSimple(pop, toolbox, cxpb=cp, mutpb=mp, ngen=ng, stats=stats, halloffame=hof)
    best_ind = tools.selBest(pop, 1)[0]
    # print(list(best_ind))
    print([i for i, e in enumerate(list(best_ind)) if e == 1])

    return pop, stats, hof

if __name__ == "__main__":
    argv = sys.argv[1:]
    ngen = 50
    pop_size = 100
    cxprob = 0.6
    mutprob = 0.2
    touSize = 6
    num_repeat = 10
    size = 6
    file = None
    plsr_n_comp = 6
    mu = 50
    ld = 20
    bfile = None
    

    try:
        opts, args = getopt.getopt(argv,"hg:p:c:m:t:r:s:f:b:",)
    except getopt.GetoptError:
        print('Error!')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-g': # -g represents number of generation
            ngen = int(arg)
        elif opt == '-p': # -p represents population size
            pop_size = int(arg)
        elif opt == '-c': # -c represents crossover probability
            cxprob = float(arg)
        elif opt == '-m': # -m represents mutation probability
            mutprob = float(arg)
        elif opt == '-t': # -t represents tournament size
            touSize = int(arg)
        elif opt == '-r': # -r represents number of repeats
            num_repeat = int(arg)
        elif opt == '-s': # -s represents the maximun number of features
            size = int(arg)
        elif opt == '-f': # -f represents file name
            file = arg
        elif opt == '-b': # -b represents bfile
            bfile = arg
    
    # read file 
    dataset_X = None
    dataset_y = None
    if file != None:
        dataset = pd.read_csv(file).to_numpy()
        dataset_X,dataset_y = mytoolbox.return_X_y(dataset, len(dataset[0])-1, fast=True)
    elif bfile != None:
        (bim, fam, bed) = read_plink(bfile,verbose=False)
        dataset_X = bed.compute().T
        # imputation
        # imputer = SimpleImputer(missing_values=numpy.nan, strategy='most_frequent')
        # imputer = imputer.fit(dataset_X)
        # dataset_X = imputer.transform(dataset_X)
        dataset_y = fam['trait']
    else:
        # default
        dataset = pd.read_csv('ML_Ready_Data.csv').to_numpy()
        dataset_X,dataset_y = mytoolbox.return_X_y(dataset, len(dataset[0])-1, fast=True)

    
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(dataset_X[0]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    for i in range(num_repeat):
        main(ts=touSize, mp=mutprob, cp=cxprob, ng=ngen, ps=pop_size, sl=size, data_X = dataset_X, data_y = dataset_y)
