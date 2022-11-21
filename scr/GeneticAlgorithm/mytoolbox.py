
from itertools import chain 
import numpy as np
import random

def return_X_y(data, indexOfLabel, fast = False):
    '''
    Return the features and label
    The option of fast returns righ-most column as y
    '''
    if fast == True:
        return data[:,:indexOfLabel], data[:,indexOfLabel]
    selection_X = [1] * (len(data[0]))
    selection_X[indexOfLabel] = 0
    selection_y = [0] * (len(data[0]))
    selection_y[indexOfLabel] = 1
    return selectFeatures(data, selection_X), list(chain.from_iterable(selectFeatures(data, selection_y))) 
    
def selectFeatures(data, arr_seletion):
    '''
    Return subset of features
    '''
    columnIndex = list(getIndexPositions(arr_seletion, 1))
    return data[:,columnIndex]

def selectFeatures_withFeatureSel(data, arr_seletion):
    '''
    Return subset of features
    '''
    columnIndex = list(getIndexPositions(arr_seletion, 1))
    return data[:,columnIndex], columnIndex

import csv
def readData(add):
    '''
    Read data as np array.
    '''
    with open(add) as file:
        Reader = csv.reader(file)
        data = list(list(float(elem) for elem in row) for row in Reader)
        data = np.asarray(data)
    return data



#########

def getIndexPositions(listOfElements, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    indexPosList = []
    indexPos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            indexPos = listOfElements.index(element, indexPos)
            # Add the index position in list
            indexPosList.append(indexPos)
            indexPos += 1
        except ValueError as e:
            break
 
    return indexPosList

def shrink(individual, size):
    nz = [i for i in range(len(individual)) if individual[i]==1] # non-zero selection
    select = np.random.choice(nz, size, replace=False)
    for i in range(len(individual)):
        individual[i] = 0
    for i in select:
        individual[i] = 1
    return individual

def Select(individual, selection, selected_features):
    '''
    individual
    selection: the index of selected feature in the subset
    selected_features: the selected features
    '''
    #nz = [i for i in range(len(individual)) if individual[i]==1] # non-zero selection
    select = selection
    for i in range(len(individual)):
        individual[i] = 0
    for i in select:
        individual[selected_features[i]] = 1
    return individual

def cross_over(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::
    
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5,6,7,8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)

    #ind1_nz = [i for i in range(len(ind1)) if ind1[i]==1]
    ind1_nz = list(np.where(ind1 == 1)[0])
    #ind2_nz = [i for i in range(len(ind2)) if ind2[i]==1]
    ind2_nz = list(np.where(ind2 == 1)[0])

    ind1_cxpoint = random.randint(0, len(ind1_nz)-1)
    ind2_cxpoint = random.randint(0, len(ind2_nz)-1)

    random.shuffle(ind1_nz)
    random.shuffle(ind2_nz)

    ind1_nz[ind1_cxpoint:], ind2_nz[ind2_cxpoint:] = ind2_nz[ind2_cxpoint:].copy(), ind1_nz[ind1_cxpoint:].copy()

    ind1.fill(0)
    ind2.fill(0)
    for i in ind1_nz:
        ind1[i] = 1
    for i in ind2_nz:
        ind2[i] = 1
        
    return ind1, ind2