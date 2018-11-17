import gzip
import csv
import random
import numpy as np
import pandas as pd
import operator
import math
import time
import scipy
import functools as ft
from multiprocessing import Pool
import multiprocessing as mp
import itertools as it
import pickle

from itertools import combinations
from Bio import Align
from Bio.SubsMat import MatrixInfo
from numpy import genfromtxt

def coord_helper(name):
    global n
    global invTripDict

    l,r = name.split('|')
    l = invTripDict[l]
    r = invTripDict[r]
    return l*n+r

def map_helper_combos(eaches, trim, q1, q2):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.substitution_matrix = MatrixInfo.blosum62

    al_helper = lambda val: (val[0]+'|'+val[1], aligner.align(val[0].lower(),val[1].lower()).score)

    first, second = zip(*map(al_helper, eaches))

    first = map(coord_helper, first)

    q1.put(first)
    q2.put(second)

    return

def getDistMatrix_andDict_notNormed(myMotifs,combo_count,trim='No'):
    tripDict = dict( zip(range(len(myMotifs)),myMotifs) )
    invTripDict = {v:k for k,v in tripDict.items()}
    n = len(myMotifs)

    s = time.time()

    combos = combinations(myMotifs, 2)

    q1 = mp.JoinableQueue()
    q2 = mp.JoinableQueue()

    helper = ft.partial(map_helper_combos, trim=trim, q1=q1, q2=q2)

    n_procs = 8 # use this many processors
    div_factor = 64 # divide into this many batches per processor

    local_names = []
    local_combos = []

    # launch processes
    for k in range(div_factor):
        proc_list = []

        for i in range(n_procs):
            sl = list(it.islice(combos, int(combo_count//n_procs//div_factor+1)))
            p = mp.Process(target=helper, kwargs = {'eaches':sl})
            proc_list.append(p)
            p.start()

        # get results from q
        name_list = []
        combo_list = []
        for i in range(n_procs):
            name_list.append(q1.get()) # get result of first queue
            q1.task_done()
            combo_list.append(q2.get()) # get result of second queue
            q2.task_done()

            p = proc_list.pop(0) # pop a process from queue
            p.join(timeout=1.0) # terminate process to prevent overflow

        local_names = it.chain.from_iterable(name_list) # add chained local iter object to list
        local_combos = it.chain.from_iterable(combo_list) # add chained local iter object to list

        with open('pickle_files/mat_' +str(k)+'.pickle', 'wb') as f:
            pickle.dump(local_combos, f, protocol=-1)
        with open('pickle_files/names_'+str(k)+'.pickle', 'wb') as f:
            pickle.dump(local_names, f, protocol=-1)

        print('Finished pickling batch' + str(k) + '/' + str(div_factor) + '. Total time: ' + str(time.time()-s))

    # join + close q
    q1.join()
    q1.close()
    q2.join()
    q2.close()

    return True

myMotifs = genfromtxt('motifsSimple.csv',delimiter=',',dtype=object)
#myMotifs, motif_names = list(zip(*enumerate(myMotifs)))
myMotifs = list(map(lambda x: x.decode('utf-8'), myMotifs)) # convert to strings
combo_count = int(len(myMotifs)*(len(myMotifs)-1)/2)
#combo_count = int(4000*3999/2)

s = time.time()
getDistMatrix_andDict_notNormed(myMotifs, combo_count)
print(time.time()-s)
