import numpy as np
import random
import timeit
import math

#The Split class is used to store the splitting criteria for nodes in the trees
class Split:
    def __init__(self, t, h, d):
        #Tracks whether a node is a split node or a leaf node
            #1: The node is a split node.
            #0: The node is a leaf node
        #split.splits[tree_id][node_id]
        self.splits = np.zeros([t, (2**h) - 1], dtype=int)
        #Tracks the attribute (or feature) used to split the data at each node
        self.attributes = np.empty([t, (2**h) - 1], dtype=int)
        #Tracks the threshold value used for splitting at each node
        self.values = np.empty([t, (2**h) - 1], dtype=float)

class Leaves:
    #W_MAX = max number of elements per leaf
    #each leaf has a capacity of storing up to W_MAX data points or indices.
    #this constraint ensures that the memory allocation for each leaf is bounded.
    #I think we can make it better by using a dynamic array for each leaf instead of a fixed size array.
    #or defining the max just as the window
    def __init__(self, t, H, W_MAX):
        # 2**H = max number of leaves, W_MAX = max number of elements per leaf
        # all set to -1
        # i guess counters are the counters of the amount of data
        self.counters = np.zeros([t, 2**H], dtype=int)
        self.table = np.zeros([t, 2**H, W_MAX], dtype=int) - 1

# anomaly score for insertion data structure, for the initial ones
#Leaves insertionDS
#I think n should be the number of instances in the leaf
def anomaly_score_ids(insertionDS, t, n):
    scores = np.zeros([n], dtype=float)
    # get table size, it's the same for all trees
    ids_size = insertionDS.table[0].shape[0]
    
    #loop the trees
    for i in range(t):
        # loop the leaves of each tree (not sure if is that what we do)
        for j in range(ids_size):
            # get leaf_size and calculate score 
            leaf_size = insertionDS.counters[i][j]
            if leaf_size != 0:
                score = leaf_size / n
                score = math.log(1/score)
                #and now loop through the data in the leaf
                for k in range(leaf_size):
                    data_pointer = insertionDS.table[i][j][k]
                    scores[data_pointer] += score
    return scores

# anomaly score for insertion data structure, for the other ones
#I think n should be the number of instances in the leaf
def anomaly_score_ids_incr(leaf_indexes, insertionDS, t, n):
    res = 0
    #loop the trees
    for i in range(t):
        # get leaf_size and calculate score
        j = leaf_indexes[i]
        leaf_size = insertionDS.counters[i][j]
        if leaf_size != 0:
            score = leaf_size / n
            score = math.log(1/score)
            res += score
    return res

def rhf_stream(data, t, h, N_init_pts):
    n, d = data.shape
    #this value should be set by the maximum number of elements in a leaf
    n = 20000
    W_MAX = n
    index_range = np.empty([t], dtype=int)
    leaf_indexes = np.empty([t], dtype=int)
    scores = np.empty([n], dtype=float)
    # create an empty forest of t trees each with n x 3 
    # 2 = (index in X, number of elems in leaf)
    indexes = np.empty([t, N_init_pts], dtype=int)
    # r-values for each Node and each tree
    r_values = np.empty([t, (2**h)-1, 2], dtype=float)
    # moments = trees * nodes * attributes * card({M1, M2, M3, M4, n})
    moments = np.zeros([t, (2**h)-1, d, 6], dtype=float)
    splits = Split(t, h, d)
    # create secondary data structure for insertion algorithm
    insertionDS = Leaves(t, h, W_MAX)
    kurtosis_arr = np.empty([d], dtype=float)
    new_indexes = np.empty([W_MAX], dtype=int)
    # intialize t trees and r values
    for i in range(t):
        for j in range((2**h) - 1):
            r0 = 0
            r1 = 0
            while r0 == 0:
                r0 = random.uniform(0, 1)
            while r1 == 0:
                r1 = random.uniform(0, 1)
            r_values[i][j][0] = r0
            r_values[i][j][1] = r1
        index_range = np.arange(0, N_init_pts, dtype=int)
        #for every every tree we have an array like [0,1,2,..,N_init_pts-1]
        #because of the window thing I would say
        indexes[i] = index_range
        rht(data, indexes[i], insertionDS, splits, moments[i], kurtosis_arr, 0, N_init_pts-1, 0, h, d, 0, i, r_values[i])
    
    print("Forest initialized...") 
    scores[:N_init_pts] = anomaly_score_ids(insertionDS, t, N_init_pts)
    print("Initial forest scored.")

    t0 = timeit.default_timer()
    for i in range(N_init_pts, n):
        process_data_instance(i, data, moments, splits, h, insertionDS, kurtosis_arr, new_indexes, leaf_indexes, scores, r_values, t, N_init_pts, indexes, W_MAX, d)
    
    t1 = timeit.default_timer()
    print("Total time for insertions=", t1 - t0)
          
    return scores

def process_data_instance(i, data, moments, splits, h, insertionDS, kurtosis_arr, new_indexes, leaf_indexes, scores, r_values, t, N_init_pts, indexes, W_MAX, d):
    """
    Process a single data instance (i) across all trees and compute the anomaly score.

    Parameters:
        i (int): Current data point index.
        data (ndarray): The dataset.
        moments (list): Statistical moments for the trees.
        splits (Split): Split structure for the trees.
        h (int): Height of the tree.
        insertionDS (Leaves): Leaves structure for the trees.
        kurtosis_arr (ndarray): Kurtosis values.
        new_indexes (list): Indexes for new data points.
        leaf_indexes (list): Leaf indexes for each tree.
        scores (ndarray): Anomaly scores.
        r_values (ndarray): Random thresholds or parameters.
        t (int): Number of trees.
        N_init_pts (int): Number of initial data points.
        indexes (list): Index ranges for trees.
        W_MAX (int): Maximum number of elements per leaf.
        d (int): Dimensionality of data.
    
    Returns:
        None: Updates `leaf_indexes`, `scores`, `insertionDS`, and `splits` in place.
    """
    # Process each tree
    for j in range(t):
        leaf_indexes[j] = insert(data, moments[j], splits, h, insertionDS, kurtosis_arr, new_indexes, i, j, r_values[j])
    
    # Compute the anomaly score for the current data point
    scores[i] = anomaly_score_ids_incr(leaf_indexes, insertionDS, t, N_init_pts + (i % N_init_pts) + 1)

    # Reset data structures periodically
    if (i + 1) % N_init_pts == 0:
        print("Forest number " + str(i // N_init_pts) + " scored.")
        print("Current instance number: ", i)
        insertionDS = Leaves(t, h, W_MAX)
        splits = Split(t, h, d)
        for j in range(t):
            index_range = np.arange(i - N_init_pts + 1, i + 1, dtype=int)
            indexes[j] = index_range
            rht(data, indexes[j], insertionDS, splits, moments[j], kurtosis_arr, 0, N_init_pts - 1, 0, h, d, 0, j, r_values[j])


def sort(data, indexes, start, end, a, a_val):
    i = start
    j = end
    while i < j:
        while data[indexes[i]][a] <= a_val and i < j:
            i += 1
        while data[indexes[j]][a] > a_val and j > i:
            j -= 1
        temp = indexes[i]
        indexes[i] = indexes[j]
        indexes[j] = temp
    return j

#first call 
#where 'i' are the different trees
#rht(data, indexes[i], insertionDS, splits, moments[i], kurtosis_arr, 0, N_init_pts-1, 0, h, d, 0, i, r_values[i])

#second call, for the other part of the data
#rht(data, indexes[j], insertionDS, splits, moments[j], kurtosis_arr, 0, N_init_pts-1, 0, h, d, 0, j, r_values[j]) 
def rht(data, indexes, insertionDS, split_info, moments, kurtosis_arr, start, end, nd, H, d, nodeID=0, t_id=0, r_values=None, insertion_pt=None):
    ls, a, split, insertion_leaf = -1, -1, -1, -1
    if end == start or nd >= H:
        # leaf size
        insertion_leaf = fill_leaf(indexes, insertionDS, nodeID, nd, H, start, end, 0, t_id)
    else:
        # calculate kurtosis
        ks = kurtosis_sum(data, indexes, moments[nodeID], kurtosis_arr, start, end)
        if ks == 0: # stop if all elems are the same
            split_info.splits[t_id][nodeID] = 0
            insertion_leaf = fill_leaf(indexes, insertionDS, nodeID, nd, H, start, end, 1, t_id)
        else: # split
            if r_values is not None:
                r_a = r_values[nodeID][0]
                r_a_val = r_values[nodeID][1]
            else:
                r_a = -1
                r_a_val = -1

            a, a_val, r_a, r_a_val = get_attribute(data, indexes, start, end, ks, kurtosis_arr, d, r_a, r_a_val)
            # sort indexes
            split = sort(data, indexes, start, end, a, a_val)
            split_info.splits[t_id][nodeID] = split
            split_info.attributes[t_id][nodeID] = a
            split_info.values[t_id][nodeID] = a_val

            # check on which side insertion point ended up 
            # return the result of that side 
            if insertion_pt is not None:
                if insertion_pt[a] <= a_val:
                    rht(data, indexes, insertionDS, split_info, moments, kurtosis_arr, split, end, nd + 1, H, d, 2*nodeID + 2, t_id, r_values, insertion_pt)
                    insertion_leaf = rht(data, indexes, insertionDS, split_info, moments, kurtosis_arr, start, split-1, nd+1, H, d, 2*nodeID + 1, t_id, r_values, insertion_pt)
                else:
                    rht(data, indexes, insertionDS, split_info, moments, kurtosis_arr, start, split-1, nd+1, H, d, 2*nodeID + 1, t_id, r_values, insertion_pt)
                    insertion_leaf = rht(data, indexes, insertionDS, split_info, moments, kurtosis_arr, split, end, nd+1, H, d, 2*nodeID + 2, t_id, r_values, insertion_pt)
            else:
                rht(data, indexes, insertionDS, split_info, moments, kurtosis_arr, start, split-1, nd+1, H, d, 2*nodeID + 1, t_id, r_values, insertion_pt)
                rht(data, indexes, insertionDS, split_info, moments, kurtosis_arr, split, end, nd+1, H, d, 2*nodeID + 2, t_id, r_values, insertion_pt)

    return insertion_leaf

def get_attribute(data, indexes, start, end, ks, kurt, d, r_a=-1, r_a_val=-1):
    a = -1
    cumul = 0
    if r_a == -1:
        r_a = random.uniform(0, ks)
        while r_a == 0:
            r_a = random.uniform(0, ks)
    else:
        r_a = r_a * ks

    for i in range(d):
        temp = kurt[i]
        kurt[i] += cumul
        if i == 0:
            if r_a <= kurt[i]:
                a = i
        else:
            if r_a > kurt[i-1] and r_a <= kurt[i]:
                a = i
        cumul += temp

    a_min = data[indexes[start]][a]
    a_max = a_min
    for i in range(start, end+1):
        temp = data[indexes[i]][a]
        if a_min > temp:
            a_min = temp
        elif a_max < temp:
            a_max = temp

    if r_a_val == -1:
        a_val = random.uniform(a_min, a_max)
        while a_val == a_min:
            a_val = random.uniform(a_min, a_max)
    else:
        a_val = r_a_val * (a_max - a_min) + a_min

    return a, a_val, r_a, r_a_val

# fill_leaf and kurtosis_sum would also need to be implemented in the Python version as needed.


def kurtosis_sum(data, indexes, moments, kurtosis_arr, start, end):
    d = data.shape[1]
    ks = 0
    for a in range(d): 
        kurtosis_arr[a] = incr_kurtosis(data, indexes, moments[a], start, end, a)
        kurtosis_arr[a] = math.log(kurtosis_arr[a] + 1)
        ks += kurtosis_arr[a]
    return ks

def incr_kurtosis(data, indexes, moments, start, end, a):
    n, mean, M2, M3, M4 = 0, 0, 0, 0, 0
    for i in range(start, end+1):
        x = data[indexes[i], a]
        n1 = n
        n += 1
        delta = x - mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n 
        term1 = delta * delta_n * n1
        mean += delta_n
        M4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3
        M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2
        M2 += term1

    moments[0] = mean
    moments[1] = M2
    moments[2] = M3
    moments[3] = M4
    moments[4] = n
    
    if M4 == 0: 
        return 0
    else:
        kurtosis = (n * M4) / (M2 * M2)
        return kurtosis

def fill_leaf(indexes, insertionDS, nodeID, nd, H, start, end, ls=0, t_id=0):
    if ls == 0:
        ls = end - start + 1

    if nodeID >= (2**H - 1):
        leaf_index = nodeID
    else:
        leaf_index = (2**(H - nd)) * (nodeID + 1) - 1
    
    leaf_index -= (2**H - 1)

    for i in range(start, end+1):  
        counter = insertionDS.counters[t_id][leaf_index]
        insertionDS.table[t_id][leaf_index][counter] = indexes[i]
        insertionDS.counters[t_id][leaf_index] += 1

    return leaf_index

def insert(data, moments, split_info, H, insertionDS, new_kurtosis_vals, new_indexes, i, t_id, r_values):
    nodeID = 0
    a = -1
    split_a = None
    leaf_index = None
    counter = None
    d = data.shape[1]
    nd = 0
    ks = 0
    split_a_val = None
    old_kurtosis_sum = None
    new_kurtosis_sum = None
    r = None
    keep = None
    cumul = None
    old_kurtosis_vals = None
    moments_calc = None
    M2 = None
    M4 = None
    split_attributes = split_info.attributes[t_id]
    split_vals = split_info.values[t_id]
    split_splits = split_info.splits[t_id]

    while nodeID < (2**H) - 1 and split_splits[nodeID] != 0:
        split_a = split_attributes[nodeID]
        split_a_val = split_vals[nodeID]
        
        new_kurtosis_sum = kurtosis_sum_ids(data, moments[nodeID], new_kurtosis_vals, i) 
        
        r = r_values[nodeID][0] * new_kurtosis_sum
        cumul = 0
        a = -1
        for j in range(d):
            keep = new_kurtosis_vals[j]
            new_kurtosis_vals[j] += cumul 
            if j == 0:
                if r <= new_kurtosis_vals[j]:
                    a = j
            else:
                if r > new_kurtosis_vals[j-1] and r <= new_kurtosis_vals[j]:
                    a = j
            cumul += keep
        
        if (split_a != a):
            start_index = (2**(H - nd) * (nodeID + 1)) - 1 - (2**H - 1)
            end_index = (2**(H - nd)) - 1 + start_index
            total_counter = 0
            counters = insertionDS.counters[t_id]
            table = insertionDS.table[t_id]
            for k in range(start_index, end_index+1):
                counter = counters[k]
                for l in range(counter):
                    new_indexes[total_counter + l] = table[k][l]
                total_counter += counter
                counters[k] = 0
            new_indexes[total_counter] = i
            leaf_index = rht(data, new_indexes, insertionDS, split_info, moments, new_kurtosis_vals, start=0, end=total_counter, 
                             nd=nd, H=H, d=d, nodeID=nodeID, t_id=t_id, r_values=r_values, insertion_pt=data[i])
            return leaf_index
        
        if data[i, split_a] <= split_a_val:
            nodeID = nodeID * 2 + 1
        else:
            nodeID = nodeID * 2 + 2

        nd += 1
    
    if nodeID >= (2**H - 1):
        leaf_index = nodeID
        leaf_index -= (2**H - 1)
        counter = insertionDS.counters[t_id][leaf_index]
        insertionDS.table[t_id][leaf_index][counter] = i
        insertionDS.counters[t_id][leaf_index] += 1
        return leaf_index
    else:
        leaf_index = (2**(H - nd)) * (nodeID + 1) - 1
        leaf_index -= (2**H - 1)
        counter = insertionDS.counters[t_id][leaf_index]
        insertionDS.table[t_id][leaf_index][counter] = i
        temp = insertionDS.table[t_id, leaf_index, :counter+1].copy()
        insertionDS.counters[t_id][leaf_index] = 0
        new_indexes = temp
        leaf_index = rht(data, new_indexes, insertionDS, split_info, moments, new_kurtosis_vals,
                         start=0, end=counter, nd=nd, H=H, d=d, nodeID=nodeID, t_id=t_id,  r_values=r_values, insertion_pt=data[i])
        return leaf_index

def kurtosis_sum_ids(data, moments, kurtosis_arr, i):
    d = data.shape[1]
    kurtosis_sum = 0
    for a in range(d): 
        mean = moments[a][0]
        M2 = moments[a][1]
        M3 = moments[a][2]
        M4 = moments[a][3]
        n = moments[a][4]
        
        x = data[i, a]
        n1 = n
        n += 1
        delta = x - mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n 
        term1 = delta * delta_n * n1
        mean += delta_n
        M4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3
        M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2
        M2 += term1
        
        if M4 == 0:
            kurtosis = 0
        else:
            kurtosis = (n * M4) / (M2 * M2)
            kurtosis = math.log(kurtosis + 1)
            kurtosis_sum += kurtosis
        
        moments[a][0] = mean
        moments[a][1] = M2
        moments[a][2] = M3
        moments[a][3] = M4
        moments[a][4] = n
        kurtosis_arr[a] = kurtosis
    
    return kurtosis_sum
