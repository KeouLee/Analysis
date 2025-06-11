import numpy as np
import copy
from typing import Sequence
from numpy.linalg import norm

def uncat(idx):
    """
    notes
    ----------
    baed on itertools.cpmbinations but only iterate the first element in Sequence.

    arguments
    ----------
    idx :: (1,5,0,3)

    return
    ----------
    [(1,5), (1,0), (1,3),]
    """
    return [(idx[0], idx[i]) for i in range(1,len(idx))]

def triangle_build(p1,p2):
    """
    notes
    ----------
    based on two two-lengthed Sequence to build a 
    (shared_index, first_argument_diff_index, second_argument_diff_index) tuple.

    arguments
    ----------
    p1, p2 :: two Sequence to be inspected

    return
    ----------
    (shared_index, first_argument_diff_index, second_argument_diff_index) tuple.
    """
    idx1, idx2 = eq_idx(p1,p2)[0]
    l1 = np.setdiff1d(p1, [p1[idx1]])[0]
    l2 = np.setdiff1d(p2, [p2[idx2]])[0]
    return p1[idx1], l1, l2

def eq_idx(p1, p2):
    """
    notes
    ----------
    general method for two 1d arbitrary-length index list

    arguments
    -----------
    p1 = [1,2,5]  p2 = [2,3,10,5]

    returns
    -----------
    [(1,0,), (2,3),]
    """
    count1 = 0 
    lt = list()

    for n1 in p1:
        count2 = 0
        for n2 in p2:
            if (n1 == n2):
                lt.append((count1, count2))
                continue
            count2 += 1
        count1 += 1
    return lt

def get_com(coords,mass_lt,unwrap=True):
    """
    notes
    -------------
    make sure that use this function only for wrapped coordinates with respect to a cluster.
    
    arguments
    -------------
    coords(AtomNum, 3) numpy.ndarray
    mass_lt(AtomNum) numpy.ndarray

    return
    -------------
    center of mass(3,) numpy array
    """
    if not (len(mass_lt.shape) == 1):
        raise ValueError("please supply a 1d numpy array for masses")

    if coords.shape[0] != mass_lt.shape[0]:
        raise ValueError(f"cluster AtomNum {coords.shape[0]} incompatiable with mass_lt length {len(mass_lt)}, plz check")

    masses = mass_lt.reshape((len(mass_lt), 1)) # for broadcast
    return np.sum(coords * masses , axis=0) / np.sum(mass_lt)

def exclude(p1: Sequence, p: Sequence):
    """return what's in p1 but not p
    p2 = p - p1
    """
    pp = list(p)
    return [i for i in pp if i not in p1]

def dist_map(fin):
    """mapping/transformation of a function from Space(FrameNum, AtomNum, 3) to Space(FrameNum, AtomNum)."""
    a, b = fin.shape[:2]
    fout = np.zeros((a,b))
    for i in range(a):
        fout[i] = np.array([norm(fin[i][j]) for j in range(b)])
    return fout

def find_node(lt_2d):
    new_ltt = list()
    d = 0
    for s in lt_2d:
        d += len(s)
        new_ltt.append(d)
    return new_ltt

def cat_to_mol_real(connectivity):
    # connectivity == initial connections between atoms
    lt = connectivity
    new_lt = connectivity
    while True:
        new_lt = cat_to_mol(new_lt)
        if new_lt == lt:
            return new_lt
        lt = new_lt

def cat_to_mol(connect):
    mol_lt_low = cat_to_mol_low(connect)
    return [list(set(mol_low)) for mol_low in mol_lt_low]

def cat_to_mol_low(connect):
    lt = copy.deepcopy(connect)
    i=0
    while True:
        while True:
            later=lt[i+1:]
            if (len(later) == 0):
                return lt
            idx_lt = []
            for j, t in enumerate(later):
                if has_same(t,lt[i]):
                    lt[i] += t
                    idx_lt.append(i+1+j)
            if (len(idx_lt) == 0):
                i+=1
                break
            for idx in reversed(idx_lt):
                lt.pop(idx)

def has_same(lt1, lt2):
    for ele1 in lt1:
        if ele1 in lt2:
            return True
    return False

def same_cat_pair(lt):
    """input: ['a','a','b','c','c','a'] 
       return: [('a',3), ('b',1), ('c',2)]
       or return dict {'a': 3, 'b': 1, 'c': 2}
   """
    #new_lt = []
    new_dict = {}
    for ele in lt:
        if ele in new_dict:
            new_dict[ele] += 1
        else:
            new_dict[ele] = 1
    return new_dict

def mean_freq(freq, signal):
    #return np.sum(signal*freq), np.sum(signal)
    #print(np.sum(signal*freq), np.sum(signal))
    return np.sum(signal*freq) / np.sum(signal)

def test():
    lt = [[0,1,2],[2,3],[4,5],[5,6],[10,11],[11,6]]
    res = cat_to_mol(lt)
    print(lt)
    print(res)
    #print(find_node([[2, 4], [2, 6], [2, 7,4], [2, 8], ]))
    #lt = [[2, 4], [2, 6], [7, 8],[2, 7],  [3, 5], [3, 9], [3, 10], [3, 11]]
    #res = cat_to_mol(lt)
    #print(lt)
    #print(res)

    #finale = cat_to_mol(res)
    #print(finale)

    #ltt = [[1,3],[2,4]]
    #res = cat_to_mol(ltt)
    #print(res)

    #l = [1,2,3,4,9,5]
    #ll = [2,5,1]
    #eq_idx(l, ll)

    #res = triangle_build((2,1), (2,3))
    #print(res)

    #res = uncat([0,1,2,3])
    #print(res)

    #np.random.seed(5)
    #l = np.random.randint(1,10,(2,5,3))
    #res = dist_map(l).reshape(2,5,1)
    #res=l/res
    #for rr in res:
    #    for r in rr:
    #        print(norm(r))
    #print(l)
    #l=np.delete(l,[0,1],axis=1)
    #print(l)
    #print(isinstance(l, Sequence|np.ndarray))


if __name__ == "__main__":
    test()
