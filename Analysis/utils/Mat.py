import numpy as np

def is_symmetric(mat, rtol=1e-05, atol=1e-08):
    return np.allclose(mat, mat.T, rtol=rtol, atol=atol)

def flip(mat, flip_downward=True, flip_upward=False):
    """flip to get symmetric."""
    assert not (flip_downward and flip_upward)
    assert mat.shape[0] == mat.shape[1]

    idx=mat.shape[0]
    if flip_downward:
        for i in range(idx):
            for j in range(idx):
                if i>j:
                    mat[i,j] = mat[j,i]

    if flip_upward:
        for i in range(idx):
            for j in range(idx):
                if i<j:
                    mat[j,i] = mat[i,j]

    assert is_symmetric(mat)
    return mat

def test():
    m = np.random.randint(1,10, size=(10,10))
    print(m)
    print("flipping...")
    m = flip(m) 
    print(m)

if __name__ == "__main__":
    test()
