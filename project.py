from gensim.models import word2vec
import numpy

def mvme(t1,t2):
    model = word2vec.Word2Vec.load('modelWord2Vec_joy');
    t1Tokens = t1[0];
    t2Tokens = t2[0];
    n = len(t1Tokens);
    m = len(t2Tokens);
    mat = [[0.0 for x in range(n)] for x in range(m)];
    for i in range(n):
        u = t1Tokens[i];
        for j in range(m):
            v = t2Tokens[j];
            try:
                mat[j][i] = model.similarity(u,v);
            except KeyError:
                mat[j][i] = 0;
            except AttributeError:
                print('Error')
                mat[j][i] = 0;
    sim = 0.0;
    mat = numpy.asarray(mat);
    count = 0
    while mat.size:
        maxValue = -100.0;
        for x in range(n-count):
            for y in range(m-count):
                if mat[y][x] > maxValue:
                    row = y;
                    col = x;
                    maxValue = mat[y][x];
        sim += mat[row][col];
        mat = numpy.delete(mat, (row), axis=0);
        mat = numpy.delete(mat, (col), axis=1);
        count += 1;
    return sim;
def isMatEmpty(mat):
    print('Empty');
    for x in range(len(mat)):
        for y in range(len(mat[x])):
            if mat[x][y] == 0.0:
                continue;
            else:
                return False;
    return True;
