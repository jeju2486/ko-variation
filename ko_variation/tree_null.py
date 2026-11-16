"""Tree working covariance algebra for intercept-only PQL nulls.

Float64 contrasts/solves use the same normalized rooted-tree covariance as K.
No fastmath, approximate sigmoid or per-response eigenvectors are used.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
from . import kinship
try:
    import numba
except ImportError:
    numba = None

def contrasts(left, right, tip, length, weights, tau, data):
    """Independent Gaussian contrasts, including the fixed-root contribution.

    Tip observation variance is 1/W. Each branch adds tau*length. Combining two
    subtrees gives one independent contrast and a precision-weighted mean.
    Zero-length branches and multifurcations (binary zero-edge expansion) work
    without jitter. No covariance inverse or eigenvectors are constructed.
    """
    n, width = data.shape
    means = np.zeros((len(tip), width))
    variance = np.empty(len(tip))
    white = np.empty((n, width))
    logdet, row = 0.0, 0
    for v in range(len(tip)):
        if tip[v] >= 0:
            variance[v] = 1.0/weights[tip[v]] + tau*length[v]
            means[v] = data[tip[v]]
        else:
            a, b = left[v], right[v]
            total = variance[a]+variance[b]
            fraction = variance[b]/total
            variance[v] = variance[a]*fraction + tau*length[v]
            for j in range(width):
                white[row,j] = (means[a,j]-means[b,j])/math.sqrt(total)
                means[v,j] = fraction*means[a,j]+(variance[a]/total)*means[b,j]
            logdet += math.log(total)
            row += 1
    for j in range(width):
        white[n-1,j] = means[-1,j]/math.sqrt(variance[-1])
    logdet += math.log(variance[-1])
    return white, logdet, means, variance


def tree_solve(left, right, tip, length, weights, tau, data):
    """Apply S^-1 by a forward contrast pass and its reverse adjoint."""
    _, _, means, variance = contrasts(left,right,tip,length,weights,tau,data)
    adjoint = np.zeros_like(means)
    adjoint[-1] = means[-1]/variance[-1]
    out = np.empty_like(data)
    for v in range(len(tip)-1,-1,-1):
        if tip[v] >= 0:
            out[tip[v]] = adjoint[v]
        else:
            a, b = left[v], right[v]
            total = variance[a]+variance[b]
            for j in range(data.shape[1]):
                contrast = (means[a,j]-means[b,j])/total
                adjoint[a,j] = variance[b]/total*adjoint[v,j]+contrast
                adjoint[b,j] = variance[a]/total*adjoint[v,j]-contrast
    return out


def tree_multiply(left, right, tip, length, data):
    """Apply the normalized shared-branch covariance K in two tree passes."""
    totals = np.zeros((len(tip),data.shape[1]))
    carry = np.zeros_like(totals)
    out = np.empty_like(data)
    for v in range(len(tip)):
        totals[v] = data[tip[v]] if tip[v]>=0 else totals[left[v]]+totals[right[v]]
    carry[-1] = length[-1]*totals[-1]
    for v in range(len(tip)-1,-1,-1):
        if tip[v]>=0:
            out[tip[v]] = carry[v]
        else:
            a, b = left[v],right[v]
            carry[a] = carry[v]+length[a]*totals[a]
            carry[b] = carry[v]+length[b]*totals[b]
    return out


if numba is not None:
    # No fastmath, parallel reductions, reduced precision or approximate sigmoid.
    numba.extending.register_jitable(contrasts)
    compiled_contrasts = numba.njit(cache=True, nogil=True, fastmath=False)(contrasts)
    compiled_solve = numba.njit(cache=True, nogil=True, fastmath=False)(tree_solve)
    compiled_multiply = numba.njit(cache=True, nogil=True, fastmath=False)(tree_multiply)
else:
    compiled_contrasts = compiled_solve = compiled_multiply = None


@dataclass
class TreeAlgebra:
    left: np.ndarray
    right: np.ndarray
    tip: np.ndarray
    length: np.ndarray
    engine: str = 'numba'

    @classmethod
    def load(cls, path, engine='numba'):
        with np.load(path, allow_pickle=False) as arrays:
            return cls(*(arrays[k] for k in ('left','right','tip','length')), engine)

    @classmethod
    def from_newick(cls, path, samples, engine='numba'):
        root = kinship._NewickParser(Path(path).read_text().strip()).parse()
        kinship._assign_depths(root, 'error')
        lookup = {name:i for i,name in enumerate(samples)}
        leaves = {v.name:v for v in kinship._walk_preorder(root) if v.is_leaf}
        if len(lookup)!=len(samples) or set(samples)-set(leaves):
            raise ValueError('Tree must contain every retained sample exactly once')
        scale = float(np.mean([leaves[name].depth for name in samples]))
        if not np.isfinite(scale) or scale<=0:
            raise ValueError('Tree has invalid covariance normalization')
        left, right, tip, length = [],[],[],[]
        def append(a,b,s,d):
            left.append(a);right.append(b);tip.append(s);length.append(d)
            return len(tip)-1
        def visit(node):
            edge = 0.0 if node.parent is None else float(node.length)
            if node.is_leaf:
                return append(-1,-1,lookup[node.name],edge) if node.name in lookup else None
            children = [v for child in node.children if (v:=visit(child)) is not None]
            if not children: return None
            current = children[0]
            for child in children[1:]: current = append(current,child,-1,0.0)
            length[current] += edge  # Contract unary nodes without moving the root.
            return current
        visit(root)
        return cls(np.array(left,dtype='i8'),np.array(right,dtype='i8'),
                   np.array(tip,dtype='i8'),np.array(length,dtype='f8')/scale,engine)

    def apply(self, operation, *args):
        function = {'white':contrasts,'solve':tree_solve,'multiply':tree_multiply}[operation]
        if self.engine=='numba':
            function = {'white':compiled_contrasts,'solve':compiled_solve,'multiply':compiled_multiply}[operation]
            if function is None: raise ValueError('Install numba, or explicitly use --engine python')
        return function(self.left,self.right,self.tip,self.length,*args)


class TreeWorking:
    def __init__(self, tree, weights, C, z):
        self.tree, self.weights, self.C, self.z = tree,weights,C,z
        if C.shape[1]!=1 or not np.all(C==1):
            raise ValueError('Tree nulls require an intercept-only design')
        self.data = np.ascontiguousarray(np.column_stack((C,z)))

    def point(self,tau):
        white,logdet,_,_ = self.tree.apply('white',self.weights,tau,self.data)
        c,z = white[:,0],white[:,1]
        information = float(c@c)
        if not np.isfinite(information) or information<=0:
            raise np.linalg.LinAlgError('Fixed-effect information is singular')
        beta = float(c@z)/information
        residual = z-beta*c
        return float(logdet+math.log(information)+residual@residual),beta,1.0/information

    def objective(self,tau):
        return self.point(tau)[0]

    def materialize(self,tau):
        objective,beta,inverse = self.point(tau)
        rhs = np.ascontiguousarray(np.column_stack((self.C,self.z-beta)))
        solved = self.tree.apply('solve',self.weights,tau,rhs)
        effect = tau*self.tree.apply('multiply',np.ascontiguousarray(solved[:,-1:]))[:,0]
        return objective,np.array([beta]),effect,solved[:,:1],np.array([[inverse]])


@dataclass
class TreeWeights:
    weights: np.ndarray
    tree: TreeAlgebra

    def score_variance(self, residual, tau):
        white, _, _, _ = self.tree.apply('white', self.weights, tau, np.ascontiguousarray(residual))
        return np.sum(white * white, axis=0)
