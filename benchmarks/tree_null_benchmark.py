#!/usr/bin/env python3
"""Paired dense versus tree PQL null fitting; no production files are changed.

Tree contrasts implement S = diag(1/W) + tau*K without dense decompositions.
The production outer PQL loop, sigmoid, convergence and tau search are retained.
Each completed response/repeat is checkpointed. See docs/tree_null_benchmark.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import time
from unittest.mock import patch

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import scipy
from scipy import optimize, special, stats
from threadpoolctl import threadpool_info, threadpool_limits

try:
    import numba
except ImportError:
    numba = None

from ko_variation import glmm, kinship
from ko_variation.discovery_execution import bounded_map
from ko_variation.discovery_store import atomic_json, digest, open_patterns, records

STATE = {}


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
            raise ValueError('This benchmark compares intercept-only null models')
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


def weighted_tree(K,weights,kinship_eigensystem=None):
    return TreeWeights(weights)


def profile_tree(decomp,z,C,K,has_shared_covariance=None):
    """Same grid, expansion, bounded optimizer and tolerances as _profile_tau.

    The tree backend changes only working-objective/solve evaluation. The
    production source hash is frozen in benchmark_identity.json for restart.
    """
    working = TreeWorking(STATE['tree'],decomp.weights,C,z)
    if not has_shared_covariance:
        return 0.0,False,working.materialize(0.0)
    cache = {}
    def evaluate(tau):
        key = float(tau)
        if key not in cache: cache[key] = working.objective(key)
        return cache[key]
    candidates = [0.0]+list(np.logspace(glmm._TAU_GRID_MIN,glmm._TAU_GRID_MAX,25))
    def safe(tau):
        try: return evaluate(tau)
        except (np.linalg.LinAlgError,ValueError,FloatingPointError): return np.inf
    values = [safe(tau) for tau in candidates]
    best,boundary = int(np.argmin(values)),False
    while best==len(candidates)-1 and candidates[-1]<glmm._TAU_ABSOLUTE_MAX:
        tau = min(glmm._TAU_ABSOLUTE_MAX,candidates[-1]*100.0)
        candidates.append(float(tau)); values.append(safe(tau))
        best = int(np.argmin(values))
        if tau>=glmm._TAU_ABSOLUTE_MAX:
            boundary = best==len(candidates)-1; break
    best_tau,best_value = float(candidates[best]),float(values[best])
    if best>0:
        lo,hi = max(1e-14,float(candidates[best-1])),float(candidates[min(best+1,len(candidates)-1)])
        if hi>lo:
            try:
                result = optimize.minimize_scalar(lambda t:evaluate(float(math.exp(t))),
                    bounds=(math.log(lo),math.log(hi)),method='bounded',options={'xatol':1e-6,'maxiter':100})
                if result.success and np.isfinite(result.fun) and float(result.fun)<best_value:
                    best_tau,best_value = float(math.exp(result.x)),float(result.fun)
            except (ValueError,FloatingPointError,np.linalg.LinAlgError): pass
    return best_tau,boundary,working.materialize(best_tau)


def tree_scores(X,cache,tree):
    adjusted = X-cache['C']@(cache['M_inv']@(cache['Sinv_C'].T@X))
    white,_,_,_ = tree.apply('white',cache['weights'],cache['tau'],np.ascontiguousarray(adjusted))
    variable = np.ptp(X,axis=0)>0
    return np.where(variable,X.T@(cache['y']-cache['mu']),np.nan),np.where(variable,np.sum(white*white,axis=0),np.nan)


def storage(cache,method):
    """Serialized float64 cache bytes, not filesystem-throughput measurement."""
    data = {key:cache[key] for key in ('y','mu','weights','C','Sinv_C','M_inv','tau')}
    def size(payload):
        stream = io.BytesIO(); np.savez(stream,**payload); return stream.tell()
    if method=='tree': return dict(cache_bytes=size(data),compact_cache_bytes=size(data))
    d = cache['decomp']
    data.update(sqrt_w=d.sqrt_w,logdet_w_inverse=d.logdet_w_inverse)
    full = dict(data,eigenvalues=d.eigenvalues,eigenvectors=d.eigenvectors)
    compact = dict(data)
    if cache['tau']!=0 and not np.all(cache['weights']==cache['weights'][0]):
        order = np.argsort(d.eigenvalues)[::-1]
        compact.update(eigenvalues=d.eigenvalues[order],eigenvectors=d.eigenvectors[:,order[:128]])
    return dict(cache_bytes=size(full),compact_cache_bytes=size(compact))


def initialize(out,engine):
    threadpool_limits(limits=1)
    out = Path(out)
    identity = json.loads((out/'benchmark_identity.json').read_text())
    with np.load(out/'tree.npz') as arrays:
        tree = TreeAlgebra(*(arrays[k] for k in ('left','right','tip','length')),engine)
    K = np.load(out/'K.npy',mmap_mode='r')
    STATE.update(out=out,identity=identity,tree=tree,K=K,
        patterns=np.load(out/'selected.npy',mmap_mode='r'),
        eigen=glmm.prepare_kinship_eigensystem(K))
    start = time.perf_counter()
    w = np.full(len(K),.2); rhs = np.ones((len(K),2))
    tree.apply('white',w,.1,rhs);tree.apply('solve',w,.1,rhs);tree.apply('multiply',rhs)
    STATE['warmup_seconds'] = time.perf_counter()-start
    STATE['libraries'] = threadpool_info()


def finite(value):
    return float(value) if np.isfinite(value) else None


def probe_indices(identity,index,population):
    rng = np.random.default_rng(identity['seed']+index*1009)
    ids = rng.choice(population,size=identity['probes'],replace=population<identity['probes'])
    included=identity.get('included_selection')
    if included and index<included['responses']:
        old_rng=np.random.default_rng(included['seed']+index*1009)
        count=included['probes']
        ids[:count]=old_rng.choice(included['responses'],size=count,replace=included['responses']<count)
    return ids


def fit_job(job):
    index,repeat = job
    s = STATE; identity = s['identity']
    y = np.array(s['patterns'][index],dtype=float)
    ids=probe_indices(identity,index,len(s['patterns']))
    X = np.ascontiguousarray(s['patterns'][ids].T,dtype=float)
    # Identical/complement probes explicitly exercise difficult/high-score cases.
    X[:,0] = y
    if X.shape[1]>1: X[:,1] = 1-y
    fits,rows,scores = {},{},{}
    methods = ['dense','tree'] if (index+repeat)%2==0 else ['tree','dense']
    for method in methods:
        timing = {}; start = time.perf_counter()
        if method=='tree':
            with patch.object(glmm,'_weighted_eigen',weighted_tree),patch.object(glmm,'_profile_tau',profile_tree):
                fit,cache = glmm.fit_null_glmm(index,y,s['K'],kinship_eigensystem=s['eigen'],timing=timing)
        else:
            fit,cache = glmm.fit_null_glmm(index,y,s['K'],kinship_eigensystem=s['eigen'],timing=timing)
        seconds = time.perf_counter()-start
        fits[method] = fit
        row = dict(index=index,response=identity['responses'][index],repeat=repeat,method=method,
            n=len(y),n_present=int(y.sum()),status=fit.status,iterations=fit.iterations,
            seconds=seconds,tau=finite(fit.tau),intercept=finite(fit.beta[0]),objective=finite(fit.objective),
            max_eta_change=finite(fit.max_eta_change),weights_clipped=fit.n_weights_clipped,
            tau_boundary=fit.tau_boundary,message=fit.message,
            weighted_matrix_seconds=timing.get('weighted_eigendecomposition',0),
            profiling_seconds=timing.get('tau_profiling',0),score_seconds=None,
            cache_bytes=None,compact_cache_bytes=None)
        if fit.status=='OK':
            row.update(storage(cache,method))
            start = time.perf_counter()
            if method=='tree': U,V = tree_scores(X,cache,s['tree'])
            else:
                result = glmm.score_predictor_block(X,cache,compute_spa_adjustment=False)
                U,V = np.array([r.score_u for r in result]),np.array([r.score_variance for r in result])
            row['score_seconds'] = time.perf_counter()-start
            scores[method] = (U,V)
        rows[method] = row
    dense,tree = fits['dense'],fits['tree']
    comparison = dict(index=index,repeat=repeat,status_match=dense.status==tree.status,
                      jointly_successful=dense.status==tree.status=='OK',dense_status=dense.status,tree_status=tree.status)
    if comparison['jointly_successful']:
        comparison.update(max_probability_difference=float(np.max(np.abs(dense.fitted_probability-tree.fitted_probability))),
            intercept_abs_difference=abs(float(dense.beta[0]-tree.beta[0])),
            tau_relative_difference=abs(dense.tau-tree.tau)/max(1.,abs(dense.tau)),
            objective_relative_difference=abs(dense.objective-tree.objective)/max(1.,abs(dense.objective)),
            speedup=rows['dense']['seconds']/rows['tree']['seconds'])
        U0,V0 = scores['dense']; U1,V1 = scores['tree']
        usable0 = np.isfinite(V0)&(V0>1e-12); usable1 = np.isfinite(V1)&(V1>1e-12)
        both = usable0&usable1
        finite_u = np.isfinite(U0)&np.isfinite(U1)
        finite_v = np.isfinite(V0)&np.isfinite(V1)
        comparison.update(probes=len(V0),variance_status_mismatches=int(np.count_nonzero(usable0!=usable1)),
            jointly_valid_probes=int(np.count_nonzero(both)),
            max_score_u_difference=float(np.max(np.abs(U0[finite_u]-U1[finite_u]))) if np.any(finite_u) else 0.,
            max_variance_relative_difference=float(np.max(np.abs(V0[finite_v]-V1[finite_v])/np.maximum(np.abs(V0[finite_v]),1e-12))) if np.any(finite_v) else 0.)
        if np.any(both):
            t0,t1 = U0[both]**2/V0[both],U1[both]**2/V1[both]
            logp0,logp1 = 2*special.log_ndtr(-np.sqrt(t0)),2*special.log_ndtr(-np.sqrt(t1))
            comparison.update(max_log10p_difference=float(np.max(np.abs(logp0-logp1))/math.log(10)),
                max_statistic_absolute_difference=float(np.max(np.abs(t0-t1))),
                max_statistic_normalized_difference=float(np.max(np.abs(t0-t1)/np.maximum(1.,np.abs(t0)))),
                nominal_decision_mismatches=int(np.count_nonzero((t0>=identity['nominal_critical'])!=(t1>=identity['nominal_critical']))),
                bonferroni_decision_mismatches=int(np.count_nonzero((t0>=identity['critical'])!=(t1>=identity['critical']))),
                probes_near_bonferroni_boundary=int(np.count_nonzero(np.abs(t0-identity['critical'])<=.1*max(identity['critical'],1))),
                probes_very_near_bonferroni_boundary=int(np.count_nonzero(np.abs(t0-identity['critical'])<=1e-4*max(identity['critical'],1))))
        comparison['within_tolerances'] = bool(
            comparison['max_probability_difference']<=identity['probability_tolerance'] and
            comparison['intercept_abs_difference']<=identity['intercept_tolerance'] and
            comparison['objective_relative_difference']<=identity['objective_tolerance'] and
            comparison['tau_relative_difference']<=identity['tau_tolerance'] and
            comparison['max_variance_relative_difference']<=identity['variance_tolerance'] and
            not comparison['variance_status_mismatches'] and
            not comparison.get('nominal_decision_mismatches',0) and
            not comparison.get('bonferroni_decision_mismatches',0))
    return dict(index=index,repeat=repeat,rows=list(rows.values()),comparison=comparison,
                worker=dict(pid=os.getpid(),warmup_seconds=s['warmup_seconds'],libraries=s['libraries']))


def include_selection(args,selected,K,names,labels,family):
    """Retain known responses/probes, refitting rather than copying old results."""
    previous=args.include_from
    identity=json.loads((previous/'benchmark_identity.json').read_text())
    old=np.load(previous/'selected.npy',allow_pickle=False)
    old_K=np.load(previous/'K.npy',allow_pickle=False)
    products=json.loads((previous/'products.json').read_text())
    for name in ('selected.npy','K.npy'):
        if digest(previous/name)!=products[name]:raise ValueError('Included benchmark product changed: '+name)
    if identity['samples']!=names or not np.array_equal(old_K,K):
        raise ValueError('--include-from has a different cohort or covariance')
    paths=(args.tree,)+(tuple(args.work/name for name in ('prepared.json','patterns.bin','K.npy')) if args.work else (args.fasta,))
    if sorted(identity['inputs'].values())!=sorted(digest(path) for path in paths):
        raise ValueError('--include-from uses different source inputs')
    if identity['alpha']!=args.alpha or identity['bonferroni_tests']!=(args.bonferroni_tests or family):
        raise ValueError('--include-from uses a different alpha or Bonferroni family')
    if args.probes<identity['probes']:raise ValueError('--probes must retain at least the previous probe count')
    old_labels=identity['responses']
    if old.shape!=(len(old_labels),len(names)):raise ValueError('Included response dimensions changed')
    real=[i for i,label in enumerate(old_labels) if not label.startswith('synthetic:')]
    target=len(selected)
    if len(real)>target:raise ValueError('--patterns must retain at least the previous sampled response count')
    merged=[row.copy() for row in old];merged_labels=list(old_labels)
    seen={np.packbits(old[i]).tobytes() for i in real}
    if len(seen)!=len(real):raise ValueError('Included sampled responses are not unique patterns')
    count=len(real)
    for y,label in zip(selected,labels):
        key=np.packbits(y).tobytes()
        if count<target and key not in seen:
            merged.append(y);merged_labels.append(label);seen.add(key);count+=1
    if count<target:raise ValueError('Insufficient additional unique responses in sampled pool')
    args._included_selection=dict(directory=str(previous),identity_sha256=digest(previous/'benchmark_identity.json'),
        selected_sha256=products['selected.npy'],responses=len(old),sampled_responses=len(real),
        probes=identity['probes'],seed=identity['seed'])
    return np.asarray(merged,dtype='u1'),merged_labels


def select_inputs(args):
    """Read immutable prepared arrays, or sample FASTA columns by a frozen seed."""
    rng = np.random.default_rng(args.seed)
    if args.work:
        info = json.loads((args.work/'prepared.json').read_text())
        count = min(args.patterns,info['patterns'])
        ids = np.sort(rng.choice(info['patterns'],count,replace=False))
        packed = open_patterns(args.work,info)
        selected = np.unpackbits(np.asarray(packed[ids]),axis=1,bitorder='little')[:,:info['n']]
        K = np.load(args.work/'K.npy')
        names = info['samples']; labels = ['pattern:'+str(i) for i in ids]
        if args.tree is None:
            candidate = args.work/'analysis.treefile'
            if candidate.exists(): args.tree = candidate
            else: raise ValueError('Provide --tree matching the original prepared K.npy and sample order')
        for name in ('patterns.bin','K.npy'):
            if digest(args.work/name)!=info['products'][name]: raise ValueError('Prepared product changed: '+name)
        family = math.comb(info['patterns'],2)
    else:
        if not args.tree: raise ValueError('--tree is required with --fasta')
        names,width = [],None
        for name,sequence in records(args.fasta):
            if width is None: width=len(sequence)
            if len(sequence)!=width: raise ValueError('Unequal FASTA sequence lengths')
            names.append(name)
        if not names or len(set(names))!=len(names):
            raise ValueError('Use a nonempty sample-ID FASTA, not repeated contig headers')
        tree = kinship.build_tree_covariance(args.tree,names,missing_samples=args.tree_missing_samples)
        lookup = {names[int(old)]:i for i,old in enumerate(tree.sample_indices)}
        names = [name for name in names if name in lookup]
        if len(names)<4: raise ValueError('At least four matched samples required')
        candidates = np.sort(rng.choice(width,min(width,max(10*args.patterns,256)),replace=False))
        matrix = np.empty((len(names),len(candidates)),dtype='u1')
        for name,sequence in records(args.fasta):
            if name in lookup: matrix[lookup[name]] = np.frombuffer(sequence,dtype='u1')[candidates]==67
        seen,selected,labels = set(),[],[]
        for j,locus in enumerate(candidates):
            y=matrix[:,j]; af=float(y.mean()); key=np.packbits(y).tobytes()
            if min(af,1-af)>=args.min_maf and 0<af<1 and key not in seen:
                selected.append(y.copy());labels.append('locus:'+str(locus));seen.add(key)
        if not selected: raise ValueError('No sampled variable patterns pass MAF; use --work or increase --patterns')
        take=rng.choice(len(selected),min(args.patterns,len(selected)),replace=False)
        labels=[labels[i] for i in take];selected=np.asarray(selected,dtype='u1')[take]
        K,_ = glmm.prepare_kinship(tree.K)
        family = math.comb(width,2)  # Explicit conservative raw-locus family.
    algebra = TreeAlgebra.from_newick(args.tree,names,args.engine)
    expected = kinship.build_tree_covariance(args.tree,names).K
    error = float(np.max(np.abs(K-expected)))
    if error>1e-10:
        raise ValueError(f'Tree covariance differs from prepared K (max abs {error:.3g}); use its original rooted tree')
    if getattr(args,'include_from',None):
        selected,labels=include_selection(args,selected,K,names,labels,family)
    if args.stress_cases and not any(label.startswith('synthetic:') for label in labels):
        n=len(names); rare=np.zeros(n,dtype='u1');rare[rng.choice(n,max(1,int(.01*n)),replace=False)]=1
        clade=np.zeros(n,dtype='u1')
        internal=[i for i in range(len(algebra.tip)) if algebra.tip[i]<0]
        node=internal[len(internal)//2]
        pending=[node]
        while pending:
            v=pending.pop()
            if algebra.tip[v]>=0:clade[algebra.tip[v]]=1
            else:pending.extend((algebra.left[v],algebra.right[v]))
        selected=np.vstack((selected,rare,1-rare,clade,1-clade,np.zeros(n,dtype='u1'),np.ones(n,dtype='u1')))
        labels+=['synthetic:rare','synthetic:rare_complement','synthetic:clade','synthetic:clade_complement','synthetic:zero','synthetic:one']
    return np.asarray(selected,dtype='u1'),K,names,labels,algebra,max(1,family),error


def checkpoint(path,repair=True):
    payloads=[]
    if path.exists():
        with path.open('rb+' if repair else 'rb') as stream:
            while line:=stream.readline():
                if not line.endswith(b'\n'):
                    if repair:stream.truncate(stream.tell()-len(line))
                    break
                payloads.append(json.loads(line))
    return payloads


def table(path,rows):
    import csv
    fields=list(dict.fromkeys(key for row in rows for key in row)) or ['index','response','repeat','failure_reasons']
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fields,delimiter='\t');writer.writeheader();writer.writerows(rows)


def accuracy_failures(comparison,limits):
    """Describe fixed acceptance criteria; sensitivity never changes fitting."""
    if not comparison['jointly_successful']:
        if not comparison['status_match']:return ['FIT_STATUS_MISMATCH']
        return ['BOTH_MONOMORPHIC'] if comparison.get('dense_status')=='MONOMORPHIC_RESPONSE' else ['BOTH_FITS_FAILED']
    failures=[]
    for metric,limit in (
            ('max_probability_difference','probability_tolerance'),
            ('intercept_abs_difference','intercept_tolerance'),
            ('objective_relative_difference','objective_tolerance'),
            ('tau_relative_difference','tau_tolerance'),
            ('max_variance_relative_difference','variance_tolerance')):
        value=comparison.get(metric)
        if value is None:failures.append('MISSING_'+metric)
        elif value>limits[limit]:failures.append(metric)
    for metric in ('variance_status_mismatches','nominal_decision_mismatches','bonferroni_decision_mismatches'):
        if comparison.get(metric,0):failures.append(metric)
    return failures


def accuracy_summary(out,payloads,identity):
    """Join failures to response labels and report sensitivity without refitting."""
    limits={key:identity[key] for key in ('probability_tolerance','intercept_tolerance',
        'objective_tolerance','tau_tolerance','variance_tolerance')}
    outliers=[];groups={'sampled':[],'synthetic':[]};comparisons=[]
    for payload in payloads:
        methods={r['method']:r for r in payload['rows']}
        c=dict(payload['comparison'],dense_status=methods['dense']['status'],tree_status=methods['tree']['status'])
        comparisons.append(c)
        response=methods['dense']['response']
        category='synthetic' if response.startswith('synthetic:') else 'sampled'
        groups[category].append(c)
        failures=accuracy_failures(c,limits)
        if failures and failures!=['BOTH_MONOMORPHIC']:
            record=dict(index=c['index'],response=response,response_type=category,repeat=c['repeat'],
                        failure_reasons=';'.join(failures),**{k:v for k,v in c.items() if k not in ('index','repeat')})
            for method,row in methods.items():
                for key in ('status','iterations','tau','intercept','objective','max_eta_change','weights_clipped','tau_boundary','message'):
                    record[method+'_'+key]=row.get(key)
            outliers.append(record)
    table(out/'accuracy_outliers.tsv',outliers)
    good=[r for r in comparisons if r['jointly_successful']]
    sensitivity=[]
    for tolerance in sorted({1e-8,1e-7,2e-7,5e-7,1e-6,limits['probability_tolerance']}):
        current=dict(limits,probability_tolerance=tolerance)
        passing=[r for r in good if not accuracy_failures(r,current)]
        sensitivity.append(dict(probability_tolerance=tolerance,jointly_successful_jobs=len(good),
            probability_only_pass=sum(r['max_probability_difference']<=tolerance for r in good),
            all_criteria_pass=len(passing),unique_responses_all_completed_repeats_pass=len({r['index'] for r in good}
                -{r['index'] for r in comparisons if accuracy_failures(r,current)}),
            fit_status_mismatches=sum(not r['status_match'] for r in comparisons)))
    grouped={}
    for name,cs in groups.items():
        failed={r['index'] for r in cs if accuracy_failures(r,limits) not in ([],['BOTH_MONOMORPHIC'])}
        grouped[name]=dict(unique_responses=len({r['index'] for r in cs}),paired_jobs=len(cs),
            jointly_successful_jobs=sum(r['jointly_successful'] for r in cs),
            both_methods_failed_responses=len({r['index'] for r in cs if accuracy_failures(r,limits)==['BOTH_FITS_FAILED']}),
            status_mismatch_responses=len({r['index'] for r in cs if not r['status_match']}),
            responses_with_accuracy_failure=len(failed))
    quantiles={}
    for metric in ('max_probability_difference','intercept_abs_difference','tau_relative_difference',
            'objective_relative_difference','max_variance_relative_difference','max_log10p_difference',
            'max_statistic_absolute_difference','max_statistic_normalized_difference'):
        values=[r[metric] for r in good if r.get(metric) is not None]
        quantiles[metric]={label:float(np.quantile(values,q)) for label,q in
            (('median',.5),('p95',.95),('p99',.99),('max',1.))} if values else None
    return dict(configured_accuracy_tolerances=limits,accuracy_by_response_type=grouped,
        accuracy_error_quantiles=quantiles,probability_tolerance_sensitivity=sensitivity,
        accuracy_outlier_jobs=len(outliers),
        tolerance_note='Sensitivity varies only the probability acceptance threshold; other criteria and all fitted models are unchanged. Status failures never pass by relaxing probability tolerance.')


def report(out,payloads):
    rows=[r for p in payloads for r in p['rows']]; comparisons=[p['comparison'] for p in payloads]
    table(out/'null_results.tsv',rows);table(out/'comparisons.tsv',comparisons)
    summary={}
    for method in ('dense','tree'):
        rs=[r for r in rows if r['method']==method];good=[r for r in rs if r['status']=='OK']
        summary[method]=dict(fits=len(rs),statuses=dict(Counter(r['status'] for r in rs)),
            unique_responses=len({r['index'] for r in rs}),total_fit_worker_seconds=sum(r['seconds'] for r in rs),
            median_fit_seconds=float(np.median([r['seconds'] for r in good])) if good else None,
            median_score_seconds=float(np.median([r['score_seconds'] for r in good])) if good else None,
            median_cache_bytes=float(np.median([r['cache_bytes'] for r in good])) if good else None,
            median_compact_cache_bytes=float(np.median([r['compact_cache_bytes'] for r in good])) if good else None)
    good=[r for r in comparisons if r['jointly_successful']]
    summary.update(completed_jobs=len(payloads),status_mismatches=sum(not r['status_match'] for r in comparisons),
        jointly_successful_jobs=len(good),within_tolerances=sum(r['within_tolerances'] for r in good),
        median_paired_fit_speedup=float(np.median([r['speedup'] for r in good])) if good else None,
        nominal_decision_mismatches=sum(r.get('nominal_decision_mismatches',0) for r in good),
        bonferroni_decision_mismatches=sum(r.get('bonferroni_decision_mismatches',0) for r in good),
        jointly_valid_probe_comparisons=sum(r.get('jointly_valid_probes',0) for r in good),
        probes_near_bonferroni_boundary=sum(r.get('probes_near_bonferroni_boundary',0) for r in good),
        probes_very_near_bonferroni_boundary=sum(r.get('probes_very_near_bonferroni_boundary',0) for r in good)
            if any('probes_very_near_bonferroni_boundary' in r for r in good) else None,
        max_probability_difference=max((r['max_probability_difference'] for r in good),default=None),
        max_tau_relative_difference=max((r['tau_relative_difference'] for r in good),default=None),
        max_variance_relative_difference=max((r['max_variance_relative_difference'] for r in good),default=None),
        production_modified=False,methods='Current dense PQL versus the same outer PQL loop with tree working algebra; float64 and sigmoid unchanged',
        notes=['Fit time excludes preparation, JIT warmup, scoring and serialization.',
               'Cache sizes are in-memory NPZ serialization measurements; not filesystem throughput or peak RAM.',
               'Compact dense size uses rank 128 and zero/shared special cases; global K/eigen/tree and source patterns excluded.',
               'Repeated runs of a pattern are timing repeats, not independent biological observations.',
               'Accuracy conclusions apply to tested patterns/probes only; inspect failures and boundary coverage.'])
    identity=json.loads((out/'benchmark_identity.json').read_text())
    summary.update(accuracy_summary(out,payloads,identity))
    summary.update(retained_previous_responses=(identity.get('included_selection') or {}).get('responses',0),
                   retained_previous_probes=(identity.get('included_selection') or {}).get('probes',0))
    summary.update(report_source_sha256=digest(Path(__file__)),
                   fit_source_sha256=identity['sources'][Path(__file__).name])
    atomic_json(out/'summary.json',summary)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['svg.fonttype']='none'
        fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
        for ax,key,label in zip(axes,('seconds','compact_cache_bytes','max_probability_difference'),
                ('Null fit time (s)','Serialized cache (bytes)','Maximum probability difference')):
            if key=='max_probability_difference':
                ax.scatter(range(len(good)),[r[key] for r in good],s=12,color='black');ax.set_xlabel('Paired fit')
            else:
                data=[[r[key] for r in rows if r['method']==m and r['status']=='OK'] for m in ('dense','tree')]
                for position,values in enumerate(data,1):
                    if values:ax.boxplot(values,positions=[position],widths=.5,
                        boxprops={'color':'black'},medianprops={'color':'black'})
                ax.set_xticks([1,2],['Dense','Tree']);ax.set_yscale('log')
            ax.set_ylabel(label);ax.spines[['top','right']].set_visible(False)
        for suffix in ('png','svg'):fig.savefig(out/('tree_null_benchmark.'+suffix),dpi=300)
        plt.close(fig)
    except ImportError:
        print('Plot skipped: install matplotlib; --plot-only redraws saved results.',flush=True)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group();source.add_argument('--work',type=Path);source.add_argument('--fasta',type=Path)
    p.add_argument('--tree',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--patterns',type=int,default=50);p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--probes',type=int,default=8);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--min-maf',type=float,default=.05)
    p.add_argument('--include-from',type=Path,help='Retain all responses and probe vectors from an earlier benchmark; refit them')
    p.add_argument('--tree-missing-samples',choices=['error','drop'],default='error')
    p.add_argument('--engine',choices=['numba','python'],default='numba')
    p.add_argument('--stress-cases',action='store_true');p.add_argument('--plot-only',action='store_true')
    p.add_argument('--alpha',type=float,default=.05);p.add_argument('--bonferroni-tests',type=int)
    p.add_argument('--probability-tolerance',type=float,default=1e-7)
    p.add_argument('--tau-tolerance',type=float,default=1e-5)
    p.add_argument('--variance-tolerance',type=float,default=1e-6)
    p.add_argument('--intercept-tolerance',type=float,default=1e-6)
    p.add_argument('--objective-tolerance',type=float,default=1e-7)
    args=p.parse_args(argv);args.out=args.out.resolve()
    if args.plot_only:
        payloads=checkpoint(args.out/'checkpoint.jsonl',repair=False)
        if not payloads:p.error('No completed benchmark jobs to plot')
        report(args.out,payloads);return
    if not (args.work or args.fasta):p.error('Provide --work or --fasta')
    if min(args.patterns,args.repeats,args.probes,args.threads)<1:p.error('Counts/threads must be positive')
    if not 0<args.alpha<1 or not 0<=args.min_maf<=.5:p.error('Invalid alpha/MAF')
    if args.bonferroni_tests is not None and args.bonferroni_tests<1:p.error('Bonferroni family must be positive')
    if min(args.probability_tolerance,args.tau_tolerance,args.variance_tolerance,args.intercept_tolerance,args.objective_tolerance)<=0:p.error('Tolerances must be positive')
    if args.engine=='numba' and numba is None:p.error('Install numba: python -m pip install numba; or use --engine python for correctness testing')
    if args.work:args.work=args.work.resolve()
    if args.fasta:args.fasta=args.fasta.resolve()
    if args.include_from:args.include_from=args.include_from.resolve()
    for original in (args.work,args.fasta.parent if args.fasta else None,args.include_from):
        if original and (args.out==original or args.out in original.parents or original in args.out.parents):
            p.error('--out must be separate from source work/FASTA directories')
    args.out.mkdir(parents=True,exist_ok=True)
    # A single coordinator owns checkpoints and frozen input publication.
    import sqlite3
    lock=sqlite3.connect(args.out/'coordinator.lock.sqlite',timeout=0)
    try:
        lock.execute('BEGIN EXCLUSIVE')
        started=time.perf_counter()
        selected,K,names,labels,tree,family,covariance_error=select_inputs(args)
        family=args.bonferroni_tests or family
        identity=dict(revision=1,inputs={str(path):digest(path) for path in
            (args.tree,)+(tuple(args.work/name for name in ('prepared.json','patterns.bin','K.npy')) if args.work else (args.fasta,))},
            samples=names,responses=labels,seed=args.seed,repeats=args.repeats,probes=args.probes,
            engine=args.engine,stress_cases=args.stress_cases,alpha=args.alpha,bonferroni_tests=family,
            included_selection=getattr(args,'_included_selection',None),
            critical=float(stats.chi2.isf(args.alpha/family,1)),nominal_critical=float(stats.chi2.isf(.05,1)),
            probability_tolerance=args.probability_tolerance,tau_tolerance=args.tau_tolerance,variance_tolerance=args.variance_tolerance,
            intercept_tolerance=args.intercept_tolerance,objective_tolerance=args.objective_tolerance,
            numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__ if numba else None,python=platform.python_version(),
            sources={str(path.name):digest(path) for path in (Path(__file__),Path(glmm.__file__),Path(kinship.__file__))},
            tree_covariance_max_absolute_difference=covariance_error)
        manifest=args.out/'benchmark_identity.json'
        if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Benchmark inputs/model/settings changed; use another --out')
        if not manifest.exists():
            atomic_json(manifest,identity)
        # Verify selected frozen products before resuming; do not reread any null caches or source SQL.
        products=args.out/'products.json'
        if not products.exists():
            np.save(args.out/'selected.npy',selected);np.save(args.out/'K.npy',K)
            np.savez(args.out/'tree.npz',left=tree.left,right=tree.right,tip=tree.tip,length=tree.length)
        expected={name:digest(args.out/name) for name in ('selected.npy','K.npy','tree.npz')}
        if products.exists() and json.loads(products.read_text())!=expected:raise ValueError('Frozen benchmark products changed')
        if not products.exists():atomic_json(products,expected)
        preparation=time.perf_counter()-started
        payloads=checkpoint(args.out/'checkpoint.jsonl');done={(r['index'],r['repeat']) for r in payloads}
        jobs=[(i,r) for i in range(len(labels)) for r in range(args.repeats) if (i,r) not in done]
        print(f'[tree null benchmark] {len(names)} samples; {len(labels)} responses; {len(done)} jobs resumed; {len(jobs)} pending; CPUs={args.threads}',flush=True)
        search=time.perf_counter()
        if jobs:
            with ProcessPoolExecutor(max_workers=args.threads,mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize,initargs=(str(args.out),args.engine)) as pool:
                with (args.out/'checkpoint.jsonl').open('ab') as stream:
                    for payload in bounded_map(pool,fit_job,jobs,args.threads,ordered=False):
                        stream.write((json.dumps(payload,allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
                        payloads.append(payload)
                        print(f'[tree null benchmark] saved {len(payloads)}/{len(labels)*args.repeats} paired jobs',flush=True)
        summary=report(args.out,payloads)
        atomic_json(args.out/'runtime.json',dict(preparation_seconds=preparation,
            benchmark_seconds_this_invocation=time.perf_counter()-search,
            workers={str(r['worker']['pid']):r['worker'] for r in payloads},
            note='Worker warmup includes Numba compilation/cache loading; per-fit timers exclude it.'))
        print('[tree null benchmark] complete: '+str(args.out/'summary.json'),flush=True)
    finally:lock.close()


if __name__=='__main__':main()
