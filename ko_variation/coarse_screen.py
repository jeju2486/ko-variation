"""Global tree-PC screen before coarse KOVAR task creation.

Residuals are float64; absolute correlations are ranked as float32. Sequential
radix passes find the global cutoff without loading all pair scores into RAM.
Only retained node pairs persist. Undefined residual correlations are rescued.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .discovery_store import atomic_json, open_patterns


def array_file(path, values):
    temporary=path.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.save(stream,values)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def residuals(directory, work, info, representatives, pcs, block):
    K=np.array(np.load(work/'K.npy',mmap_mode='r'))
    K-=K.mean(axis=0)[None,:]+K.mean(axis=1)[:,None]-K.mean()
    values,vectors=np.linalg.eigh((K+K.T)*.5)
    available=int(np.count_nonzero(values>max(1e-12,float(values[-1])*1e-10)))
    used=min(pcs,available,max(0,info['n']-2))
    axes=vectors[:,-used:] if used else np.empty((info['n'],0))
    Q=np.linalg.qr(np.column_stack((np.ones(info['n']),axes)))[0]
    packed=open_patterns(work,info)
    temporary=directory/'residuals.partial.npy'
    Z=np.lib.format.open_memmap(temporary,mode='w+',dtype='f8',shape=(len(representatives),info['n']))
    invalid=np.zeros(len(representatives),dtype=bool)
    for start in range(0,len(representatives),block):
        X=np.unpackbits(packed[representatives[start:start+block]],axis=1,bitorder='little')[:,:info['n']].astype(float)
        R=X-(X@Q)@Q.T
        norms=np.linalg.norm(R,axis=1)
        bad=(~np.isfinite(norms)) | (norms<=1e-10*math.sqrt(info['n']))
        R[bad]=0;norms[bad]=1
        Z[start:start+len(R)]=R/norms[:,None]
        invalid[start:start+len(R)]=bad
    Z.flush();del Z
    os.replace(temporary,directory/'residuals.npy')
    array_file(directory/'invalid.npy',invalid)
    return used,int(invalid.sum())


def threshold(path, count, keep, threads, chunk, state, save):
    """Exact order statistic of nonnegative float32 scores, including +inf."""
    mapped=np.memmap(path,mode='r',dtype='<u4',shape=(count,))
    prefix=state.get('radix_prefix',0);mask=state.get('radix_mask',0)
    rank=state.get('radix_rank',count-keep)
    start_pass=state.get('radix_pass',0)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for step in range(start_pass,4):
            shift=24-8*step
            def histogram(start):
                bits=mapped[start:min(start+chunk,count)]
                if mask: bits=bits[(bits&mask)==prefix]
                return np.bincount((bits>>shift)&255,minlength=256)
            hist=np.zeros(256,dtype=np.int64)
            for part in pool.map(histogram,range(0,count,chunk)): hist+=part
            cumulative=np.cumsum(hist)
            bucket=int(np.searchsorted(cumulative,rank,side='right'))
            rank-=int(cumulative[bucket-1]) if bucket else 0
            prefix|=bucket<<shift;mask|=255<<shift
            state.update(radix_pass=step+1,radix_prefix=prefix,radix_mask=mask,radix_rank=rank,
                         threshold_ties=int(hist[bucket])-rank)
            save()
            print(f'[KOVAR] PC screen: global cutoff pass {step+1}/4 complete',flush=True)
    return prefix,state['threshold_ties']


def prepare_screen(db, work, info, args, get_meta, set_meta):
    """Bounded RAM and stage checkpoints; no SQLite row per rejected pair."""
    ready=get_meta(db,'coarse_screen_summary')
    directory=work/'coarse_screen';directory.mkdir(exist_ok=True)
    if ready:
        if (directory/'candidates.u32').stat().st_size!=ready['selected_pairs']*8:
            raise ValueError('Saved coarse candidate file is truncated')
        if not (args.out/'coarse_screen_summary.json').exists():
            atomic_json(args.out/'coarse_screen_summary.json',ready)
        for name in ('scores.f32','residuals.npy','invalid.npy'):
            (directory/name).unlink(missing_ok=True)
        return ready
    started=time.monotonic()
    records=db.execute('SELECT n.id,r.pid FROM nodes n JOIN reps r ON r.node=n.id '
                       'WHERE n.parent=0 ORDER BY r.pid,n.id').fetchall()
    roots=np.array([r[0] for r in records],dtype='i8')
    representatives=np.array([r[1] for r in records],dtype='i8')
    count=len(roots)*(len(roots)-1)//2
    block=min(args.screen_block_size,max(1,int(args.memory_gb*(1<<30)*.05/max(12*len(roots)+8*info['n'],1))))
    state_path=directory/'progress.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(stage='residuals',score_row=1)
    previous_seconds=state.get('elapsed_total_seconds',0.)
    def save():
        state['elapsed_total_seconds']=previous_seconds+time.monotonic()-started
        atomic_json(state_path,state)
        with db:
            set_meta(db,'coarse_screen_progress',state)
            set_meta(db,'active_stage','coarse_screen')
    if state['stage']=='residuals':
        array_file(directory/'roots.npy',roots)
        with threadpool_limits(limits=args.threads):
            used,bad=residuals(directory,work,info,representatives,args.screen_pcs,block)
        forced=count-(len(roots)-bad)*(len(roots)-bad-1)//2
        scorable=count-forced
        keep=forced+(max(1,int(scorable*args.coarse_retain)) if scorable else 0)
        state.update(stage='correlations',total_pairs=count,selected_pairs=keep,pcs_used=used,
                     undefined_representatives=bad,forced_pairs=forced)
        save()
    scores=directory/'scores.f32'
    if state['stage']=='correlations':
        if state['score_row']>1 and (not scores.exists() or scores.stat().st_size!=count*4):
            raise ValueError('Committed PC score file is missing/truncated; restore it before resuming')
        remaining=max(0,count-state.get('scored_pairs',0))*4+state['selected_pairs']*8
        if shutil.disk_usage(directory).free<remaining:
            raise ValueError(f'PC screen needs approximately {remaining/(1<<30):.2f} GiB additional temporary/candidate space')
        Z=np.load(directory/'residuals.npy',mmap_mode='r')
        invalid=np.load(directory/'invalid.npy')
        mode='r+b' if scores.exists() else 'w+b'
        with scores.open(mode) as stream,threadpool_limits(limits=args.threads):
            stream.truncate(count*4)
            for first in range(state['score_row'],len(roots),block):
                end=min(first+block,len(roots))
                values=np.abs(Z[first:end]@Z[:end].T)
                np.clip(values,0,1,out=values)
                stream.seek(first*(first-1)//2*4)
                for row in range(first,end):
                    ranked=values[row-first,:row].astype('<f4')
                    if invalid[row]: ranked.fill(np.inf)
                    else: ranked[invalid[:row]]=np.inf
                    ranked.tofile(stream)
                stream.flush();os.fsync(stream.fileno())
                state.update(score_row=end,scored_pairs=end*(end-1)//2)
                save()
                print(f"[KOVAR] PC{state['pcs_used']} correlations: {state['scored_pairs']:,}/{count:,}",flush=True)
        del Z
        state['stage']='selection';save()
    if state['stage']=='selection':
        chunk=max(65536,min(1<<20,int(args.memory_gb*(1<<30)*.05/max(args.threads*24,1))))
        if state['selected_pairs']:
            cutoff,ties=threshold(scores,count,state['selected_pairs'],args.threads,chunk,state,save)
        else: cutoff,ties=0,0
        state.update(stage='emit',cutoff_bits=int(cutoff),ties_left=int(ties),emit_row=1,emitted=0)
        save()
    temporary=directory/'candidates.partial';final=directory/'candidates.u32'
    if state['stage']=='emit':
        # Response-major order and deterministic ties; truncate uncommitted tail.
        if not final.exists():
            offsets_path=directory/'candidate_offsets.i64'
            if state['emitted'] and (not temporary.exists() or temporary.stat().st_size<state['emitted']*8 or
                    not offsets_path.exists() or offsets_path.stat().st_size<(state['emit_row']+1)*8):
                raise ValueError('Committed PC candidate prefix is missing/truncated')
            mode='r+b' if temporary.exists() else 'w+b'
            offset_mode='r+b' if offsets_path.exists() else 'w+b'
            with temporary.open(mode) as target,scores.open('rb') as source,offsets_path.open(offset_mode) as offsets:
                target.truncate(state['emitted']*8);target.seek(state['emitted']*8)
                first=state['emit_row'];source.seek(first*(first-1)//2*4)
                if first==1:
                    offsets.seek(0);np.zeros(2,dtype='<i8').tofile(offsets)
                offsets.truncate((first+1)*8);offsets.seek((first+1)*8)
                for row in range(first,len(roots)):
                    values=np.fromfile(source,dtype='<u4',count=row)
                    selected=values>state['cutoff_bits']
                    equal=np.flatnonzero(values==state['cutoff_bits'])
                    take=min(len(equal),state['ties_left'])
                    selected[equal[:take]]=True;state['ties_left']-=take
                    left=np.flatnonzero(selected)
                    pairs=np.empty((len(left),2),dtype='<u4');pairs[:,0]=left;pairs[:,1]=row
                    pairs.tofile(target);state['emitted']+=len(left)
                    np.array([state['emitted']],dtype='<i8').tofile(offsets)
                    if (row+1)%block==0 or row+1==len(roots):
                        target.flush();os.fsync(target.fileno());offsets.flush();os.fsync(offsets.fileno())
                        state['emit_row']=row+1;save()
                target.flush();os.fsync(target.fileno())
                offsets.flush();os.fsync(offsets.fileno())
            os.replace(temporary,final)
        if state['emitted']!=state['selected_pairs']:
            raise ValueError('Coarse candidate count differs from the global selection budget')
        state['stage']='complete';save()
    cutoff=float(np.array([state['cutoff_bits']],dtype='<u4').view('<f4')[0])
    summary=dict(method='tree_pc_residual_correlation',requested_pcs=args.screen_pcs,
        effective_pcs=state['pcs_used'],retained_fraction=args.coarse_retain,
        total_pairs=count,selected_pairs=state['selected_pairs'],filtered_pairs=count-state['selected_pairs'],
        scorable_pairs=count-state['forced_pairs'],ranked_pairs_kept=state['selected_pairs']-state['forced_pairs'],
        undefined_representatives=state['undefined_representatives'],forced_pairs=state['forced_pairs'],
        cutoff=float(cutoff) if np.isfinite(cutoff) else None,ranking_dtype='float32',
        tie_rule='response-major representative order',seconds_this_invocation=time.monotonic()-started,
        elapsed_total_seconds=state['elapsed_total_seconds'],heuristic=True)
    with db: set_meta(db,'coarse_screen_summary',summary)
    atomic_json(args.out/'coarse_screen_summary.json',summary)
    for name in ('scores.f32','residuals.npy','invalid.npy'):
        (directory/name).unlink(missing_ok=True)
    print(f"[KOVAR] PC screen retained {summary['selected_pairs']:,}/{count:,} coarse pairs; "
          f"undefined-correlation rescue pairs={summary['forced_pairs']:,}",flush=True)
    return summary


def open_candidates(work, count):
    directory=work/'coarse_screen'
    roots=np.load(directory/'roots.npy')
    pairs=np.memmap(directory/'candidates.u32',mode='r',dtype='<u4',shape=(count,2)) if count else np.empty((0,2),dtype='u4')
    offsets=np.fromfile(directory/'candidate_offsets.i64',dtype='<i8')
    if len(offsets)!=len(roots)+1 or int(offsets[-1])!=count:
        raise ValueError('Coarse candidate row offsets differ from the saved candidate count')
    return roots,pairs,offsets


def candidate_tasks(db, arrays, capacity, get_meta, set_meta, workers=1):
    roots,pairs,offsets=arrays
    offset=get_meta(db,'pc_coarse_cursor',0)
    total=len(pairs)
    if offset>=total: return [],True
    active=get_meta(db,'pc_coarse_columns',[])
    next_right=get_meta(db,'pc_coarse_next',1)
    quantum=max(1,min(256,capacity//max(1,workers)))
    tasks=[]
    while len(tasks)<capacity:
        while len(active)<workers and next_right<len(roots):
            if offsets[next_right]<offsets[next_right+1]:
                active.append([next_right,int(offsets[next_right])])
            next_right+=1
        if not active: break
        right,position=active.pop(0)
        stop=min(int(offsets[right+1]),position+quantum,position+capacity-len(tasks))
        tasks.extend(tuple(sorted((int(roots[a]),int(roots[b])))) for a,b in pairs[position:stop])
        if stop<offsets[right+1]: active.append([right,stop])
    with db:
        db.executemany('INSERT OR IGNORE INTO tasks(a,b) VALUES (?,?)',tasks)
        set_meta(db,'pc_coarse_cursor',offset+len(tasks))
        set_meta(db,'pc_coarse_columns',active)
        set_meta(db,'pc_coarse_next',next_right)
        set_meta(db,'coarse_scheduled_count',offset+len(tasks))
    return tasks,offset+len(tasks)>=total
