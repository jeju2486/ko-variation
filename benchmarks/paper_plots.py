"""Editable PNG/SVG network figures from a saved map layout; no QQ or refits."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

import numpy as np

from ko_variation.checkpoint import fingerprint_file
from ko_variation.discovery_store import atomic_json
from ko_variation.functional_annotation import FUNCTION_COLORS


def normalise(value):
    return re.sub('[^a-z0-9]','',str(value).lower())


def gene_name(node):
    name = node.get('annotation_names') or node.get('eggnog_preferred_name') or ''
    if normalise(name) in {'','hypotheticalprotein','unannotatedprotein','unknown','putativeprotein'}:
        return ''
    if re.fullmatch(r'group_\d+',name,re.I):
        return ''
    if node.get('n_coding_loci') == 0 and node.get('n_nearby_loci',0) > 0:
        name = 'near '+name
    if not node.get('annotation_names'):
        name += ' (predicted)'
    return name


def wrap_label(text, inches, font):
    """Measure label widths in points; split long names without truncating them."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath
    prop = FontProperties(size=font)
    renderer = TextToPath()
    lines, current = [], ''
    for char in text:
        proposed = current+char
        width = renderer.get_text_width_height_descent(proposed,prop,ismath=False)[0]
        if current and width > inches*72:
            lines.append(current.rstrip())
            current = char.lstrip()
        else:
            current = proposed
    if current: lines.append(current.rstrip())
    return lines or ['']


def panel_data(ids, nodes, width, font):
    ordered = sorted(ids)
    raw = np.array([[nodes[g]['x'],nodes[g]['y']] for g in ordered],dtype=float)
    spread = np.ptp(raw,axis=0)
    xy = (raw-raw.min(axis=0))/np.where(spread>0,spread,1)
    xy[:,0] = np.where(spread[0]>0,.30+.40*xy[:,0],.5)
    xy[:,1] = np.where(spread[1]>0,.90-.80*xy[:,1],.5)
    positions = dict(zip(ordered,xy))
    named = [(g,gene_name(nodes[g])) for g in ordered if gene_name(nodes[g])]
    # Numeric suffixes distinguish equal names without exposing group IDs.
    duplicates = Counter(name for _,name in named)
    seen = Counter()
    left, right = [], []
    for g,name in sorted(named,key=lambda item:(positions[item[0]][0],item[0])):
        seen[name] += 1
        if duplicates[name]>1: name += f' [{seen[name]}]'
        entry = (g,wrap_label(name,width*.21,font))
        (left if len(left)<math.ceil(len(named)/2) else right).append(entry)
    for side in (left,right):
        side.sort(key=lambda item:(-positions[item[0]][1],item[0]))
    lines = max(sum(len(v) for _,v in side) for side in (left,right))
    height = max(2.6,(lines+max(len(left),len(right))*.45)*font*1.35/72/.82)
    return positions,left,right,height


def draw_panel(ax, ids, nodes, edges, focus, width, height, font):
    positions,left,right,_ = panel_data(ids,nodes,width,font)
    marker_scale = 1.0
    if len(ids)>1:
        from scipy.spatial import cKDTree
        points = np.array([positions[g] for g in sorted(ids)])*np.array([width,height])*72
        nearest = cKDTree(points).query(points,k=2)[0][:,1].min()
        marker_scale = min(1.0,max(.05,nearest/(10 if focus else 7)))
    for edge in edges:
        if edge['gene_a'] not in ids or edge['gene_b'] not in ids:
            continue
        a,b = positions[edge['gene_a']],positions[edge['gene_b']]
        direction = edge.get('adjusted_direction','unknown')
        style = '--' if direction=='negative' else '-.' if direction=='mixed' else ':' if direction=='unknown' else '-'
        ax.plot([a[0],b[0]],[a[1],b[1]],color='#555555',alpha=.65,
                linewidth=max(.45,float(edge.get('display_width',.7))*.65),linestyle=style,zorder=1)
    for g in sorted(ids):
        node = nodes[g]
        radius = 4.5 if g in focus else 3.0 if gene_name(node) else 1.6
        color = FUNCTION_COLORS.get(node.get('function_category','Unassigned'),FUNCTION_COLORS['Unassigned'])
        ax.scatter(*positions[g],s=(2*radius*marker_scale)**2,c=[color],
                   edgecolors='#D00000' if g in focus else 'white',
                   linewidths=1.4 if g in focus else .35,zorder=3)
    for side,right_side in ((left,False),(right,True)):
        line_height = font*1.35/72/height
        gap = font*.6/72/height
        total = sum(len(lines)*line_height for _,lines in side)+max(0,len(side)-1)*gap
        top = .5+total/2
        for g,lines in side:
            block = len(lines)*line_height
            y = top-block/2
            x,end = (.76,.74) if right_side else (.24,.26)
            ax.plot([positions[g][0],end,x],[positions[g][1],y,y],color='#B0B0B0',lw=.4,zorder=0)
            ax.text(x,y,'\n'.join(lines),ha='left' if right_side else 'right',va='center',
                    fontsize=font,linespacing=1.35,color='#D00000' if g in focus else 'black',
                    parse_math=False)
            top -= block+gap
    ax.set(xlim=(0,1),ylim=(0,1))
    ax.axis('off')


def network_page(groups, nodes, edges, out, name, focus, font, dpi, dimer_note=''):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    columns = min(2,max(1,len(groups)))
    panel_width = 4.2
    rows = max(1,math.ceil(len(groups)/columns))
    heights = [max(panel_data(g,nodes,panel_width*.9,font)[3] for g in groups[i:i+columns])
               for i in range(0,len(groups),columns)] or [3.0]
    categories = sorted({nodes[g].get('function_category','Unassigned') for group in groups for g in group})
    footer = 1.0+.22*math.ceil(len(categories)/2)
    fig = plt.figure(figsize=(panel_width*columns,sum(heights)+footer))
    grid = fig.add_gridspec(rows,columns,height_ratios=heights,
                           left=.015,right=.985,top=.995,bottom=footer/(sum(heights)+footer),hspace=.08,wspace=.04)
    for i,group in enumerate(groups):
        ax = fig.add_subplot(grid[i//columns,i%columns])
        bounds = ax.get_position()
        draw_panel(ax,group,nodes,edges,focus,bounds.width*fig.get_figwidth(),
                   bounds.height*fig.get_figheight(),font)
    if not groups:
        fig.text(.5,.6,'No components meet this display criterion',ha='center')
    handles = [Line2D([],[],marker='o',color='none',markerfacecolor=FUNCTION_COLORS.get(c,FUNCTION_COLORS['Unassigned']),
                      markersize=5,label=c) for c in categories]
    if focus:
        handles.append(Line2D([],[],marker='o',color='none',markeredgecolor='#D00000',
                              markerfacecolor='white',markersize=6,label='Focal gene'))
    handles += [Line2D([],[],color='#555555',lw=.8,label='Positive coefficient'),
                Line2D([],[],color='#555555',lw=.8,linestyle='--',label='Negative coefficient')]
    if any(e.get('adjusted_direction')=='mixed' for e in edges):
        handles.append(Line2D([],[],color='#555555',lw=.8,linestyle='-.',label='Mixed signs across unitig pairs'))
    if any(e.get('adjusted_direction')=='unknown' for e in edges):
        handles.append(Line2D([],[],color='#555555',lw=.8,linestyle=':',label='Beta unavailable'))
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,fontsize=7,
               bbox_to_anchor=(.5,.07/(sum(heights)+footer)))
    if dimer_note:
        fig.text(.5,(footer-.14)/(sum(heights)+footer),dimer_note,ha='center',fontsize=8)
    files = []
    for suffix in ('svg','png'):
        path = out/f'{name}.{suffix}'
        fig.savefig(path,dpi=dpi,facecolor='white')
        files.append(str(path))
    plt.close(fig)
    return files


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout',type=Path,required=True,help='map.layout.json or functional-map directory')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--focus',action='append',help='Exact gene alias or stable ID; repeat to add targets [mprF]')
    p.add_argument('--min-component-nodes',type=int,default=3)
    p.add_argument('--components-per-page',type=int,default=6)
    p.add_argument('--label-fontsize',type=float,default=7)
    p.add_argument('--dpi',type=int,default=300)
    args = p.parse_args(argv)
    if min(args.min_component_nodes,args.components_per_page,args.dpi,args.label_fontsize)<=0:
        p.error('Display sizes must be positive')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import networkx as nx
    plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans'})
    source = args.layout/'map.layout.json' if args.layout.is_dir() else args.layout
    payload = json.loads(source.read_text())
    args.out = args.out.resolve()
    if args.out == source.resolve().parent:
        p.error('Use a separate paper-figure output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    nodes = {n['gene']:n for n in payload['nodes']}
    if len(nodes)!=len(payload['nodes']):
        raise ValueError('Duplicate stable gene IDs in layout')
    for node in nodes.values():
        if not np.isfinite([node['x'],node['y']]).all():
            raise ValueError('Layout has missing/nonfinite coordinates')
    edges = payload['edges']
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for e in edges:
        if e['gene_a'] not in nodes or e['gene_b'] not in nodes:
            raise ValueError('Edge endpoint absent from node table')
        graph.add_edge(e['gene_a'],e['gene_b'])
    components = sorted(nx.connected_components(graph),key=lambda g:(-len(g),sorted(g)))
    retained = [g for g in components if len(g)>=args.min_component_nodes]
    dimers = sum(len(g)==2 for g in components)
    isolates = sum(len(g)==1 for g in components)
    note = f'+{dimers:,} two-gene components' if args.min_component_nodes>2 and dimers else ''
    if isolates and args.min_component_nodes>1:
        note += ('; ' if note else '')+f'{isolates:,} isolated genes omitted'
    outputs, captions = [], []
    pages = [retained[i:i+args.components_per_page] for i in range(0,len(retained),args.components_per_page)] or [[]]
    for i,groups in enumerate(pages):
        name = 'network_overview' if len(pages)==1 else f'network_overview_{i+1:02d}'
        outputs += network_page(groups,nodes,edges,args.out,name,set(),args.label_fontsize,args.dpi,note)
        drawn = set().union(*groups) if groups else set()
        captions.append(f'{name}: {len(drawn):,} genes in {len(groups):,} connected components on this page. '
                        f'The complete network contains {len(nodes):,} genes and {len(edges):,} gene edges. '
                        f'Overview component cutoff: {args.min_component_nodes} genes. '
                        f'{dimers} two-gene components and {isolates} isolated genes occur in the complete network.')
    focuses = {}
    for target in args.focus or ['mprF']:
        query = normalise(target)
        matches = set()
        for gene,node in nodes.items():
            aliases = [gene]+re.split(r';\s*',node.get('annotation_names',''))+[node.get('eggnog_preferred_name','')]
            if any(normalise(alias)==query for alias in aliases):
                matches.add(gene)
        if not matches:
            focuses[target] = dict(status='NOT_IN_SELECTED_NETWORK',matched_gene_ids=[])
            print(f'[paper] focal target {target}: not in the selected network; no focal panel generated',flush=True)
            continue
        focuses[target] = dict(status='DRAWN',matched_gene_ids=sorted(matches),neighbourhoods={})
        for depth in (1,2):
            group = set().union(*(nx.single_source_shortest_path_length(graph,g,cutoff=depth) for g in matches))
            name = 'focus_'+(query or 'gene')+f'_N{depth}'
            outputs += network_page([group],nodes,edges,args.out,name,matches,args.label_fontsize,args.dpi)
            n_edges = graph.subgraph(group).number_of_edges()
            focuses[target]['neighbourhoods'][str(depth)] = dict(genes=len(group),edges=n_edges)
            captions.append(f'{name}: {len(group)} genes and {n_edges} gene edges within {depth} '
                            f'network steps of {len(matches)} matching cluster(s) for {target}. '
                            'Red outlines and text identify focal clusters; fill retains functional assignment.')
    # These summaries use the complete network, including omitted dimers.
    sizes = Counter(map(len,components))
    function_counts = Counter(n.get('function_category','Unassigned') for n in nodes.values())
    for name,counts,xlabel,ylabel in (
        ('component_sizes',sizes,'Genes per component','Number of components'),
        ('functional_composition',function_counts,'Genes','Predicted functional category')):
        fig,ax = plt.subplots(figsize=(6.4,4.2),layout='constrained')
        if name=='component_sizes':
            keys = sorted(counts)
            bars = ax.bar(range(len(keys)),[counts[k] for k in keys],color='black',width=.65)
            ax.set_xticks(range(len(keys)),keys)
            ax.bar_label(bars,fontsize=8,padding=3)
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            keys = sorted(counts,key=lambda k:(counts[k],k))
            bars = ax.barh(keys,[counts[k] for k in keys],color=[FUNCTION_COLORS.get(k,FUNCTION_COLORS['Unassigned']) for k in keys])
            ax.bar_label(bars,fontsize=8,padding=3)
            ax.set_xlim(0,max(counts.values(),default=1)*1.15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.spines[['top','right']].set_visible(False)
        if not counts: ax.text(.5,.5,'No network genes',transform=ax.transAxes,ha='center')
        for extension in ('png','svg'):
            path = args.out/f'{name}.{extension}'
            fig.savefig(path,dpi=args.dpi,facecolor='white')
            outputs.append(str(path))
        plt.close(fig)
    common = ('Node fill: broad COG functional bins; grey: unassigned function. Named nodes are larger; '
              'unnamed nodes are smaller. Labels occupy separate margins and equal names retain distinct '
              'clusters with numeric suffixes within each panel. Edge width describes representative absolute beta; '
              'line style describes coefficient-sign consistency. Graphical positions are diagram coordinates. '
              'Neighbour order is shortest-path distance in the selected gene network, not genomic distance. '
              'Gene edges summarize significant distal unitig associations, not gene-level tests, causality '
              'or experimentally demonstrated epistasis. Functional annotation does not change selected edges.')
    (args.out/'figure_captions.txt').write_text('\n\n'.join(captions+[common])+'\n')
    atomic_json(args.out/'paper_figures.json',dict(source=fingerprint_file(source),genes=len(nodes),
        gene_edges=len(edges),components=len(components),two_gene_components=dimers,isolates=isolates,
        displayed_component_minimum=args.min_component_nodes,focus=focuses,
        component_size_counts=dict(sizes),functional_counts=dict(function_counts),outputs=outputs,
        label_fontsize=args.label_fontsize,dpi=args.dpi,components_per_page=args.components_per_page,
        matplotlib=matplotlib.__version__,networkx=nx.__version__,
        refitted=False,qq_plot_generated=False,svg_text_editable=True))
    print(f'[paper] PNG/SVG figures and captions: {args.out}',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
