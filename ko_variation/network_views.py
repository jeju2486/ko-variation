"""Display-only community summaries and sparse ordered adjacency views."""
from __future__ import annotations

from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def read_layout(path):
    text=Path(path).read_text(encoding="utf-8")
    match=re.search(r'<script\b[^>]*\bid=[\'"]data[\'"][^>]*>(.*?)</script>',text,re.S)
    data=json.loads(match[1] if match else text)
    ids=[n["gene"] for n in data["nodes"]]
    if len(ids)!=len(set(ids)): raise ValueError("Layout contains duplicate gene IDs")
    known=set(ids)
    if any(e["gene_a"] not in known or e["gene_b"] not in known for e in data["edges"]):
        raise ValueError("Layout edge endpoint has no matching node")
    return data


def add_views(payload):
    """Group unsigned retained connectivity; never filter genes or associations."""
    import networkx as nx
    from .network_display import _positions
    data={**payload,"nodes":[dict(n) for n in payload["nodes"]],
          "edges":[dict(e) for e in payload["edges"]],"meta":dict(payload["meta"])}
    graph=nx.Graph()
    graph.add_nodes_from(sorted(n["gene"] for n in data["nodes"]))
    graph.add_edges_from((e["gene_a"],e["gene_b"]) for e in data["edges"] if e["gene_a"]!=e["gene_b"])
    seed=data["meta"].get("seed",42)
    partitions=nx.community.louvain_communities(graph,weight=None,seed=seed) if graph.number_of_edges() else [{g} for g in graph]
    # Louvain can return disconnected groups; split these explicitly. This is
    # not Leiden and does not assert functional coherence or statistical support.
    partitions=[part for group in partitions for part in nx.connected_components(graph.subgraph(group))]
    partitions.sort(key=lambda members:(-len(members),sorted(members)))
    lookup={g:f"C{i+1}" for i,members in enumerate(partitions) for g in members}
    by_id={n["gene"]:n for n in data["nodes"]}
    summaries=[]
    for i,members in enumerate(partitions):
        members=sorted(members,key=lambda g:(str(by_id[g].get("display_label",g)).casefold(),g))
        functions=Counter(by_id[g].get("function_category","Unassigned") for g in members)
        summaries.append(dict(id=f"C{i+1}",members=members,n_genes=len(members),
            n_internal_edges=0,n_external_edges=0,functions=dict(functions),
            radius=float(19+min(25,3*np.sqrt(len(members))))))
        for g in members: by_id[g]["community"]=f"C{i+1}"
    by_group={c["id"]:c for c in summaries}
    between=defaultdict(list)
    for edge in data["edges"]:
        a,b=lookup[edge["gene_a"]],lookup[edge["gene_b"]]
        if a==b: by_group[a]["n_internal_edges"]+=1
        else:
            between[tuple(sorted((a,b)))].append(edge)
            by_group[a]["n_external_edges"]+=1; by_group[b]["n_external_edges"]+=1
    community_edges=[]
    for (a,b),edges in sorted(between.items()):
        directions=Counter(e.get("adjusted_direction","unknown") for e in edges)
        direction=next(iter(directions)) if len(directions)==1 else "mixed"
        community_edges.append(dict(gene_a=a,gene_b=b,n_gene_edges=len(edges),
            link_density=len(edges)/(by_group[a]["n_genes"]*by_group[b]["n_genes"]),
            directions=dict(directions),adjusted_direction=direction,target_link_distance=150.))
    frame=pd.DataFrame(community_edges,columns=["gene_a","gene_b","target_link_distance"])
    points,_,_,width,height=_positions(pd.DataFrame({"gene":list(by_group)}),frame,seed,
                                    {c["id"]:c["radius"] for c in summaries},35)
    for c in summaries: c["x"],c["y"]=points[c["id"]]
    data.update(communities=summaries,community_edges=community_edges,
                matrix_order=[g for c in summaries for g in c["members"]])
    data["meta"].update(view_schema=2,default_view="overview",community_width=width,community_height=height,
        community_method="NetworkX Louvain, unweighted unsigned connectivity, resolution=1; disconnected groups split",
        community_interpretation="Descriptive connectivity communities, not validated functional modules",
        community_edge_encoding="Distinct retained gene-edge count; no pooled P-value or beta",
        matrix_blank="No retained map edge; untested and nonsignificant pairs are not distinguished")
    return data


def write_views(payload,prefix,dpi=300):
    """Render an existing payload without annotation, inference or refitting."""
    data=add_views(payload)
    prefix=Path(prefix); prefix.parent.mkdir(parents=True,exist_ok=True)
    paths={key:Path(str(prefix)+suffix) for key,suffix in [
        ("html",".html"),("layout",".layout.json"),("nodes",".nodes.tsv"),("edges",".edges.tsv"),
        ("communities",".communities.tsv"),("community_edges",".community_edges.tsv"),
        ("png",".png"),("svg",".svg"),("matrix_png",".matrix.png"),("matrix_svg",".matrix.svg")]}
    serialized=json.dumps(data,allow_nan=False)
    paths["layout"].write_text(serialized,encoding="utf-8")
    package=Path(__file__).parent
    document=(package/"templates/network_d3.html").read_text(encoding="utf-8")
    d3=(package/"vendor/d3.v7.9.0.min.js").read_text(encoding="utf-8").replace("</script","<\\/script")
    safe=serialized.replace("<","\\u003c").replace(">","\\u003e").replace("&","\\u0026")
    document=document.replace("__TITLE__",html.escape(data["meta"].get("title") or "Gene covariation map"))
    paths["html"].write_text(document.replace("__DATA__",safe).replace("__D3__",d3),encoding="utf-8")
    pd.DataFrame(data["nodes"]).to_csv(paths["nodes"],sep="\t",index=False)
    pd.DataFrame(data["edges"]).to_csv(paths["edges"],sep="\t",index=False)
    rows=[{**c,"members":";".join(c["members"]),"functions":json.dumps(c["functions"])} for c in data["communities"]]
    pd.DataFrame(rows,columns=["id","members","n_genes","n_internal_edges","n_external_edges","functions","radius","x","y"]).to_csv(paths["communities"],sep="\t",index=False)
    pd.DataFrame(data["community_edges"],columns=["gene_a","gene_b","n_gene_edges","link_density","directions","adjusted_direction"]).to_csv(paths["community_edges"],sep="\t",index=False)
    static_views(data,paths,dpi)
    return paths


def static_views(data,paths,dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Wedge
    from matplotlib.collections import PolyCollection
    from .network_display import COLORS
    from .functional_annotation import FUNCTION_COLORS
    communities={c["id"]:c for c in data["communities"]}
    width,height=data["meta"]["community_width"],data["meta"]["community_height"]
    fig,ax=plt.subplots(figsize=(10,max(5,min(9,10*height/width))))
    for e in data["community_edges"]:
        a,b=communities[e["gene_a"]],communities[e["gene_b"]]
        ax.plot([a["x"],b["x"]],[a["y"],b["y"]],color=COLORS.get(e["adjusted_direction"],".6"),
                lw=min(4,1+np.log1p(e["n_gene_edges"])),alpha=.5,zorder=1)
    for c in communities.values():
        center=(c["x"],c["y"]); radius=c["radius"]
        angle=0.
        for category,count in sorted(c["functions"].items()):
            end=angle+360*count/c["n_genes"]
            ax.add_patch(Wedge(center,radius,angle,end,width=radius*.3,
                              facecolor=FUNCTION_COLORS.get(category,"#AAB2BC"),edgecolor="white",lw=.4,zorder=2))
            angle=end
        ax.add_patch(Circle(center,radius*.69,facecolor="white",edgecolor="none",zorder=2))
        ax.text(c["x"],c["y"]-3,c["id"],fontsize=9,ha="center",va="center",zorder=3)
        ax.text(c["x"],c["y"]+8,f"{c['n_genes']} genes",fontsize=6,ha="center",va="center",zorder=3)
    if not communities: ax.text(width/2,height/2,"No retained gene edges",ha="center")
    ax.set(xlim=(0,width),ylim=(height,0),aspect="equal"); ax.axis("off")
    fig.text(.5,.025,"Connectivity communities; ring colours show broad function. Link width counts retained gene edges.\nGroups are descriptive; positions are not genomic coordinates.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1))
    with matplotlib.rc_context({"svg.fonttype":"none"}):
        fig.savefig(paths["png"],dpi=dpi); fig.savefig(paths["svg"])
    plt.close(fig)
    order=data["matrix_order"]; index={g:i for i,g in enumerate(order)}
    fig,ax=plt.subplots(figsize=(10,10))
    polygons,colors=[],[]
    for edge in data["edges"]:
        a,b=index[edge["gene_a"]],index[edge["gene_b"]]
        for x,y in {(a,b),(b,a)}:
            polygons.append([(x,y),(x+1,y),(x+1,y+1),(x,y+1)])
            colors.append(COLORS.get(edge.get("adjusted_direction"),"#AAB2BC"))
    if polygons: ax.add_collection(PolyCollection(polygons,facecolors=colors,edgecolors="none"))
    n=max(len(order),1); ticks=[]; labels=[]; start=0
    for c in data["communities"]:
        if c["n_genes"]>=max(2,len(order)*.025): ticks.append(start+c["n_genes"]/2); labels.append(c["id"])
        if start: ax.axhline(start,color=".8",lw=.4); ax.axvline(start,color=".8",lw=.4)
        start+=c["n_genes"]
    ax.set(xlim=(0,n),ylim=(n,0),aspect="equal",title="Retained gene edges, ordered by connectivity community")
    ax.set_xticks(ticks,labels,fontsize=8); ax.set_yticks(ticks,labels,fontsize=8)
    fig.text(.5,.025,"Red: positive; blue: negative; purple: mixed; grey: neutral/unestimated.\nWhite: no retained map edge; it does not establish absence of association.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.07,1,1))
    with matplotlib.rc_context({"svg.fonttype":"none"}):
        fig.savefig(paths["matrix_png"],dpi=dpi); fig.savefig(paths["matrix_svg"])
    plt.close(fig)
