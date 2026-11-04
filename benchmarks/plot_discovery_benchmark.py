"""Plot saved production significance benchmark metrics as editable PNG/SVG."""
import argparse
import csv
from pathlib import Path


def plot(out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    with (out/'comparison.tsv').open() as stream:
        rows = list(csv.DictReader(stream,delimiter='\t'))
    if not rows:
        raise ValueError('comparison.tsv has no method rows')
    figures = out/'figures'
    figures.mkdir(exist_ok=True)
    names = {'exact':'Exact', 'bounds':'Adaptive bounds', 'hierarchy_bounds':'Hierarchy + bounds',
             'hierarchy_pc10':'PC40 + hierarchy + bounds'}
    labels = [names.get(r['method'],r['method']) for r in rows]
    records = [('recovery','recall','Significant-pair recovery (%)',100),
               ('runtime','seconds','Invocation wall time (s)',1),
               ('queries','queries','Queried pattern pairs',1)]
    for name, key, ylabel, factor in records:
        values = [float(r[key])*factor if r[key] not in ('NA','') else np.nan for r in rows]
        fig,ax = plt.subplots(figsize=(6.8,4.3),layout='constrained')
        ax.bar(range(len(rows)),values,color='black',width=.55)
        ax.set_xticks(range(len(rows)),labels,rotation=20,ha='right')
        ax.set_ylabel(ylabel)
        ax.spines[['top','right']].set_visible(False)
        if key == 'recall':
            ax.set_ylim(0,105)
        for x,value in enumerate(values):
            if np.isfinite(value):
                ax.annotate(f'{value:,.2f}' if key=='seconds' else f'{value:,.0f}',
                            (x,value),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
            else:
                ax.text(x,1,'NA',ha='center')
        for extension in ('png','svg'):
            fig.savefig(figures/f'significance_{name}.{extension}',dpi=300)
        plt.close(fig)
    (figures/'significance_figure_notes.txt').write_text(
        'Recovery uses final Bonferroni-significant exact-reference pairs in the frozen panel. '
        'NA means no reference discoveries, not zero recovery. Query counts cover evaluated '
        'pairs and differ from coarse-candidate counts. Wall times include preparation/fitting '
        'within each invocation; later arms reuse compatible nulls, so these are not independent '
        'cold-start kernel timings. The random unique-pattern panel was selected without phenotype '
        'or association scores. Small-panel recovery does not establish full-genome completeness.\n')
    print(f'PNG/SVG figures: {figures}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    plot(args.out.resolve())


if __name__ == '__main__':
    main()
