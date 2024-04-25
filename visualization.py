import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

def plot_before_after_df(df_after, df_diff, colName: str = 'sal', markersize= 1.0, xLable= 'Date', yLable= 'Salinity', title='Salinity Before/After', size= (18, 10)):
    fig, ax = plt.subplots()

    ax.plot(df_after.index, df_after[colName])
    ax.plot(df_diff.index, df_diff[colName], marker='o', linestyle='None', color='red', markersize= markersize)

    ax.set_xlabel(xLable)
    ax.set_ylabel(yLable)
    ax.set_title(title)

    fig.set_size_inches(size[0], size[1], forward=True)
    return fig, ax

def hist(before, after, colName: str = 'sal'):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=before[colName], name='BEFORE', marker_color='red'))
    fig.add_trace(go.Histogram(x=after[colName], name='AFTER', marker_color='blue'))
    return fig


def pprint(results):
    for idx, result in enumerate(results):
        print(f'Model: {idx}')
        for k in result['min'].keys():
            print(f'{k}:\nmax: {result["max"][k]}\nmean: {result["mean"][k]}\nmin: {result["min"][k]}\n')


def prep_result_for_excel(result, name):
        line = ''
        for mmm in result.values():
            for v in mmm.values():
                line = f'{line}, {v}'
        return f'{name}{line}'