import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from ydata_profiling import ProfileReport
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import ipympl
import warnings
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import datetime as dt
import warnings
from sklearn.preprocessing import OneHotEncoder, StandardScaler , MinMaxScaler , PowerTransformer
from sklearn.compose import  ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from cycler import cycler
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold,cross_validate
from imblearn.pipeline import Pipeline #PipeLine do Imlearn for Imbalanced Datasets!
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.under_sampling import RandomUnderSampler
#Funtions

#limits
def limits(columns):
    Q1 = columns.quantile(0.25)
    Q3 = columns.quantile(0.75)
    amplitude = Q3 -Q1
    inferior_limit = Q1 - 1.5 * amplitude
    superior_limit = Q3 + 1.5 *amplitude
    return inferior_limit,superior_limit


#boxplot function
def boxplot(columns):
    fig,(ax1,ax2) = plt.subplots(1,2)
    fig.set_size_inches(15,5)
    sns.boxplot(x=columns,ax=ax1)
    ax2.set_xlim(limits(columns))
    sns.boxplot(x=columns,ax=ax2)
    plt.show()
    return

def inspect_outliers(DataFrame, columns, whisker_width=1.5):
    Q1 = DataFrame[columns].quantile(0.25)
    Q3 = DataFrame[columns].quantile(0.75)
    iqr = Q3 - Q1
    inferior_limit = Q1 - whisker_width * iqr
    superior_limit = Q3 + whisker_width * iqr
    return DataFrame[
        (DataFrame[columns] < inferior_limit) | (DataFrame[columns] > superior_limit)
    ]


def delete_outliers(data,column_name):
    total_lines = data.shape[0]
    inferior_limits,superior_limits = limits(data[column_name])
    data = data.loc[(data[column_name]>= inferior_limits) & (data[column_name] <=superior_limits),:]
    removed_lines = total_lines -  data.shape[0]
    return data, removed_lines


def pairplot(DataFrame, columns,diag_kind=None, hue_colum=None,alpha=0.5,corner=False):
    analysis = columns.copy() + [hue_colum]
    sns.pairplot(
        DataFrame[analysis],
        diag_kind=diag_kind,
        hue=hue_colum,
        plot_kws=dict(alpha=alpha),
        corner=corner
         
    )
   

    plt.show()
    return 




def graphic_plot(column):
    plt.figure(figsize=(15,5))
    ax = sns.barplot(x=column.value_counts().index,y=column.value_counts())
    ax.set_xlim(limits(column))
    plt.show()
    return





def elbow_silhouette_graphic(X,random_state=42,intervalo_k=(2,11)):
    elbow = {}
    silhouette = []
    k_range = range(*intervalo_k)


    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), tight_layout=True)

    for i in k_range:
        kmeans = KMeans(n_clusters=i, random_state=random_state, n_init=10)
        kmeans.fit(X)
        elbow[i] = kmeans.inertia_
        labels = kmeans.labels_
        silhouette.append(silhouette_score(X, labels))


    sns.lineplot(x=list(elbow.keys()), y=list(elbow.values()), ax=axs[0])
    axs[0].set_xlabel('K')
    axs[0].set_ylabel('Inertia')
    axs[0].set_title('Elbow Method')


    sns.lineplot(x=list(k_range), y=silhouette, ax=axs[1])
    axs[1].set_xlabel('K')
    axs[1].set_ylabel('Silhouette Score')
    axs[1].set_title('Silhouette Score Method')
    
    plt.show()
    return

def view_clusters(
    dataframe,
    colunas,
    quantidade_cores,
    centroids,
    mostrar_centroids=True,
    mostrar_pontos=False,
    coluna_clusters=None,
):
    """
    Generate 3D graph with clusters

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe with the data.
    colunas: List[str]
        List with the name of the columns (strings) to be used.
    quantidade_cores: int
        Number of colors for the chart.
    centroids: np.ndarray
        Array with centroids.
    mostrar_centroids : bool, optional
        Whether the graph will show the centroids or not, by default True
    mostrar_pontos: bool, optional
        Whether the graph will show the points or not, by default False
    coluna_clusters : List[int], optional
        Column with cluster numbers to color the points (if show_points is True), by default None
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    cores = plt.cm.tab10.colors[:quantidade_cores]
    cores = ListedColormap(cores)

    x = dataframe[colunas[0]]
    y = dataframe[colunas[1]]
    z = dataframe[colunas[2]]

    for i, centroid in enumerate(centroids):
        if mostrar_centroids:
            ax.scatter(*centroid, s=500,alpha=1)
            ax.text(
                *centroid,
                f"{i}",
                fontsize=20,
                horizontalalignment="center",
                verticalalignment="center",
            )

    if mostrar_pontos:
        s = ax.scatter(x, y, z, c=coluna_clusters, cmap=cores)
        ax.legend(*s.legend_elements(), bbox_to_anchor=(1.3, 1))

    ax.set_xlabel(colunas[0])
    ax.set_ylabel(colunas[1])
    ax.set_zlabel(colunas[2])
    ax.set_title("Clusters")

    plt.show()
    return







from matplotlib.ticker import PercentFormatter
def plot_columns_percent_by_cluster(DataFrame, columns, nrows_ncols=(2,3), figsize=(15,10), sharey=True, column_cluster='Cluster'):
    fig, axs = plt.subplots(nrows=nrows_ncols[0], ncols=nrows_ncols[1], figsize=figsize, sharey=sharey)
    for ax, col in zip(axs.flatten(), columns):
        h = sns.histplot(x=column_cluster, data=DataFrame, ax=ax, hue=col, multiple='fill', stat='percent', discrete=True, shrink=0.8)
        ax.set_title(f'Histplot of {col}')
        n_clusters = DataFrame[column_cluster].nunique()
        h.set_xticks(range(n_clusters))
        h.set_yticklabels('')
        h.yaxis.set_major_formatter(PercentFormatter(1))
        h.set_ylabel('')
        h.tick_params(axis='both', which='both', length=0)
        for bars in h.containers:
            h.bar_label(bars, label_type='center', labels=[f'{b.get_height():.1%}' for b in bars], color='white', weight='bold', fontsize=11, padding=2)
        for bar in h.patches:
            bar.set_linewidth(0)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  
    plt.show()
    return



def plot_columns_percent_hue_cluster(
    dataframe,
    columns,
    rows_cols=(2, 3),
    figsize=(15, 8),
    column_cluster="",
    palette="tab10",
):
    """Function to generate bar plots with the percentage of each value with cluster as hue.

Parameters
----------
dataframe : pandas.DataFrame
    DataFrame with the data.
columns : List[str]
    List of column names (strings) to be used.
rows_cols : tuple, optional
    Tuple with the number of rows and columns for the axis grid, by default (2, 3).
figsize : tuple, optional
    Tuple with the width and height of the figure, by default (15, 8).
column_cluster : str, optional
    Name of the column with the cluster numbers, by default "cluster".
palette : str, optional
    Palette to be used, by default "tab10".
    """
    fig, axs = plt.subplots(
        nrows=rows_cols[0], ncols=rows_cols[1], figsize=figsize, sharey=True
    )

    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)

    for ax, col in zip(axs.flatten(), columns):
        h = sns.histplot(
            x=col,
            hue=column_cluster,
            data=dataframe,
            ax=ax,
            multiple="fill",
            stat="percent",
            discrete=True,
            shrink=0.8,
            palette=palette,
        )

        if dataframe[col].dtype != "object":
            h.set_xticks(range(dataframe[col].nunique()))

        h.yaxis.set_major_formatter(PercentFormatter(1))
        h.set_ylabel("")
        h.tick_params(axis="both", which="both", length=0)

        for bars in h.containers:
            h.bar_label(
                bars,
                label_type="center",
                labels=[f"{b.get_height():.1%}" for b in bars],
                color="white",
                weight="bold",
                fontsize=11,
            )

        for bar in h.patches:
            bar.set_linewidth(0)

        legend = h.get_legend()
        legend.remove()

    labels = [text.get_text() for text in legend.get_texts()]

    fig.legend(
        handles=legend.legend_handles,
        labels=labels,
        loc="upper center",
        ncols=dataframe[column_cluster].nunique(),
        title="Clusters",
    )

    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.show()
    return



def plot_clusters_2D(
    dataframe,
    columns,
    n_colors,
    centroids,
    show_centroids=True,
    show_points=False,
    column_clusters=None,
):
    """Generate 2D plot with clusters.

Parameters
----------
dataframe : pandas.DataFrame
    DataFrame with the data.
columns : List[str]
    List of column names (strings) to be used.
n_colors : int
    Number of colors for the plot.
centroids : np.ndarray
    Array with the centroids.
show_centroids : bool, optional
    If the plot will show the centroids or not, by default True.
show_points : bool, optional
    If the plot will show the points or not, by default False.
column_clusters : List[int], optional
    Column with the cluster numbers to color the points
    (if show_points is True), by default None.

    """

    fig, ax = plt.subplots()

    cores = plt.cm.tab10.colors[:n_colors]
    cores = ListedColormap(cores)

    x = dataframe[columns[0]]
    y = dataframe[columns[1]]

    if show_centroids and centroids is not None:
        for i, centroid in enumerate(centroids):
            ax.scatter(centroid[0], centroid[1], s=500, alpha=0.5, color='black', marker='X')
            ax.text(
                centroid[0],
                centroid[1],
                f"{i}",
                fontsize=20,
                horizontalalignment="center",
                verticalalignment="center",
                color='black'
            )

    if show_points and column_clusters is not None:
        scatter = ax.scatter(x, y, c=column_clusters, cmap=cores)
        legend = ax.legend(*scatter.legend_elements(), bbox_to_anchor=(1.3, 1))
        ax.add_artist(legend)

    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_title("Clusters")

    plt.show()
