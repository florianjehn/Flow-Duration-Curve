# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 10:09:13 2018

@author: Florian Ulrich Jehn
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def flow_duration_curve(x, comparison=None, axis=0, ax=None, plot=True, 
                        log=True, percentiles=(5, 95), decimal_places=1,
                        fdc_kwargs=None, fdc_range_kwargs=None, 
                        fdc_comparison_kwargs=None):
    """
    Calculates and plots a flow duration curve from x. 
    
    All observations/simulations are ordered and the empirical probability is
    calculated. This is then plotted as a flow duration curve. 
    
    When x has more than one dimension along axis, a range flow duration curve 
    is plotted. This means that for every probability a min and max flow is 
    determined. This is then plotted as a fill between. 
    
    Additionally a comparison can be given to the function, which is plotted in
    the same ax.
    
    :param x: numpy array or pandas dataframe, discharge of measurements or 
    simulations
    :param comparison: numpy array or pandas dataframe of discharge that should
    also be plotted in the same ax
    :param axis: int, axis along which x is iterated through
    :param ax: matplotlib subplot object, if not None, will plot in that 
    instance
    :param plot: bool, if False function will not show the plot, but simply
    return the ax object
    :param log: bool, if True plot on loglog axis
    :param percentiles: tuple of int, percentiles that should be used for 
    drawing a range flow duration curve
    :param decimal_places: defines how finely grained the range flow duration curve
    is calculated and drawn. A low values makes it more finely grained.
    A value which is too low might create artefacts.
    :param fdc_kwargs: dict, matplotlib keywords for the normal fdc
    :param fdc_range_kwargs: dict, matplotlib keywords for the range fdc
    :param fdc_comparison_kwargs: dict, matplotlib keywords for the comparison 
    fdc
    
    return: subplot object with the flow duration curve in it
    """
    # Convert x to an pandas dataframe, for easier handling
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
        
    # Get the dataframe in the right dimensions, if it is not in the expected
    if axis != 0:
        x = x.transpose()
        
    # Convert comparison to a dataframe as well
    if comparison is not None and not isinstance(comparison, pd.DataFrame):
        comparison = pd.DataFrame(comparison)
        # And transpose it is neccesary
        if axis != 0:
            comparison = comparison.transpose()
    
    # Create an ax is neccesary
    if ax is None:
        fig, ax = plt.subplots(1,1)
        
    # Make the y scale logarithmic if needed
    if log:
        ax.set_yscale("log")
        
    # Determine if it is a range flow curve or a normal one by checking the 
    # dimensions of the dataframe
    # If it is one, make a single fdc
    if x.shape[1] == 1:
        plot_single_flow_duration_curve(ax, x[0], fdc_kwargs)   
        
    # Make a range flow duration curve
    else:
        plot_range_flow_duration_curve(ax, x, percentiles, decimal_places,
                                       fdc_range_kwargs)
        
    # Add a comparison to the plot if is present
    if comparison is not None:
        ax = plot_single_flow_duration_curve(ax, comparison[0], 
                                             fdc_comparison_kwargs)    

    # Name the x-axis
    ax.set_xlabel("Exceedence [%]")
    
    # show if requested
    if plot:
        plt.show()
        
    return ax


def plot_single_flow_duration_curve(ax, timeseries, kwargs):
    """
    Plots a single fdc into an ax.
    
    :param ax: matplotlib subplot object
    :param timeseries: list like iterable
    :param kwargs: dict, keyword arguments for matplotlib

    return: subplot object with a flow duration curve drawn into it
    """
    # Get the probability
    ex_prob = calc_probabilities(timeseries)
    # Plot the curve, check for empty kwargs
    if kwargs is not None:
        ax.plot(ex_prob, sorted(timeseries, reverse=True), **kwargs)
    else:
        ax.plot(ex_prob, sorted(timeseries, reverse=True))
    return ax


def plot_range_flow_duration_curve(ax, x, percentiles, decimal_places, kwargs):
    """
    Plots a single range fdc into an ax.
    
    :param ax: matplotlib subplot object
    :param x: dataframe of several timeseries
    :param decimal_places: defines how finely grained the range flow duration curve
    is calculated and drawn. A low values makes it more finely grained.
    A value which is too low might create artefacts.
    :param kwargs: dict, keyword arguments for matplotlib
    
    return: subplot object with a range flow duration curve drawn into it
    """
    # Get the high and low values for every probability
    ex_prob_df = calc_all_probabilities(x, decimal_places)
    ex_prob, low_percentile, high_percentile = find_percentiles(ex_prob_df,
                                                               percentiles)  
    # Plot it, check for empty kwargs
    if kwargs is not None:
        ax.fill_between(ex_prob, low_percentile, high_percentile, **kwargs)
    else:
        ax.fill_between(ex_prob, low_percentile, high_percentile)
    return ax
    

def calc_probabilities(timeseries):
    """
    Calculates the rank and the exedence probability for a given timeseries
    
    :param timeseries: one dimensional time series
    
    return: sorted exceedence probabilities for timeseries
    """    
    # Sort the timeseries, so the probabilies are ordered right
    timeseries = sorted(timeseries, reverse=True)
    
    # Calculate the excedence probability
    ex_prob = []
    len_timeseries = len(timeseries)
    for rank in range(len_timeseries):
        ex_prob_value = 100 * ((rank + 1) / (len_timeseries + 1))
        ex_prob.append(ex_prob_value)
    return ex_prob


def calc_all_probabilities(x, decimal_places):
    """
    Calculates the high and low values for the exeedence probability for a 
    dataframes consisting of several timeseries. 
    
    :param x: dataframe of several timeseries
    :param decimal_places: defines how finely grained the range flow duration curve
    is calculated and drawn. A low values makes it more finely grained.
    A value which is too low might create artefacts.

    return: dict, the keys are the rounded (to stepsize) probabilities, the 
    value is all values that where assigned to that probability. 
    """
    # Create the main dictionary
    ex_prob_dict = {}
    for i in np.arange(0, 100 + 10**-decimal_places, 10**-decimal_places):
        # Round i, as floats do sometimes not have an exact value
        i = round(i, decimal_places)
        # Add the key to the dict
        ex_prob_dict[i] = []
    
    # Calculate the exedeence probability for a all timeseries
    # Create a list, where all produced temporary dicts are saved
    temp_list = []
    for column in x:
        timeseries = x[column]
        timeseries_sorted = sorted(timeseries, reverse=True)
        ex_prob_timeseries = calc_probabilities(timeseries)
        
        # Create a dictionary where the probabilities are the keys and the
        # corresponding value of the timeseries is the value
        timeseries_dict = {}
        for i in range(len(ex_prob_timeseries)):
            rounded_ex_prob = round(ex_prob_timeseries[i], decimal_places)
            timeseries_dict[rounded_ex_prob] = timeseries_sorted[i]
            if rounded_ex_prob > 100:
                print(rounded_ex_prob)
        # Sometimes the values 0.0 and 100.0 do not get values. Fill them up
        # With the ones before
        if 0.0 not in timeseries_dict.keys():
            timeseries_dict[0.0] = timeseries_dict[10**-decimal_places]
        if 100.0 not in timeseries_dict.keys():
            timeseries_dict[100.0] = timeseries_dict[100-10**-decimal_places]
        temp_list.append(timeseries_dict)
        
    # Now iterate through all just created dictionaries. Round the keys to 
    # the desired accurazy
    for timeseries_dict in temp_list:
        for ex_prob in timeseries_dict.keys():
            rounded_ex_prob = round(ex_prob, decimal_places)
            # Now add the entry to the main dict, so every value of the
            # timeseries is matched to the corresponding probability over all
            # probabilites
            ex_prob_dict[rounded_ex_prob].append(timeseries_dict[ex_prob])
    
    return ex_prob_dict


def find_percentiles(ex_prob_dict, percentiles):
    """
    Finds the defined percentiles in a dict of exceedence probabilities.
    
    :param ex_prob_dict: dict of exceedence probabilities for several
    of timeseries
    :param percentiles: tuple, with int value (low_percentile, high_percentile)
    
    return: probabilities and correspondnig high and low percentiles of the 
    values for all exceedence probabilities
    """
    # Find the high and low values
    percentile_dict = {}
    for key in sorted(ex_prob_dict.keys()):
        low = np.percentile(ex_prob_dict[key], percentiles[0], 0)
        high = np.percentile(ex_prob_dict[key], percentiles[1], 0)   
        percentile_dict[key] = [low, high]
    
    # Get them in a plotable form
    low_percentile = []
    high_percentile = []
    ex_probs = []
    for ex_prob, (low, high) in percentile_dict.items():
        low_percentile.append(low)
        high_percentile.append(high)
        ex_probs.append(ex_prob)
    
    return ex_probs, low_percentile, high_percentile     


if __name__ == "__main__":
    # Create test data
    np_array_one_dim = np.random.uniform(1,20,[1,1000])
    np_array_100_dim = np.random.uniform(1,20,[100,1000])
    df_one_dim = pd.DataFrame(np.random.uniform(1,20,[1,1000]))
    df_100_dim = pd.DataFrame(np.random.normal(1,20,[100,1000]))
    df_100_dim_transposed = pd.DataFrame(np.random.beta(
                                                1,20,[100,1000])).transpose()
    
    
    # Call the function with all different arguments
    fig, subplots = plt.subplots(nrows=2, ncols=3)
    ax1 = flow_duration_curve(np_array_one_dim, ax=subplots[0,0], plot=False,
                              axis=1, fdc_kwargs={"linewidth":0.5})
    ax1.set_title("np array one dim")
    
    ax2 = flow_duration_curve(np_array_100_dim, ax=subplots[0,1], plot=False,
                              axis=1)
    ax2.set_title("np array 100 dim")
    
    ax3 = flow_duration_curve(df_one_dim, ax=subplots[0,2], plot=False, axis=1,
                              log=False, fdc_kwargs={"linewidth":0.5})
    ax3.set_title("\ndf one dim\nno log")
    
    ax4 = flow_duration_curve(df_100_dim, ax=subplots[1,0], plot=False, axis=1)
    ax4.set_title("df 100 dim")
    
    ax5 = flow_duration_curve(df_100_dim_transposed, ax=subplots[1,1], 
                              plot=False)
    ax5.set_title("df 100 dim transposed")
    
    ax6 = flow_duration_curve(df_100_dim, ax=subplots[1,2], plot=False,
                              comparison=np_array_one_dim, axis=1, 
                              fdc_comparison_kwargs={"color":"black", 
                                                     "label":"comparison",
                                                     "linewidth":0.5},
                              fdc_range_kwargs={"label":"range_fdc"})
    ax6.set_title("df 100 dim\n with comparison\nand kwargs")
    ax6.legend()
    
    plt.suptitle("The different shapes are caused by different distributions")
    
    # Show the beauty
    fig.tight_layout()
    plt.show()
