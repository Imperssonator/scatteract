import matplotlib
matplotlib.use('Agg') #To make sure plots are not being displayed during the generation.
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb

import os
import click
from glob import glob
import pandas as pd
from tqdm import tqdm

import numpy as np
np.random.seed(0) #For consistent data generation.
import scipy.stats as stats

from PIL import Image
import cv2
import copy


### SET UP PARAMETER SAMPLING #####
def discrete_sample(df, param):
    """
    Given a dataframe with column names corresponding
    to parameters and rows with discrete parameter options,
    return a uniformly random sampled parameter choice -
    probably a string.
    """
    
    return df[param].dropna().sample(1).iloc[0]


def continuous_sample(df, param):
    """
    Given a dataframe with index corresponding
    to parameter names and a column called "sampler"
    containing truncated normal sampler objects,
    return a sample from that parameter's distribution
    between its lower and upper bounds
    """
    
    return df.loc[param, 'sampler'].rvs(1)[0]


def trunc_norm_sampler(lower, upper, mu, n_stds):
    """
    Return a truncated normal distribution sampler object
    that will only return values between lower and upper
    with a normal pdf with a number of standard deviations
    between the midpoint and each edge equal to n_stds
    """
    
    if pd.isnull(mu):
        mu = np.mean([lower,upper])
    else:
        mu = float(mu)

    if pd.isnull(n_stds):
        n_stds = 1
    else:
        n_stds = float(n_stds)
    
    sigma = (upper-lower)/2 / n_stds
    
    X = stats.truncnorm(
                        (lower - mu) / sigma,
                        (upper - mu) / sigma,
                        loc=mu,
                        scale=sigma
                        )
    return X


def dist_sample(name, dfd, dfc):
    
    """
    'name' will either be a binary probability between 0 and 1
    or the name of a distribution in either dfc or dfd
    
    If it's a number, return True or False with p(True) = name
    If it's a string, find the row of dfc or column of dfd and sample it
    """
    
    try:
        
        thresh = float(name)
        return np.random.rand()<thresh

    except:
        
        if name in dfc.index.values:
            return continuous_sample(dfc, name)
        elif name in dfd.columns:
            return discrete_sample(dfd, name)
        else:
            print('No distribution named {}'.format(name))
            return None


def build_kw_dict(kwcsv):
    """
    A kwcsv file has two columns: param and dist
    param refers to a field of kwargs for a matplotlib function
    dist refers to the name of a distribution
    distributions are either rows of dfd or columns of dfc
    """
    
    df = pd.read_csv(kwcsv)
    kw_dict = {p:dist_sample(d, dfd, dfc) for p,d in zip(df['param'],df['dist'])}
    
    return kw_dict


### FIGURE GENERATION ###

def generate_figure(figwidth=5, figaspect=1.25, dpi=150, facecolor='w'):
    """
    Generate a basic Figure object given width, aspect ratio, dpi
    and facecolor
    """
    
    figsize = (figwidth, figwidth/figaspect)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    plt.minorticks_on()
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    
    return fig, ax


### DATA GENERATION ###

def power_data_gen(x_min=0, x_range=3, n_points=20,
                   poly_scale=1, poly_deg=1,
                   noise_std_prct=10):
    """ Given:
        x_min: minimum of x range (as in 10^x_min)
        x_range: ... range of x values (max-min) (as in 10^x_max)
        n_points: number of points in series
        
        Return Y = A*X^(B)+noise
        Where poly_scale=A, poly_deg=B, and noise
        is normally distributed at with sigma = Y*noise_std_prct/100
        
        If x_range<3 (<10^3), make linear spaced x points
        If >3, make log-spaced x points... if they're log spaced,
        then make a log plot!
        """
    log_cutoff = 3
    
    if x_range>log_cutoff:
        x_spacing='log'
        X = np.logspace(x_min,
                        x_min + x_range,
                        int(n_points))
    else:
        X = np.linspace(10**x_min,
                        10**x_min + 10**x_range,
                        int(n_points))
        x_spacing='linear'
    
    Y = poly_scale * X ** poly_deg
    
    y_spacing = 'linear' if max(Y)*min(Y)<0 or np.abs(np.log10(max(Y)/min(Y)))<log_cutoff else 'log'
    if y_spacing=='log' and np.any(Y<0):
        Y = np.abs(Y)
    
    Y_err = np.random.normal(loc=np.zeros(Y.shape), scale=np.abs(Y*noise_std_prct/100))
    
    return X, Y, Y_err, x_spacing, y_spacing


### FULL PLOT GENERATION ###

def generate_training_plot(data_folder, id_str, label_colors):
    
    """
    Given a folder and the ID# for a new random plot, generate it and stick
    it in the folder
    """
    
    ### GENERATE FIGURE ###
    fig_kwargs = build_kw_dict('plot_params/fig_properties.csv')
    fig, ax = generate_figure(**fig_kwargs)

    ### PLOT DATA ###
    data_kwargs = build_kw_dict('plot_params/data_gen.csv')
    marker_kwargs = build_kw_dict('plot_params/marker_styles.csv')
    X, Y, Ye, x_spacing, y_spacing = power_data_gen(**data_kwargs)
    ax.plot(X,Y+Ye,**marker_kwargs)
    ax.set_xscale(x_spacing, nonposx='clip')
    ax.set_yscale(y_spacing, nonposy='clip')

    ### ERROR BARS ###
    error_kwargs = build_kw_dict('plot_params/errorbar_styles.csv')
    error_kwargs['linestyle']='None'
    ax.errorbar(X, Y+Ye, yerr=Y*data_kwargs['noise_std_prct']/100, **error_kwargs)

    ### BOX AND GRID ###
    plt_bools = build_kw_dict('plot_params/plt_boolean.csv')
    for k,v in plt_bools.items():
        eval('plt.{}({})'.format(k,v))
    plt.grid(True)

    ### TICKS ###
    tick_param_kwargs = build_kw_dict('plot_params/tick_params_major.csv')
    ax.tick_params(which='major', **tick_param_kwargs)

    tick_param_minor_kwargs = build_kw_dict('plot_params/tick_params_minor.csv')
    ax.tick_params(which='minor', **tick_param_minor_kwargs)

    ### TICK LABELS ###
    tick_font = font_manager.FontProperties(**build_kw_dict('plot_params/font_properties.csv'))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(tick_font)

    plt.tight_layout()
    
    ### SAVE RAW AND LABELED IMAGES ###
    fig.savefig('{}/{}.png'.format(data_folder, id_str), facecolor=fig.get_facecolor(), edgecolor='none')
    label_img_array = generate_label_image(fig, ax, label_colors)
    label_img = Image.fromarray(label_img_array)
    label_img.save('{}/{}.png'.format(data_folder+'_labels', id_str))
    
    return fig, ax


def generate_label_image(fig, ax, label_colors):
    """
    This somehow turned out more complicated than plot generation...
    Given the Figure and Axes objects of a random plot,
    and label_colors {'plot element': [r, g, b] as uint8}
    Return label_image, an image (numpy array) where the pixels representing each plot component have been labeled according to the provided colors (label_colors) so it can be used as input to Semantic Segmentation Suite
    Also df_lc: dataframe of label colors that can be dumped to csv for the dataset
    """
    
    mask_dict = {}
    # probably need some defensive code to check the label_colors dict
    
    bg_color = np.array([int(c*255) for c in fig.get_facecolor()])[:3].astype(np.uint8)

    kids = ax.get_children()
    
    ### MARKERS ###
    visible = [1]
    for i in range(len(kids)):
        if i not in visible:
            kids[i].set_visible(False)
        else:
            kids[i].set_visible(True)
            kids[i].set_linestyle('None')
    
    fig.canvas.draw()
    class_img = np.array(fig.canvas.renderer._renderer)[:,:,:3]
    mask_dict['markers'] = ~np.all(np.isclose(class_img, bg_color, rtol=0.01), axis=-1)
    
#     ### ERROR BARS ###
#     visible = [0,3,4]
#     for i in range(len(kids)):
#         if i not in visible:
#             kids[i].set_visible(False)
#         else:
#             kids[i].set_visible(True)

#     fig.canvas.draw()
#     class_img = np.array(fig.canvas.renderer._renderer)[:,:,:3]
#     mask_dict['error_bars'] = ~np.all(np.isclose(class_img, bg_color, rtol=0.01), axis=-1)

    ### TICKS & LABELS ###
    
    for aa in ['x', 'y']:
        axis = eval('ax.{}axis'.format(aa))
        mlf = copy.copy(axis.get_major_formatter())
        
        # Make only the _axis visible
        [k.set_visible(False) for k in kids]
        axis.set_visible(True)
        
        # Make only the major ticks+grid visible
        [t.set_visible(False) for t in axis.get_minor_ticks()]
        axis.set_major_formatter(plt.NullFormatter())
        
        # Generate tick mask
        fig.canvas.draw_idle()
        class_img = np.array(fig.canvas.renderer._renderer)[:,:,:3]
        mask_dict[aa+'_ticks'] = ~np.all(np.isclose(class_img, bg_color, rtol=0.01), axis=-1)
        
        # Make only the tick labels visible
        axis.set_major_formatter(mlf)
        [[ch.set_visible(False) for ch in tick.get_children() if not hasattr(ch,'_text')] for tick in axis.get_major_ticks()]
        [g.set_visible(False) for g in axis.get_gridlines()]
        
        # Generate label mask
        fig.canvas.draw_idle()
        class_img = np.array(fig.canvas.renderer._renderer)[:,:,:3]
        cv2.imwrite('temp/label_test.png',class_img)
        mask_dict[aa+'_tick_labels'] = ~np.all(np.isclose(class_img, bg_color, rtol=0.01), axis=-1)
        
        # Reset visibilities
        [k.set_visible(True) for k in kids]
        [t.set_visible(True) for t in axis.get_major_ticks()]
        [t.set_visible(True) for t in axis.get_minor_ticks()]
        [g.set_visible(True) for g in axis.get_gridlines()]
    
    ### FINAL LABEL IMAGE ###
    
    label_image = np.zeros(class_img.shape).astype(np.uint8)
    for kk, mm in mask_dict.items():
        label_image = set_color_mask(label_image, mm, label_colors[kk])
    
    bg_mask = np.all(label_image==np.zeros(3).astype(np.uint8), axis=-1)
    label_image = set_color_mask(label_image, bg_mask, label_colors['background'])

    return label_image


def str2color(color_string):
    """ Convert color string to uint8 array representation """
    return (np.array(to_rgb(color_string))*255).astype(np.uint8)


def set_color_mask(A, M, c):
    """ Given image array A (h, w, 3) and mask M (h, w, 1),
        Apply color c (3,) to pixels in array A at locations M
        """
    for i in range(3):
        A_i = A[:,:,i]
        A_i[M]=c[i]
        A[:,:,i] = A_i
    return A


### DISCRETE PARAMETERS ###
dfd = pd.read_csv('plot_params/discrete.csv')

### CONTINUOUS PARAMETERS ###
dfc = pd.read_csv('plot_params/continuous.csv', index_col='param')
dfc['sampler'] = \
    dfc.apply(lambda row:
              trunc_norm_sampler(row['min'],
                                 row['max'],
                                 row['mean'],
                                 row['n_stds']),
              axis=1)


@click.command()
@click.argument('base_folder', type=click.Path())
@click.option('--num-train', '-n', type=int, default=1000)
@click.option('--num-val', '-v', type=int, default=400)
@click.option('--num-test', '-t', type=int, default=400)
def generate_dataset(base_folder, num_train=1000, num_val=400, num_test=400):
    
    os.makedirs(base_folder, exist_ok=True)

    ### SET LABEL PIXEL COLORS ###
    label_colors = {'markers': str2color('xkcd:blue'),
                    'x_ticks': str2color('xkcd:dark red'),
                    'x_tick_labels': str2color('xkcd:red'),
                    'y_ticks': str2color('xkcd:violet'),
                    'y_tick_labels': str2color('xkcd:light purple'),
                    'error_bars': str2color('xkcd:dark grey'),
                    'background': str2color('xkcd:eggshell')}

    df_lc = pd.DataFrame.from_dict(label_colors).transpose().reset_index()
    df_lc.columns=['name','r','g','b']
    df_lc.to_csv(os.path.join(base_folder,'class_dict.csv'), index=False)

    ### GENERATE PLOT IMAGES AND CLASS LABEL IMAGES ###
    for dataset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_folder, dataset), exist_ok=True)
        os.makedirs(os.path.join(base_folder, dataset+'_labels'), exist_ok=True)

    for dataset in ['train', 'val', 'test']:
        print('Generating ', dataset)
        for i in tqdm(range(eval('num_'+dataset))):
            data_folder = os.path.join(base_folder, dataset)
            fig, ax = generate_training_plot(data_folder,
                                             str(i).zfill(6),
                                             label_colors)
            plt.close(fig)

    return


if __name__ == '__main__':
    generate_dataset()
