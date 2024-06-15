""" Plotting functions

"""

__author__ = 'Atoosa Parsa'
__copyright__ = 'Copyright 2024, Atoosa Parsa'
__credits__ = 'Atoosa Parsa'
__license__ = 'MIT License'
__version__ = '2.0.0'
__maintainer__ = 'Atoosa Parsa'
__email__ = 'atoosa.parsa@gmail.com'
__status__ = "Dev"


import warnings
warnings.filterwarnings("ignore", message=".*No contour levels were found.*")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, CenteredNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse

mpl.use('Agg',)
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 600

#fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
def colorFader(c1, c2, mix=0):
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
    
# plot loss vs epoch
def plot_loss(loss_iter, plotdir):
    fig = plt.figure()
    plt.plot(loss_iter, 'o-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(color='grey', linestyle=':', linewidth=0.75)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir+'loss.png')
    plt.close(fig)

# bar plots for the output at each epoch
# u is the output (normalized power), p is the desired output
def plot_output(u, p, epoch, plotdir, name=''):
    fig = plt.figure()
    plt.bar(range(1, 1+u.size()[0]), u.detach().cpu().numpy(), color='k')
    plt.xlabel("output port")
    plt.ylabel("output")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir+f'output_epoch{epoch}_X{p.detach().cpu().numpy()}_{name}.png')
    plt.close(fig)

# plot the displacements
def plot_waves(epoch, X, labels='', plotdir='', name='output'): # lables: list of strings

    x = X.detach().cpu().numpy()

    c1='red'
    c2='blue'

    fig = plt.figure()
    ax = plt.axes()
    N = x.shape[1]
    if labels == '':
        labels = [str(i) for i in np.arange(1, N+1)]
    handles = []
    for i in range(N):
        h = ax.plot(x[:, i], color=colorFader(c1, c2, i/N), linestyle='solid', linewidth=2, label=labels[i])
        handles.append(h)

    plt.xlabel("Time Steps", fontsize=18)
    plt.ylabel("Displacement", fontsize=18)
    plt.title(f"waves_epoch{epoch}", fontsize=18, fontweight="bold")
    plt.grid(color='grey', linestyle=':', linewidth=0.75)
    if labels:
        handles_, labels_ = ax.get_legend_handles_labels()
        ax.legend(handles_, labels_, loc='upper right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.ylim((-0.0025, 0.0025))
    plt.tight_layout()

    if plotdir:
        fig.savefig(plotdir+'waves_'+name+'_epoch%d.png' % (epoch))
        plt.close(fig)
        
    return fig  

# plot the displacements
def plot_waves_2(epoch, X, labels, plotdir, name, y_lim, colors): # lables: list of strings

    x = X.detach().cpu().numpy()

    lines = ["-","--","-.",":"]
    
    fig = plt.figure()
    ax = plt.axes()
    N = x.shape[1]
    if labels == '':
        labels = [str(i) for i in np.arange(1, N+1)]
    handles = []
    for i in range(N):
        h = ax.plot(x[:, i], color=colors[i], linestyle=lines[i], linewidth=2, label=labels[i])
        handles.append(h)

    plt.xlabel("Time Steps", fontsize=18)
    plt.ylabel("Displacement", fontsize=18)
    plt.title(f"Particle Displacements, Epoch={epoch}", fontsize=18, fontweight="bold")
    plt.grid(color='grey', linestyle=':', linewidth=0.75)
    if labels:
        handles_, labels_ = ax.get_legend_handles_labels()
        ax.legend(handles_, labels_, loc='upper right', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if (y_lim != None):
        plt.ylim((y_lim[0], y_lim[1]))
    plt.tight_layout()

    if plotdir:
        fig.savefig(plotdir+'waves_'+name+'_epoch%d.png' % (epoch), bbox_inches='tight', dpi=300, transparent=True)
        plt.close(fig)
        
    return fig 

# plot the configuration
def plot_config(epoch, N_, x_ini, y_ini, D0, Lx, Ly, K_padded, K_min, K_max, sources, probes, plotdir=''):    

    k = K_padded.detach().cpu().numpy()
    x = x_ini.detach().cpu().numpy()
    y = y_ini.detach().cpu().numpy()
    D = D0.item()
    L_x = Lx.item()
    L_y = Ly.item()
    N = int(N_.item())
    
    src = []
    for i in sources:
        src.append(i.coordinates())
    prb = []
    for j in probes:
        prb.append(j.coordinates())
    
    if K_min != None:
        k_min = K_min#.item()
    else:
        k_min = np.min(k[k>=0])
    if K_max != None:
        k_max = K_max#.item()
    else:
        k_max = np.max(k)
    
    if (k_min == k_max):
        k_min = 0

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ells = []
    alpha_all = []
    colors = []
    
    for i in range(N):
        x_now = x[i]%L_x
        y_now = y[i]%L_y
        for j in range(-1, 2):
            for l in range(-1, 2):     
                e = Ellipse((x_now+j*L_x, y_now+l*L_y), width=D, height=D, angle=0)
                ells.append(e)
                if k[i] != -1:
                    alpha_all.append((k[i]-k_min)/(k_max-k_min))
                    colors.append('k')
                else:
                    alpha_all.append(0.3)
                    colors.append('green')
    i = 0
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor(colors[i])
        e.set_alpha(alpha_all[i])
        e.set_zorder(0)
        i += 1

    ells = []
    for i in range(N):
        x_now = x[i]%L_x
        y_now = y[i]%L_y
        for j in range(-1, 2):
            for l in range(-1, 2):                        
                e = Ellipse((x_now+j*L_x, y_now+l*L_y), width=D, height=D, angle=0)
                e.set_edgecolor('k')
                e.set_linewidth(1.5)
                if i in src:
                    e.set_edgecolor('b')
                    e.set_linewidth(1.5)
                    plt.scatter(x_now, y_now, marker='o', s=100, color=(0, 0, 1, 1))
                elif i in prb:
                    e.set_edgecolor('r')
                    e.set_linewidth(1.5)
                    plt.scatter(x_now, y_now, marker='*', s=100, color=(1, 0, 0, 1))
                e.set_zorder(20)
                ells.append(e)

    for e in ells:
        ax.add_artist(e)
        e.set_facecolor('none')
        e.set_alpha(1)

    ax.set_xlim(0, L_x)
    ax.set_ylim(0, L_y)

    # draw walls
    ax.plot([0, L_x], [0, 0], color='black', linewidth=5)
    ax.plot([0, L_x], [L_y, L_y], color='black', linewidth=5)

    import matplotlib.lines as mlines
    
    red_star = mlines.Line2D([], [], color=(1, 0, 0), marker='*', linestyle='None',
                            markersize=40, label='Output')
    
    blue_circle = mlines.Line2D([], [], color=(0, 0, 1), marker='o', linestyle='None',
                            markersize=40, label='Input')
    
    #plt.legend(loc='upper center', handles=[red_star, blue_circle], bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=12)
    
    plt.set_cmap("Greys")
    plt.clim(k_min, k_max)
    cbar=plt.colorbar()
    cbar.set_label("Stiffness", fontsize=18)
    cbar.ax.tick_params(labelsize=18)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.tight_layout()
    if plotdir:
        fig.savefig(plotdir+'geometry_epoch%d.png' % (epoch), bbox_inches='tight', dpi=300, transparent=False)
        plt.close(fig)
        
    return fig  

# plot the configuration
def plot_config_epochs(epoch, N_, x_ini, y_ini, D0, Lx, Ly, K_padded, K_min, K_max, sources, probes, plotdir=''):    

    k = K_padded.detach().cpu().numpy()
    x = x_ini.detach().cpu().numpy()
    y = y_ini.detach().cpu().numpy()
    D = D0.item()
    L_x = Lx.item()
    L_y = Ly.item()
    N = int(N_.item())
    
    src = []
    for i in sources:
        src.append(i.coordinates())
    prb = []
    for j in probes:
        prb.append(j.coordinates())
    
    if K_min != None:
        k_min = K_min#.item()
    else:
        k_min = np.min(k[k>=0])
    if K_max != None:
        k_max = K_max#.item()
    else:
        k_max = np.max(k)
    
    if (k_min == k_max):
        k_min = 0

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ells = []
    alpha_all = []
    colors = []
    
    for i in range(N):
        x_now = x[i]%L_x
        y_now = y[i]%L_y
        for j in range(-1, 2):
            for l in range(-1, 2):     
                e = Ellipse((x_now+j*L_x, y_now+l*L_y), width=D, height=D, angle=0)
                ells.append(e)
                if k[i] != -1:
                    alpha_all.append((k[i]-k_min)/(k_max-k_min))
                    colors.append('k')
                else:
                    alpha_all.append(0.3)
                    colors.append('green')
    i = 0
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_facecolor(colors[i])
        e.set_alpha(alpha_all[i])
        e.set_zorder(0)
        i += 1

    ells = []
    for i in range(N):
        x_now = x[i]%L_x
        y_now = y[i]%L_y
        for j in range(-1, 2):
            for l in range(-1, 2):                        
                e = Ellipse((x_now+j*L_x, y_now+l*L_y), width=D, height=D, angle=0)
                e.set_edgecolor('k')
                e.set_linewidth(1.5)
                if i in src:
                    e.set_edgecolor('b')
                    e.set_linewidth(1.5)
                    plt.scatter(x_now, y_now, marker='o', s=100, color=(0, 0, 1, 1))
                elif i in prb:
                    e.set_edgecolor('r')
                    e.set_linewidth(1.5)
                    plt.scatter(x_now, y_now, marker='*', s=100, color=(1, 0, 0, 1))
                e.set_zorder(20)
                ells.append(e)

    for e in ells:
        ax.add_artist(e)
        e.set_facecolor('none')
        e.set_alpha(1)

    ax.set_xlim(0, L_x)
    ax.set_ylim(0, L_y)

    # draw walls
    ax.plot([0, L_x], [0, 0], color='black', linewidth=5)
    ax.plot([0, L_x], [L_y, L_y], color='black', linewidth=5)

    import matplotlib.lines as mlines
    
    red_star = mlines.Line2D([], [], color=(1, 0, 0), marker='*', linestyle='None',
                            markersize=40, label='Output')
    
    blue_circle = mlines.Line2D([], [], color=(0, 0, 1), marker='o', linestyle='None',
                            markersize=40, label='Input')
    
    #plt.legend(loc='upper center', handles=[red_star, blue_circle], bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=12)
    
    plt.set_cmap("Greys")
    plt.clim(k_min, k_max)
    #cbar=plt.colorbar()
    #cbar.set_label("Stiffness", fontsize=18)
    #cbar.ax.tick_params(labelsize=18)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.tight_layout()
    if plotdir:
        fig.savefig(plotdir+'geometry_epoch%d.png' % (epoch), bbox_inches='tight', dpi=300, transparent=False)
        plt.close(fig)
        
    return fig