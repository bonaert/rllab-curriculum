from rllab.misc.nb_utils import ExperimentDatabase
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np
from glob import glob

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def plot_reward(fig,data_unpickle,color,fig_dir):
    env = data_unpickle['env']
    #retrieve original policy
    poli = data_unpickle['policy']
    mean = poli.get_action(np.array((0,0)))[1]['mean']
    logstd = poli.get_action(np.array((0,0)))[1]['log_std'] 
    # def normal(x): return 1/(np.exp(logstd)*np.sqrt(2*np.pi) )*np.exp(-0.5/np.exp(logstd)**2*(x-mean)**2) 
    ax = fig.gca(projection='3d')
    X = np.arange(-2, 2, 0.05)
    Y = np.arange(-2, 2, 0.05)
    X, Y = np.meshgrid(X, Y)
    X_flat = X.reshape((-1,1))
    Y_flat = Y.reshape((-1,1))
    XY = np.concatenate((X_flat,Y_flat),axis=1)
    rew=np.array([env.reward_state(xy) for xy in XY]).reshape(np.shape(X))

    surf = ax.plot_surface(X, Y, rew, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # policy_at0 = [normal(s) for s in x]
    # plt.plot(x,policy_at0,color=color*0.5,label='Policy at 0')
    plt.title('Reward acording to the state')
    fig.colorbar(surf,shrink=0.8)
    # plt.show()
    if fig_dir:
        plt.savefig(os.path.join(fig_dir,'Reward_function'))
    else:
        print "No directory for saving plots"

#Plot learning curve
def plot_learning_curve(exp,color,fig_dir):#######
    lab = "bimod point mdp"
    plt.plot(exp.progress['AverageDiscountedReturn'], color=color,label = lab )
    plt.legend(loc='best')
    plt.title('Learning curve')
    plt.xlabel('iteration (500 steps each)')
    plt.ylabel('mean Reward')
    if fig_dir:
        plt.savefig(os.path.join(fig_dir,'learning_curve'))
    else:
        print "No directory for saving plots"


#final policy learned
def plot_policy_learned(data_unpickle,color,fig_dir=None):
    #recover the policy
    poli = data_unpickle['policy']
    #range to plot it
    X = np.arange(-2, 2, 0.5)
    Y = np.arange(-2, 2, 0.5)
    X, Y = np.meshgrid(X, Y)
    X_flat = X.reshape((-1,1))
    Y_flat = Y.reshape((-1,1))
    XY = np.concatenate((X_flat,Y_flat),axis=1)
    means_1  = np.zeros((np.size(X_flat),1))
    means_2  = np.zeros((np.size(Y_flat),1))
    logstd = np.zeros((np.size(X_flat),2))
    for i,xy in enumerate(XY):
        means_1[i], means_2[i] = poli.get_action(xy)[1]['mean']
        logstd[i] = poli.get_action(xy)[1]['log_std']
    means_1=means_1.reshape(np.shape(X))
    means_2=means_2.reshape(np.shape(Y))
    means_norm = np.sqrt(means_1**2+means_2**2)
    plt.quiver(X, Y, means_1, means_2, scale=1, scale_units='x')

    plt.title('Final policy')
    # plt.show()
    if fig_dir:
        plt.savefig(os.path.join(fig_dir,'policy_learned'))
    else:
        print "No directory for saving plots"

## estimate by MC the policy at 0!
def plot_snn_at0(fig, data_unpickle, itr=0, color=(1,0.1,0.1),fig_dir=None):
    #recover the policy
    poli = data_unpickle['policy']
    #range to plot it
    bound = 3
    num_bins=600
    step = (2.*bound)/num_bins
    samples=(num_bins)**2
    x = np.arange(-bound,bound+step, step)
    y = np.arange(-bound,bound+step, step)
    x, y = np.meshgrid(x,y)
    p_xy = np.zeros_like(x)
    for _ in xrange(samples):
        a = poli.get_action(np.array((0,0)))[0]
        idx_x=int(np.floor(a[0]/step) + bound/step)
        idx_y=int(np.floor(a[1]/step) + bound/step)
        # find the coord of the action in the grid
        if idx_x>=0 and idx_x<np.shape(x)[1]:
            px=idx_x
        elif idx<0:
            px=0
        else:
            px=np.shape(x)[1]
        # same for y
        if idx_y >= 0 and idx_y < np.shape(y)[0]:
            py = idx_y
        elif idx_y < 0:
            py = 0
        else:
            py = np.shape(y)[0]
        p_xy[px,py]+=1

    ax = fig.gca(projection='3d')
    p_xy=p_xy/float(samples)
    surf = ax.plot_surface(y,x,p_xy, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.title('Policy distribution at 0 after {} iter'.format(itr))
    fig.colorbar(surf,shrink=0.8)
    #plt.xlabel('next state')
    #plt.ylabel('probability mass')
    if fig_dir:
        plt.savefig(os.path.join(fig_dir,'MC_policy_learned_at0_iter{}'.format(itr)))
    else:
        print "No directory for saving plots"


def plot_all_policy_at0(path_experiment,color,num_iter=100,fig_dir=None):
    mean_at_0 = []
    var_at_0 = []
    for itr in range(num_iter):
        data_bimodal_1d = joblib.load(os.path.join(path_experiment,'itr_{}.pkl'.format(itr)))
        poli = data_bimodal_1d['policy']
        action_at_0 = poli.get_action(np.array((0,)))
        mean_at_0.append(action_at_0[1]['mean'])
        var_at_0.append(action_at_0[1]['log_std'])
    itr = range(num_iter)
    plt.plot(itr,mean_at_0, color=color, label = 'mean at 0')
    plt.plot(itr, var_at_0, color=color * 0.7, label = 'logstd at 0')
    plt.title('How the policy variates accross iterations')
    plt.xlabel('iteration')
    plt.ylabel('mean and variance at 0')
    plt.legend(loc=3)
    if fig_dir:
        plt.savefig(os.path.join(fig_dir,'policy_at_0'))
    else:
        print "No directory for saving plots"

## plot for all the experiments
def plot_all_exp(datadir):
    database = ExperimentDatabase(datadir,names_or_patterns='*')
    exps = database._experiments
    colors=[(1,0.1,0.1),(0.1,1,0.1),(0.1,0.1,1),(1,1,0)]

    for i, exp in enumerate(exps):
        #get the last pickle
        exp_name=exp.params['exp_name']
        path_experiment=os.path.join(datadir,exp_name)
        last_iter = np.size(exp.progress['Iteration']) - 1
        # pkl_name= 'itr_{}'.format(last_iter)
        # last_data_unpickle = joblib.load(os.path.join(path_experiment,pkl_name+'.pkl'))
        # first_data_unpickle = joblib.load(os.path.join(path_experiment,'itr_0.pkl'))
        pkl_file='params.pkl'
        last_data_unpickle  = joblib.load(os.path.join(path_experiment,pkl_file))
        first_data_unpickle = joblib.load(os.path.join(path_experiment,pkl_file))
        #create fig_dir
        fig_dir = os.path.join(path_experiment,'Figures')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        #fix a color for plots of this exp
        color = np.array(colors[i%3])
        #plot everything
        print 'Plotting for: ',exp_name
        fig1=plt.figure(1+(i/4)*6)
        plot_reward(fig1,first_data_unpickle,color,fig_dir)
        print 'Plotting learning curve'
        fig2=plt.figure(2+i*6)#(2+(i/3)*6)
        plot_learning_curve(exp,color,fig_dir)
        print 'Plotting last policy'
        fig3=plt.figure(3+i*6)#(3+(i/3)*6)
        # plot_policy_learned(last_data_unpickle, color, fig_dir=fig_dir)
        plot_snn_at0(fig3, last_data_unpickle, color=color, fig_dir=fig_dir)
        # print 'Plotting policy progress'
        # plt.figure(4)
        # plot_all_policy_at0(path_experiment,color,num_iter=last_iter+1,fig_dir=fig_dir)
        if (i+1)/3 > i/3:
            plt.close('all')

## plot for all the experiments
if __name__ == "__main__":
    import sys
    name_dir=sys.argv[1]
    path_dir = "./data/local/"+name_dir
    print "plotting all experiments in: " +path_dir
    plot_all_exp(path_dir)

