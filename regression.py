import numpy as np
import matplotlib.pyplot as plt
import util
from itertools import product
from util import density_Gaussian

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    mu = np.array([0,0])
    mu = mu.T

    covariance = np.array([[beta, 0], [0, beta]])
    a0vals = np.linspace(-1,1,100)
    a1vals = np.linspace(-1,1,100)

    a0mesh, a1mesh = np.meshgrid(a0vals, a1vals, sparse = False)

    rava0 = a0mesh.ravel()
    rava1 = a1mesh.ravel()

    avals = np.concatenate((rava0.reshape(-1,1), rava1.reshape(-1,1)), axis = 1)
    probability = util.density_Gaussian(mu, covariance, avals).reshape(a0mesh.shape)

    ctm = plt.contour(a0mesh, a1mesh, probability)
    areal = plt.plot(np.array([-0.1]), np.array([-0.5]), 'rx', label='True a value')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title('Prior Distribution Contour Map P(a)')
    plt.savefig('prior.png')
    plt.show()
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here

    bigZ = np.array(z).reshape(z.shape)
    X = np.concatenate((np.ones(x.shape), x), axis = 1)
    covariance = sigma2 * np.linalg.inv(X.T@X + (sigma2/beta)*np.identity(x.shape[1]))

    mu =  np.linalg.inv(X.T@X + (sigma2/beta) * np.identity(x.shape[1]))@X.T@z

    mu = mu.reshape(-1)
    a0vals = np.linspace(-1,1,100)
    a1vals = np.linspace(-1,1,100)

    a0mesh, a1mesh = np.meshgrid(a0vals, a1vals, sparse = False)

    rava0 = a0mesh.ravel()
    rava1 = a1mesh.ravel()

    avals = np.concatenate((rava0.reshape(-1,1), rava1.reshape(-1,1)), axis = 1)
    probability = util.density_Gaussian(mu, covariance, avals).reshape(a0mesh.shape)

    ctm = plt.contour(a0mesh, a1mesh, probability)
    areal = plt.plot(np.array([-0.1]), np.array([-0.5]), 'rx', label='True a value')
    plt.xlabel('a0')
    plt.ylabel('a1')
    if x.shape[0] == 1:
        title = 'Posterior Distribution Contour Map P(a|x1,z1)'
        plt.title(title)
    else:
        title = 'Posterior Distribution Contour Map P(a|x1,z1,..., x{},z{})'.format(x.shape[0], x.shape[0])
        plt.title(title)
    ghettotitle = 'posterior{}.png'.format(x.shape[0])
    plt.savefig(ghettotitle)
    plt.show()
    Cov = covariance
    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    x = np.array(x).reshape(-1,1)
    X = np.concatenate((np.ones(x.shape), x), axis=1)
    sigma2_pred = sigma2 + X@Cov@X.T
    variances = np.diag(sigma2_pred)
    mu_pred = X@mu
    stdevs = np.sqrt(variances)

    plt.xlabel('x')
    plt.ylabel('z')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title('Predictions for {} training samples'.format(x_train.shape[0]))
    plt.errorbar(x, mu_pred, yerr=stdevs, barsabove=True, ecolor='maroon', elinewidth=1, linewidth=1, capsize=1, color='firebrick')
    plt.scatter(x_train, z_train, label='Training Samples', c='navy', alpha = 0.5)
    plt.legend()
    ghettotitle = 'predict{}.png'.format(x_train.shape[0])
    plt.savefig(ghettotitle)
    plt.show()
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 100
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]

    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    print('mu = ',mu)
    print('cov = ', Cov)

    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
