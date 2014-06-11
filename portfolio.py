import pandas as pd
import numpy as np
import scipy as sp

class Portfolio(object):

    
    def __init__(self):
        '''establishes a portfolio class
        '''
        import pandas as pd
        import numpy as np
        import scipy as sp
        
        self.Assets=pd.DataFrame(columns=['mu','vol','weight'], dtype='float')
        self.Assets.index.name='Asset'
        self.Rho=pd.DataFrame(columns=[], dtype='float')
        
    
    def add_asset(self, asset, mu, vol, weight):
        '''Adds an asset class to the portfolio
        '''
        self.Assets.loc[asset]=[mu, vol, weight]
        self.Rho.ix[asset,asset]=1
        
    def add_rho(self, asset1, asset2, rho):
        '''adds correlation coefficients  to the table, also ensuring it remains symetric
        '''
        if set([asset1, asset2]).issubset(self.Assets.index):
            self.Rho.ix[asset1, asset2]=rho
            self.Rho.ix[asset2, asset1]=rho
        else:
            print("One of the assets is not defined.  Please add the asset before assigning correlation")
        
    def monte_carlo(self, time):
        '''performs an N period montecarlo simulation of the portfolio, without rebalanceing.
        Returns: end value factor
        '''
        #convert everything germane into np arrays
        mu=self.Assets[['mu']].values        
        vol=self.Assets[['vol']].values
        var=self.var_covar()
        
        #generate 1xN matrix of Nrand numbers
        Z=np.random.randn(len(self.Assets.index),1)
        #generate a matrix of multivariate normal numbers by Chol(VAR).T*Z
        Z_interaction=np.dot(np.linalg.cholesky(var).T,Z)
        
        drift=np.dot((mu-np.power(vol,2)/2), time)
        noise=Z_interaction*time**.5
        #combine GBR
        value=np.exp(np.multiply(np.add(drift,noise), self.Assets[['weight']].values).sum())
    
        return(value)
    
    def var_covar(self):
        '''Returns VAR_COVAR matrix of the portfolio
        '''
        vold=np.diag(self.Assets['vol'])
        
        return(np.dot( np.dot(vold, self.Rho.values), vold))
    def set_default_portfolio(self):
        '''sets up default portfolio'''
        equity=.60
        domestic=.7
        international=.3

        debt=.35

        reit=.1

        #setup portfolio
    
        self.add_asset('large domestic', .0849, .1475, equity*domestic*.65)
        self.add_asset('mid domestic', .0919, .1775, equity*domestic*.27)
        self.add_asset('small domestic', .0924, .1975, equity*domestic*.08)
        self.add_asset('international equity', .087, .145, equity*international)
        self.add_asset('agg debt', .0435, .045, debt)
        self.add_asset('US REIT', .0855, .2, reit)

        self.add_rho('large domestic', 'mid domestic', .96)
        self.add_rho('large domestic', 'small domestic', .92)
        self.add_rho('large domestic', 'international equity', .88)
        self.add_rho('large domestic', 'agg debt', .04)
        self.add_rho('large domestic', 'US REIT', .77)


        self.add_rho('mid domestic', 'small domestic', .94)
        self.add_rho('mid domestic', 'international equity', .88)
        self.add_rho('mid domestic', 'agg debt', .03)
        self.add_rho('mid domestic', 'US REIT', .79)

        self.add_rho('small domestic', 'international equity', .82)
        self.add_rho('small domestic', 'agg debt', -.04)
        self.add_rho('small domestic', 'US REIT', .79)

        self.add_rho('international equity', 'agg debt', -.02)
        self.add_rho('international equity', 'US REIT', .66)

        self.add_rho('agg debt', 'US REIT', .22)
