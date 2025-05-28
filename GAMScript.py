# %%
import numpy as np,pandas as pd
# %%
import statsmodels.api as sm

import statsmodels.formula.api as smf

data = sm.datasets.get_rdataset("dietox", "geepack").data

md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])

mdf = md.fit()
print(mdf.summary())


    
    #         Mixed Linear Model Regression Results
    # ========================================================
    # Model:            MixedLM Dependent Variable: Weight    
    # No. Observations: 861     Method:             REML      
    # No. Groups:       72      Scale:              11.3669   
    # Min. group size:  11      Log-Likelihood:     -2404.7753
    # Max. group size:  12      Converged:          Yes       
    # Mean group size:  12.0                                  
    # --------------------------------------------------------
    #             Coef.  Std.Err.    z    P>|z| [0.025 0.975]
    # --------------------------------------------------------
    # Intercept    15.724    0.788  19.952 0.000 14.179 17.268
    # Time          6.943    0.033 207.939 0.000  6.877  7.008
    # Group Var    40.394    2.149                            
    # ========================================================

# %%
import statsmodels.api as sm

from statsmodels.gam.api import GLMGam, BSplines

# import data
from statsmodels.gam.tests.test_penalized import df_autos

# create spline basis for weight and hp
x_spline = df_autos[['weight']]

bs = BSplines(x_spline, df=[12, 10], degree=[3, 3])

# penalization weight
alpha = np.array([21833888.8, 6460.38479])[0]

gam_bs = GLMGam.from_formula('city_mpg ~ fuel + drive', data=df_autos, smoother=bs, alpha=alpha)


res_bs = gam_bs.fit()

print(res_bs.summary())

res_bs.plot_partial(0)

gam_bs.get_robustcov_results()

# %%
