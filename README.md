# aipyw
minimal, fast object oriented implementation of the AIPW estimator with scikitlearners and cross-fitting.

minimal example

```python
import numpy as np
import sklearn
import sklearn.pipeline as skpipe
import celer as cel

# dml dgps
from doubleml import datasets
# this module
from aipyw import AIPyW
```

```python
#######################################################################
# make some data - true effect is 1
X, y, w = datasets.make_irm_data(1_000, theta=1, return_type='array')
# naive estimate is biased
y[w==1].mean() - y[w==0].mean()

1.5326252413874115
```

```python
# scale to unit interval and sieve
ppl = skpipe.Pipeline([
	('minmax', sklearn.preprocessing.MinMaxScaler()),
	('sieve',  sklearn.preprocessing.PolynomialFeatures(2)),
])
XX = ppl.fit_transform(X)
#######################################################################
# initialise it with data and model objects
doubledouble = aipyw(y, w, XX,
                    omod = cel.ElasticNetCV(l1_ratio= [.5, .7, .9],
                                            n_alphas=20, cv=5, n_jobs = 8),
                    pmod = cel.LogisticRegression(C=1)
)
# fit
doubledouble.fit()
# summarise
doubledouble.summary()

#                                     ATE        SE  95% CI-LB  95% CI-UB
# Treat level 1 - Treat level 0  1.062369  0.071074   0.923063   1.201675
```

For a more detailed walkthrough and an example with multiple discrete treatments, see `00_demo.ipynb`.
