import cPickle
import rpy2.robjects as robjects
from rpy2.robjects.numpy2ri import numpy2ri
robjects.conversion.py2ri = numpy2ri
robjects.r("library(BayesTree)")
robjects.r("set.seed(99)")

feattr, featcv, fluxtr, fluxcv = cPickle.load(open("CHER.pickle", "rb"))

rbart     = robjects.r("bart")
fluxtrVec = robjects.FloatVector(fluxtr.transpose().reshape((fluxtr.size)))
bartFit   = rbart(feattr, fluxtrVec, featcv, keepevery=10)
preds     = np.array(bartFit[7])

# From BayesTree/R/bart.R
#   retval = list(
# 0     call=match.call(),
# 1     first.sigma=first.sigma,
# 2     sigma=sigma,
# 3     sigest=sigest,
# 4     yhat.train=yhat.train,
# 5     yhat.train.mean=yhat.train.mean,
# 6     yhat.test=yhat.test,
# 7     yhat.test.mean=yhat.test.mean,
# 8     varcount=varcount,
# 9     y = y.train
#   )

import pdb; pdb.set_trace()
