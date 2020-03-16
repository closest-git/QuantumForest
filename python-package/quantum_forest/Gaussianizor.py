"""
Transform data so that it is approximately normally distributed

This code written by Greg Ver Steeg, 2015.
"""

from typing import Text, List, Union

import numpy as np
from scipy import special
from scipy.stats import kurtosis, norm, rankdata, boxcox
from scipy import optimize  # TODO: Explore efficacy of other opt. methods
import sklearn
from matplotlib import pylab as plt
from scipy import stats
import warnings
import os

np.seterr(all='warn')


# Tolerance for == 0.0 tolerance.
_EPS = 1e-6
_EPS = 1e-2

def _update_x(x: Union[np.ndarray, List]) -> np.ndarray:
    x = np.asarray(x)
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    elif len(x.shape) != 2:
        raise ValueError("Data should be a 1-d list of samples to transform or a 2d array with samples as rows.")
    return x


class Gaussianize(sklearn.base.TransformerMixin):
    """
    Gaussianize data using various methods.

    Conventions
    ----------
    This class is a wrapper that follows sklearn naming/style (e.g. fit(X) to train).
    In this code, x is the input, y is the output. But in the functions outside the class, I follow
    Georg's convention that Y is the input and X is the output (Gaussianized) data.

    Parameters
    ----------
    
    strategy : str, default='lambert'. Possibilities are 'lambert'[1], 'brute'[2] and 'boxcox'[3].

    tol : float, default = 1e-4

    max_iter : int, default = 100
        Maximum number of iterations to search for correct parameters of Lambert transform.

    Attributes
    ----------
    coefs_ : list of tuples
        For each variable, we have transformation parameters.
        For Lambert, e.g., a tuple consisting of (mu, sigma, delta), corresponding to the parameters of the
        appropriate Lambert transform. Eq. 6 and 8 in the paper below.

    References
    ----------
    [1] Georg M Goerg. The Lambert Way to Gaussianize heavy tailed data with
                        the inverse of Tukey's h transformation as a special case
        Author generously provides code in R: https://cran.r-project.org/web/packages/LambertW/
    [2] Valero Laparra, Gustavo Camps-Valls, and Jesus Malo. Iterative Gaussianization: From ICA to Random Rotations
    [3] Box cox transformation and references: https://en.wikipedia.org/wiki/Power_transform
    """

    def __init__(self, strategy: Text = 'lambert', 
                 tol: float = 1e-5, 
                 max_iter: int = 100, 
                 verbose: bool = False):
        self.tol = tol
        self.max_iter = max_iter
        self.strategy = strategy
        self.coefs_ = []  # Store tau for each transformed variable
        self.verbose = verbose
        
    def fit(self, x: np.ndarray, y=None):
        """Fit a Gaussianizing transformation to each variable/column in x."""
        # Initialize coefficients again with an empty list.  Otherwise
        # calling .fit() repeatedly will augment previous .coefs_ list.
        self.coefs_ = []
        x = _update_x(x)
        if self.verbose:
            print("Gaussianizing with strategy='%s'" % self.strategy)
        
        if self.strategy == "lambert":
            _get_coef = lambda vec: igmm(vec, self.tol, max_iter=self.max_iter)
        elif self.strategy == "brute":
            _get_coef = lambda vec: None   # TODO: In principle, we could store parameters to do a quasi-invert
        elif self.strategy == "boxcox":
            _get_coef = lambda vec: boxcox(vec)[1]
        else:
            raise NotImplementedError("stategy='%s' not implemented." % self.strategy)

        for x_i in x.T:
            self.coefs_.append(_get_coef(x_i))

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform new data using a previously learned Gaussianization model."""
        x = _update_x(x)
        if x.shape[1] != len(self.coefs_):
            raise ValueError("%d variables in test data, but %d variables were in training data." % (x.shape[1], len(self.coefs_)))

        if self.strategy == 'lambert':
            return np.array([w_t(x_i, tau_i) for x_i, tau_i in zip(x.T, self.coefs_)]).T
        elif self.strategy == 'brute':
            return np.array([norm.ppf((rankdata(x_i) - 0.5) / len(x_i)) for x_i in x.T]).T
        elif self.strategy == 'boxcox':
            return np.array([boxcox(x_i, lmbda=lmbda_i) for x_i, lmbda_i in zip(x.T, self.coefs_)]).T
        else:
            raise NotImplementedError("stategy='%s' not implemented." % self.strategy)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Recover original data from Gaussianized data."""
        if self.strategy == 'lambert':
            return np.array([inverse(y_i, tau_i) for y_i, tau_i in zip(y.T, self.coefs_)]).T
        elif self.strategy == 'boxcox':
            return np.array([(1. + lmbda_i * y_i) ** (1./lmbda_i) for y_i, lmbda_i in zip(y.T, self.coefs_)]).T
        else:
            raise NotImplementedError("Inversion not supported for gaussianization transform '%s'" % self.strategy)

    def qqplot(self, x: np.ndarray, prefix: Text = 'qq', output_dir: Text = "/tmp/"):
        """Show qq plots compared to normal before and after the transform."""
        x = _update_x(x)
        y = self.transform(x)
        n_dim = y.shape[1]
        for i in range(n_dim):
            stats.probplot(x[:, i], dist="norm", plot=plt)
            plt.savefig(os.path.join(output_dir, prefix + '_%d_before.png' % i))
            plt.clf()
            stats.probplot(y[:, i], dist="norm", plot=plt)
            plt.savefig(os.path.join(output_dir, prefix + '_%d_after.png' % i))
            plt.clf()


def w_d(z, delta):
    # Eq. 9
    if delta < _EPS:
        return z
    return np.sign(z) * np.sqrt(np.real(special.lambertw(delta * z ** 2)) / delta)


def w_t(y, tau):
    # Eq. 8
    return tau[0] + tau[1] * w_d((y - tau[0]) / tau[1], tau[2])


def inverse(x, tau):
    # Eq. 6
    u = (x - tau[0]) / tau[1]
    return tau[0] + tau[1] * (u * np.exp(u * u * (tau[2] * 0.5)))


def igmm(y: np.ndarray, tol: float = 1e-6, max_iter: int = 100):
    # Infer mu, sigma, delta using IGMM in Alg.2, Appendix C
    if np.std(y) < _EPS:
        return np.mean(y), np.std(y).clip(_EPS), 0
    delta0 = delta_init(y)
    tau1 = (np.median(y), np.std(y) * (1. - 2. * delta0) ** 0.75, delta0)
    for k in range(max_iter):
        tau0 = tau1
        z = (y - tau1[0]) / tau1[1]
        delta1 = delta_gmm(z)
        x = tau0[0] + tau1[1] * w_d(z, delta1)
        mu1, sigma1 = np.mean(x), np.std(x)
        tau1 = (mu1, sigma1, delta1)

        if np.linalg.norm(np.array(tau1) - np.array(tau0)) < tol:
            break
        else:
            if k == max_iter - 1:
                warnings.warn("Warning: No convergence after %d iterations. Increase max_iter." % max_iter)
    return tau1


def delta_gmm(z):
    # Alg. 1, Appendix C
    delta0 = delta_init(z)

    def func(q):
        u = w_d(z, np.exp(q))
        if not np.all(np.isfinite(u)):
            return 0.
        else:
            u_0,u_1=np.min(u),np.max(u)
            if(u_1-u_0)<1.0e-6:
                return 0
            k = kurtosis(u, fisher=True, bias=False)**2
            if not np.isfinite(k) or k > 1e10:
                return 1e10
            else:
                return k

    res = optimize.fmin(func, np.log(delta0), disp=0)
    return np.around(np.exp(res[-1]), 6)


def delta_init(z):
    gamma = kurtosis(z, fisher=False, bias=False)
    with np.errstate(all='ignore'):
        delta0 = np.clip(1. / 66 * (np.sqrt(66 * gamma - 162.) - 6.), 0.01, 0.48)
    if not np.isfinite(delta0):
        delta0 = 0.01
    return delta0


if __name__ == '__main__':
    # Command line interface
    # Sample commands:
    # python gaussianize.py test_data.csv
    import csv
    import sys, os
    import traceback
    from optparse import OptionParser, OptionGroup

    parser = OptionParser(usage="usage: %prog [options] data_file.csv \n"
                                "It is assumed that the first row and first column of the data CSV file are labels.\n"
                                "Use options to indicate otherwise.")
    group = OptionGroup(parser, "Input Data Format Options")
    group.add_option("-c", "--no_column_names",
                     action="store_true", dest="nc", default=False,
                     help="We assume the top row is variable names for each column. "
                          "This flag says that data starts on the first row and gives a "
                          "default numbering scheme to the variables (1,2,3...).")
    group.add_option("-r", "--no_row_names",
                     action="store_true", dest="nr", default=False,
                     help="We assume the first column is a label or index for each sample. "
                          "This flag says that data starts on the first column.")
    group.add_option("-d", "--delimiter",
                     action="store", dest="delimiter", type="string", default=",",
                     help="Separator between entries in the data, default is ','.")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Transform Options")
    group.add_option("-s", "--strategy",
                     action="store", dest="strategy", type="string", default="lambert",
                     help="Strategy.")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Output Options")
    group.add_option("-o", "--output",
                     action="store", dest="output", type="string", default="gaussian_output.csv",
                     help="Where to store gaussianized data.")
    group.add_option("-q", "--qqplots",
                     action="store_true", dest="q", default=False,
                     help="Produce qq plots for each variable before and after transform.")
    parser.add_option_group(group)

    (options, args) = parser.parse_args()
    if not len(args) == 1:
        warnings.warn("Run with '-h' option for usage help.")
        sys.exit()

    #Load data from csv file
    filename = args[0]
    with open(filename, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")  #options.delimiter)
        if options.nc:
            variable_names = None
        else:
            variable_names = reader.next()[(1 - options.nr):]
        sample_names = []
        data = []
        for row in reader:
            if options.nr:
                sample_names = None
            else:
                sample_names.append(row[0])
            data.append(row[(1 - options.nr):])

    print(len(data), data[0])
    try:
        for i in range(len(data)):
            data[i] = map(float, data[i])
        X = np.array(data, dtype=float)  # Data matrix in numpy format
    except:
        raise ValueError("Incorrect data format.\nCheck that you've correctly specified options "
                         "such as continuous or not, \nand if there is a header row or column.\n"
                         "Run 'python gaussianize.py -h' option for help with options.")
        traceback.print_exc(file=sys.stdout)
        sys.exit()

    ks = []
    for xi in X.T:
        ks.append(kurtosis(xi))
    print(np.mean(np.array(ks) > 1))
    from matplotlib import pylab
    pylab.hist(ks, bins=30)
    pylab.xlabel('excess kurtosis')
    pylab.savefig('excess_kurtoses_all.png')
    pylab.clf()
    pylab.hist([k for k in ks if k < 2], bins=30)
    pylab.xlabel('excess kurtosis')
    pylab.savefig('excess_kurtoses_near_zero.png')
    print(np.argmax(ks))
    pdict = {}
    for k in np.argsort(- np.array(ks))[:50]:
        pylab.clf()
        p = np.argmax(X[:, k])
        pdict[p] = pdict.get(p, 0) + 1
        pylab.hist(X[:, k], bins=30)
        pylab.xlabel(variable_names[k])
        pylab.ylabel('Histogram of patients')
        pylab.savefig('high_kurtosis/'+variable_names[k] + '.png')
    print(pdict)  # 203, 140 appear three times.
    sys.exit()
    out = Gaussianize(strategy=options.strategy)
    y = out.fit_transform(X)
    with open(options.output, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=options.delimiter)
        if not options.nc:
            writer.writerow([""] * (1 - options.nr) + variable_names)
        for i, row in enumerate(y):
            if not options.nr:
                writer.writerow([sample_names[i]] + list(row))
            else:
                writer.writerow(row)

    if options.q:
        print('Making qq plots')
        prefix = options.output.split('.')[0]
        if not os.path.exists(prefix+'_q'):
            os.makedirs(prefix+'_q')
        out.qqplot(X, prefix=prefix + '_q/q')