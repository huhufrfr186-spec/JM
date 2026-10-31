import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import t

def surface_metrics(prediction,reference,threshold):
    a=cKDTree(reference).query(prediction)[0]; b=cKDTree(prediction).query(reference)[0]
    precision=float(np.mean(a<threshold)); recall=float(np.mean(b<threshold))
    return dict(accuracy=float(a.mean()),completeness=float(b.mean()),cd_l1=float((a.mean()+b.mean())/2),precision=precision,recall=recall,f_score=2*precision*recall/max(precision+recall,1e-30))

def paired_interval(a,b):
    """Input shape: buildings x repeats. Buildings receive equal weight."""
    d=np.asarray(a).mean(axis=1)-np.asarray(b).mean(axis=1)
    if len(d)<2: raise ValueError('At least two paired buildings required.')
    margin=t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
    return dict(mean=float(d.mean()),lower=float(d.mean()-margin),upper=float(d.mean()+margin))
