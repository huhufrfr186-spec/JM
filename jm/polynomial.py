"""Sparse polynomials in (g, xi, v), with vector channel coefficients."""
import numpy as np

BASIS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))

def constant(v):
    return {(0, 0, 0): np.asarray(v, dtype=np.float64)}

def spatial(c):
    return {(0, x, y): np.asarray(v) for (x, y), v in zip(BASIS, c)}

def add(*ps):
    out = {}
    for p in ps:
        for k, v in p.items():
            out[k] = out.get(k, 0.) + v
    return out

def scale(p, a):
    return {k: a*v for k, v in p.items()}

def mul(p, q):
    out = {}
    for (k, x, y), a in p.items():
        for (l, u, v), b in q.items():
            key = k+l, x+u, y+v
            out[key] = out.get(key, 0.) + a*b
    return out

def shift(p, x, y):
    return {(k, u+x, v+y): a for (k, u, v), a in p.items()}

def contract(p, moments):
    return sum((a*moments[k, x, y] for (k, x, y), a in p.items()), np.array(0.))

def coefficients(mu, precision, log_o, center, half):
    d = center-mu
    linear = -(precision@d)*half
    q = precision*np.outer(half, half)
    return np.array([log_o-.5*d@precision@d, *linear, -.5*q[0,0], -q[0,1], -.5*q[1,1]])

def quadratic_range(c):
    """Exact extrema of a quadratic on [-1,1]^2, including edges."""
    a,b,d,e,f,g = c
    pts = [(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)]
    for x in (-1.,1.):
        if abs(g)>1e-30:
            y=-(d+f*x)/(2*g)
            if -1<=y<=1: pts.append((x,y))
    for y in (-1.,1.):
        if abs(e)>1e-30:
            x=-(b+f*y)/(2*e)
            if -1<=x<=1: pts.append((x,y))
    h=np.array([[2*e,f],[f,2*g]])
    if abs(np.linalg.det(h))>1e-30:
        p=np.linalg.solve(h,-np.array([b,d]))
        if np.all(abs(p)<=1): pts.append(p)
    vals=[a+b*x+d*y+e*x*x+f*x*y+g*y*y for x,y in pts]
    return min(vals),max(vals)
