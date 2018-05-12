from __future__ import print_function
import numpy as np
from sympy import Symbol,Function
import sympy

"""
Routines to reduce and evaluate symbolic mode coupling integrals
"""

def factorize_2d_convolution_integral(expr,l1x=None,l1y=None,l2x=None,l2y=None,validate=True):
    """Reduce a sympy expression of variables l1x,l1y,l2x,l2y into a sum of 
    products of factors that depend only on vec(l1) and vec(l2). If the expression
    appeared as the integrand in an integral over vec(l1), where 
    vec(l2) = vec(L) - vec(l1) then this reduction allows one to evaluate the 
    integral as a function of vec(L) using FFTs instead of as a convolution.
    
    

    """

    val_fail_message = "Validation failed. This expression is likely not reducible to FFT form."
    if l1x is None: l1x = Symbol('l1x')
    if l1y is None: l1y = Symbol('l1y')
    if l2x is None: l2x = Symbol('l2x')
    if l2y is None: l2y = Symbol('l2y')

    terms = []
    if validate: prodterms = []
    expr = sympy.expand( expr )
    for arg in expr.args:
        ll1terms = []
        ll2terms = []
        loterms = []
        for argi in arg.args:
            if argi.has(l1x) or argi.has(l1y):
                ll1terms.append(argi)
            elif argi.has(l2x) or argi.has(l2y):
                ll2terms.append(argi)
            else:
                loterms.append(argi)

        tdict = {}
                
        tdict['l1'] = sympy.Mul(*ll1terms)
        tdict['l2'] = sympy.Mul(*ll2terms)
        tdict['other'] = sympy.Mul(*loterms)
        terms.append(tdict)
        if validate:
            products = sympy.Mul(*tdict.values())
            assert sympy.simplify(products-arg)==0, val_fail_message
            prodterms.append(products)
            assert not(tdict['l1'].has(l2x)), val_fail_message
            assert not(tdict['l2'].has(l1x)), val_fail_message
            assert not(tdict['other'].has(l1x)), val_fail_message
            assert not(tdict['other'].has(l2x)), val_fail_message
            assert not(tdict['l1'].has(l2y)), val_fail_message
            assert not(tdict['l2'].has(l1y)), val_fail_message
            assert not(tdict['other'].has(l1y)), val_fail_message
            assert not(tdict['other'].has(l2y)), val_fail_message
    if validate:
        fexpr = sympy.Add(*prodterms)
        assert sympy.simplify(expr-fexpr)==0, val_fail_message
    return terms



def qe_tt():

    Lx = Symbol('Lx')
    Ly = Symbol('Ly')
    l1x = Symbol('l1x')
    l1y = Symbol('l1y')
    l2x = Symbol('l2x')
    l2y = Symbol('l2y')

    uC1 = Function('uC1')(l1x,l1y)
    uC2 = Function('uC1')(l2x,l2y)
    tC1 = Function('tC1')(l1x,l1y)
    tC2 = Function('tC2')(l2x,l2y)

    ldl1 = (Lx*l1x+Ly*l1y)
    ldl2 = (Lx*l2x+Ly*l2y)
    falpha = uC1*ldl1 + uC2*ldl2
    F = ldl1 * uC1 / tC1 / tC2

    expr = falpha*F
    terms = factorize_2d_convolution_integral(expr)
    for term in terms:
        print(term)

qe_tt()
