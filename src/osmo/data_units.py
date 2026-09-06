"""Explicit observation-unit conversions. Unknown units never receive guesses."""
import re
import numpy as np

def clean_unit(unit):
    return str(unit).strip().lower().replace('\u00b5','u').replace('\u03bc','u').replace(' ','')

def time_factor(unit):
    return {'h':1.,'hr':1.,'hour':1.,'hours':1.,'min':1/60,'minute':1/60,'s':1/3600,'sec':1/3600,'day':24.,'d':24.}.get(clean_unit(unit))

def concentration_factor(unit,molecular_weight=None):
    u=clean_unit(unit)
    mass={'ng/ml':1.,'ug/l':1.,'mg/l':1000.,'ug/ml':1000.,'pg/ml':.001,'ng/l':.001,'mg/ml':1e6,'kg/l':1e9}
    if u in mass:return mass[u]
    if molecular_weight is None or not np.isfinite(molecular_weight) or molecular_weight<=0:return None
    return {'umol/l':molecular_weight,'nmol/l':molecular_weight/1000,'mol/l':molecular_weight*1e6}.get(u)

def parse_dose(text,unit=None):
    s=clean_unit(text)
    if unit and re.fullmatch(r'[0-9.eE+-]+',s):s+=clean_unit(unit)
    m=re.fullmatch(r'([0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?)(kg|mg|ug|ng|g)(/kg)?',s)
    if not m:return None,None
    v=float(m[1])*{'kg':1e6,'g':1000,'mg':1,'ug':.001,'ng':1e-6}[m[2]]
    return (v,'mg/kg' if m[3] else 'mg') if np.isfinite(v) and v>0 else (None,None)

def single_administration(text):
    try:
        v=float(str(text).strip())
        return v if np.isfinite(v) else None
    except (ValueError,TypeError):return None
