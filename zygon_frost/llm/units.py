"""Units for Frost Math: exact conversion factors (fractions, never floats), a dimension check, and a reader for '5 ft', '3.5 gpm', '500 BTU/hr', '75 F'.
A dimension is 7 whole or fractional powers of: mass, length, time, current, temperature, amount, light.   Every value is held in SI inside Frost Math.
Where a unit has more than one definition the one used is named in its label (BTU = International Table, gallon = US, ton = refrigeration ton of 12,000 BTU/hr)."""
import re
from fractions import Fraction as Fr
import sympy as sp

DIMS = ("M", "L", "T", "I", "TH", "N", "J")
Z = (0,) * 7


def d(**k): return tuple(Fr(k.get(x, 0)) for x in DIMS)
def dadd(a, b): return tuple(x + y for x, y in zip(a, b))
def dsub(a, b): return tuple(x - y for x, y in zip(a, b))
def dmul(a, n): return tuple(x * n for x in a)


LEN, MASS, TIME, CUR, TEMP, AMT = d(L=1), d(M=1), d(T=1), d(I=1), d(TH=1), d(N=1)
FORCE = d(M=1, L=1, T=-2); PRESS = d(M=1, L=-1, T=-2); ENERGY = d(M=1, L=2, T=-2); POWER = d(M=1, L=2, T=-3)
VOL = d(L=3); AREA = d(L=2); VEL = d(L=1, T=-1); VOLFLOW = d(L=3, T=-1); MASSFLOW = d(M=1, T=-1); DENS = d(M=1, L=-3)
VOLT = d(M=1, L=2, T=-3, I=-1); OHM = d(M=1, L=2, T=-3, I=-2); FREQ = d(T=-1); ANGLE = Z; TORQUE = ENERGY
R = lambda s: sp.sympify(s, rational=True)

# name: (factor to SI as exact text, dims, label, offset to SI in the unit itself, e.g. Celsius)
U = {}


def add(names, factor, dims, label, offset="0"):
    for n in names.split():
        U[n] = (R(factor), dims, label, R(offset))


add("m meter metre", "1", LEN, "metre"); add("mm", "1/1000", LEN, "millimetre"); add("cm", "1/100", LEN, "centimetre"); add("km", "1000", LEN, "kilometre")
add("in inch inches", "254/10000", LEN, "inch"); add("ft foot feet", "3048/10000", LEN, "foot"); add("yd", "9144/10000", LEN, "yard"); add("mi", "1609344/1000", LEN, "mile")
add("m2 sqm", "1", AREA, "square metre"); add("mm2", "1/1000000", AREA, "square millimetre"); add("cm2", "1/10000", AREA, "square centimetre"); add("ft2 sqft", "9290304/100000000", AREA, "square foot"); add("in2 sqin", "64516/100000000", AREA, "square inch")
add("m3", "1", VOL, "cubic metre"); add("L l litre liter", "1/1000", VOL, "litre"); add("mL ml", "1/1000000", VOL, "millilitre"); add("ft3 cuft", "28316846592/1000000000000", VOL, "cubic foot"); add("gal", "3785411784/1000000000000", VOL, "US gallon"); add("in3", "16387064/1000000000000", VOL, "cubic inch")
add("s sec", "1", TIME, "second"); add("min", "60", TIME, "minute"); add("h hr hour", "3600", TIME, "hour"); add("day", "86400", TIME, "day")
add("kg", "1", MASS, "kilogram"); add("g", "1/1000", MASS, "gram"); add("lb lbs", "45359237/100000000", MASS, "pound"); add("oz", "28349523125/1000000000000", MASS, "ounce")
add("N", "1", FORCE, "newton"); add("kN", "1000", FORCE, "kilonewton"); add("lbf", "44482216152605/10000000000000", FORCE, "pound-force")
add("Pa", "1", PRESS, "pascal"); add("kPa", "1000", PRESS, "kilopascal"); add("MPa", "1000000", PRESS, "megapascal"); add("bar", "100000", PRESS, "bar"); add("psi", "6894757293168/1000000000", PRESS, "pound per square inch")
add("inwg inH2O inwc", "24908891/100000", PRESS, "inch of water (conventional: 1000 kg/m3, g = 9.80665)"); add("atm", "101325", PRESS, "standard atmosphere")
add("J", "1", ENERGY, "joule"); add("kJ", "1000", ENERGY, "kilojoule"); add("MJ", "1000000", ENERGY, "megajoule"); add("Wh", "3600", ENERGY, "watt-hour"); add("kWh", "3600000", ENERGY, "kilowatt-hour"); add("BTU Btu btu", "105505585262/100000000", ENERGY, "BTU (International Table)"); add("cal", "4184/1000", ENERGY, "thermochemical calorie"); add("kcal", "4184", ENERGY, "kilocalorie")
add("W", "1", POWER, "watt"); add("kW", "1000", POWER, "kilowatt"); add("MW", "1000000", POWER, "megawatt"); add("hp", "74569987158227022/100000000000000", POWER, "mechanical horsepower"); add("hpe", "746", POWER, "electrical horsepower (746 W)")
add("BTU/hr BTUH btuh", "105505585262/360000000000", POWER, "BTU per hour"); add("ton TR", "12000*105505585262/360000000000", POWER, "ton of refrigeration (12,000 BTU/hr)")
add("K", "1", TEMP, "kelvin"); add("C degC", "1", TEMP, "degree Celsius", "27315/100"); add("F degF", "5/9", TEMP, "degree Fahrenheit", "(27315/100)-(32*5/9)"); add("R degR", "5/9", TEMP, "degree Rankine")
add("dC", "1", TEMP, "Celsius degrees (difference)"); add("dF", "5/9", TEMP, "Fahrenheit degrees (difference)"); add("dK", "1", TEMP, "kelvin (difference)")
add("A", "1", CUR, "ampere"); add("mA", "1/1000", CUR, "milliampere"); add("V", "1", VOLT, "volt"); add("kV", "1000", VOLT, "kilovolt"); add("mV", "1/1000", VOLT, "millivolt"); add("ohm Ohm", "1", OHM, "ohm"); add("kohm", "1000", OHM, "kilo-ohm")
add("Hz", "1", FREQ, "hertz"); add("rpm RPM", "1/60", FREQ, "revolutions per minute"); add("rad", "1", ANGLE, "radian"); add("deg", "pi/180", ANGLE, "degree of angle")
add("mol", "1", AMT, "mole"); add("ratio", "1", Z, "no unit"); add("pct percent", "1/100", Z, "percent")
add("m/s", "1", VEL, "metre per second"); add("m/min", "1/60", VEL, "metre per minute"); add("km/h kph", "1/3.6", VEL, "kilometre per hour"); add("ft/s fps", "3048/10000", VEL, "foot per second"); add("ft/min fpm", "3048/600000", VEL, "foot per minute"); add("mph", "44704/100000", VEL, "mile per hour")
add("m3/s", "1", VOLFLOW, "cubic metre per second"); add("m3/h", "1/3600", VOLFLOW, "cubic metre per hour"); add("L/s lps", "1/1000", VOLFLOW, "litre per second"); add("L/min lpm", "1/60000", VOLFLOW, "litre per minute")
add("cfm CFM", "28316846592/60000000000000", VOLFLOW, "cubic foot per minute"); add("gpm GPM", "3785411784/60000000000000", VOLFLOW, "US gallon per minute")
add("kg/s", "1", MASSFLOW, "kilogram per second"); add("lb/hr lbh", "45359237/360000000000", MASSFLOW, "pound per hour"); add("kg/m3", "1", DENS, "kilogram per cubic metre"); add("lb/ft3", "(45359237/100000000)/(28316846592/1000000000000)", DENS, "pound per cubic foot")
add("W/m2", "1", d(M=1, T=-3), "watt per square metre"); add("BTU/hr.ft2", "105505585262/360000000000/(9290304/100000000)", d(M=1, T=-3), "BTU per hour per square foot")
add("N.m Nm", "1", TORQUE, "newton metre"); add("lbf.ft ftlb", "(44482216152605/10000000000000)*(3048/10000)", TORQUE, "pound-force foot")
add("1/s", "1", FREQ, "per second"); add("rad/s", "1", FREQ, "radian per second")
LOWER = {}
for k in U: LOWER.setdefault(k.lower(), k)


def unit(name):
    n = name.strip().replace("°", "").replace("Ω", "ohm").replace("²", "2").replace("³", "3").replace("µ", "u")
    if n in U: return U[n]
    if n.lower() in LOWER and (n.lower() not in ("m", "mm", "pa", "n", "k", "c", "f", "r", "a", "v", "l") or n.islower()): return U[LOWER[n.lower()]]
    return None


TOK = re.compile(r"\s*([A-Za-z_µΩ]+[0-9]?(?:\^-?\d+)?|\d+)\s*([*/.]?)")


def unit_expr(s):
    """'ft', 'lb/ft3', 'W/m2', 'kg*m/s2' -> (factor, dims, offset, label) or None"""
    s = s.strip()
    if not s: return None
    one = unit(s)
    if one: return one[0], one[1], one[3], one[2]
    fac, dim, pos, sign = sp.Integer(1), Z, 0, 1
    parts = re.split(r"([*/])", s.replace(" ", ""))
    sign = 1
    for i, p in enumerate(parts):
        if p in ("*", "/"): sign = 1 if p == "*" else -1; continue
        m = re.match(r"^([A-Za-z_µΩ.]+?)(?:\^?(-?\d+))?$", p)
        if not m: return None
        u = unit(m.group(1)) or (unit(p) if unit(p) else None)
        pw = int(m.group(2)) if m.group(2) else 1
        if not u:
            u = unit(p); pw = 1
            if not u: return None
        fac = fac * u[0] ** (sign * pw); dim = dadd(dim, dmul(u[1], sign * pw))
    return fac, dim, sp.Integer(0), s


def dim_text(dm):
    names = {"M": "kg", "L": "m", "T": "s", "I": "A", "TH": "K", "N": "mol", "J": "cd"}; up, dn = [], []
    for k, v in zip(DIMS, dm):
        if v == 0: continue
        t = names[k] + ("" if abs(v) == 1 else ("^" + (str(v.numerator) if v.denominator == 1 else f"({v.numerator}/{v.denominator})")))
        (up if v > 0 else dn).append(t if v > 0 else t.replace("^" + (str(v.numerator) if v.denominator == 1 else ""), "^" + str(abs(v.numerator)) if abs(v) != 1 else ""))
    if not up and not dn: return "no unit"
    return ("*".join(up) or "1") + ("/" + "*".join(dn) if len(dn) == 1 else ("/(" + "*".join(dn) + ")" if dn else ""))


PREFERRED = ["V", "A", "ohm", "W", "Pa", "N", "J", "N.m", "m", "m2", "m3", "s", "kg", "Hz", "K", "m/s", "m3/s", "kg/s", "kg/m3", "W/m2"]
KIND = {"V": "voltage", "A": "current", "ohm": "resistance", "W": "power", "Pa": "pressure", "N": "force", "J": "energy", "N.m": "torque", "m": "length", "m2": "area", "m3": "volume", "s": "time", "kg": "mass", "Hz": "frequency", "K": "temperature", "m/s": "speed", "m3/s": "volume flow", "kg/s": "mass flow", "kg/m3": "density", "W/m2": "heat flux"}


def friendly(dm):
    for n in PREFERRED:
        if same_dims(U[n][1], dm): return f"{KIND[n]} ({n})"
    return dim_text(dm)


def same_dims(a, b): return all(Fr(x) == Fr(y) for x, y in zip(a, b))


def is_difference(var): return bool(re.match(r"^(d|D|delta|Delta)[A-Z_]", var)) or "delta" in var.lower()


def to_si(text, want_dims=None, var=None):
    """'5 ft' or '12' -> (exact SI value, note). A bare number is taken as SI. A unit must match the dimension the box needs, or it is refused."""
    from frostmath import parse_value
    t = str(text).strip()
    try: return parse_value(t), None                                      # a bare number: SI as it stands
    except ValueError: pass
    for i in range(len(t) - 1, 0, -1):
        pre, suf = t[:i].strip(), t[i:].strip()
        if not pre or not suf: continue
        ue = unit_expr(suf)
        if not ue: continue
        try: x = parse_value(pre)
        except ValueError: continue
        fac, dm, off, lab = ue
        if want_dims is not None and not same_dims(dm, want_dims): raise ValueError(f"'{suf}' is a unit of {friendly(dm)}, but this box needs {friendly(want_dims)}")
        pure_temp = same_dims(dm, TEMP) and off != 0 and not (var and is_difference(var))
        v = x * fac + (off if pure_temp else 0)
        return v, f"{pre} {lab} = {v} in SI" if v != x else None
    raise ValueError(f"'{text}' is not a number, or its unit is unknown")


def compatible(dm, k=8):
    seen, out = set(), []
    for n, (f, dd, lab, off) in U.items():
        if same_dims(dd, dm) and lab not in seen: seen.add(lab); out.append(n)
    return out[:k]


def from_si(v, unit_name, var=None):
    ue = unit_expr(unit_name)
    if not ue: raise ValueError(f"unknown unit {unit_name}")
    fac, dm, off, lab = ue
    pure_temp = same_dims(dm, TEMP) and off != 0 and not (var and is_difference(var))
    return (v - (off if pure_temp else 0)) / fac, lab
