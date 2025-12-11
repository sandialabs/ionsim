import pint
ureg = pint.UnitRegistry()

# Redefine Hz in such a way that it properly converts to and from
# radians per second
ureg.define('cycle = 2 * pi * rad = cyc')
ureg.define('cps = cycle / second')
ureg.define('Hz = cps = hertz')
