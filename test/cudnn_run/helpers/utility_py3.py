import sys

is_py3 = sys.version_info >= (3, 0)

if is_py3:
    xrange = range
    basestring = str

if is_py3:
    from io import StringIO
else:
    from StringIO import StringIO

# copied from https://github.com/benjaminp/six/blob/master/six.py
if sys.version_info[0] == 3:
    def reraise(tp, value, tb=None):
        try:
            if value is None:
                value = tp()
            if value.__traceback__ is not tb:
                raise value.with_traceback(tb)
            raise value
        finally:
            value = None
            tb = None
else:
    exec("""def reraise(tp, value, tb=None):
    try:
        raise tp, value, tb
    finally:
        tb = None
""")

# eof
