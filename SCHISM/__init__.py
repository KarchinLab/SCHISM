try:
    # For Python 3.8+
    from importlib.metadata import version
except ImportError:
    # Fallback to pkg_resources for older versions
    from pkg_resources import get_distribution
    version = lambda pkg: get_distribution(pkg).version

__version__ = version('SCHISM')
