from .mnist import get_mnist
from .usps import get_usps

from .office_31 import get_office_31
from .office_home import get_office_home

__all__ = (get_usps, get_mnist, \
            get_office_31, \
            get_office_home)
