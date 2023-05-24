REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .basic_central_controller import CentralBasicMAC
from .dop_controller import DOPMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["dop_mac"] = DOPMAC