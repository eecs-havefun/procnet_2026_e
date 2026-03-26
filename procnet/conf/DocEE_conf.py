from dataclasses import dataclass
from procnet.conf.basic_conf import BasicConfig


@dataclass
class DocEEConfig(BasicConfig):
    def __init__(self):
        self.proxy_slot_num = getattr(self, "proxy_slot_num", None) or 16
        self.node_size = getattr(self, "node_size", None) or 512
        self.max_len = getattr(self, "max_len", None) or 510
        self.max_epochs = getattr(self, "max_epochs", None) or 1

        self.return_procnet_entity_nodes = getattr(self, "return_procnet_entity_nodes", False)
        self.use_procnet_entity_nodes = getattr(self, "use_procnet_entity_nodes", False)
