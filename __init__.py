# __init__.py for vace-radial ComfyUI node pack
# This file registers the nodes in nodes.py with ComfyUI

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .nodes import NODE_CLASS_MAPPINGS as NODES, NODE_DISPLAY_NAME_MAPPINGS as DISPLAY_NAMES
    NODE_CLASS_MAPPINGS.update(NODES)
    NODE_DISPLAY_NAME_MAPPINGS.update(DISPLAY_NAMES)
except ImportError:
    pass
