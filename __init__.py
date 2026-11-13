from comfy_api.latest import ComfyExtension

from .nodes import MossTileControlLoader, MossTileControlApply
from .lllite_nodes import MossLLLiteLoader, MossLLLiteApply


class MossLLLiteExtension(ComfyExtension):
    async def get_node_list(self):
        return [MossTileControlLoader, MossTileControlApply, MossLLLiteLoader, MossLLLiteApply]


async def comfy_entrypoint():
    return MossLLLiteExtension()
