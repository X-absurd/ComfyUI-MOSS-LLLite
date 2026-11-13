"""Run from the ComfyUI root: python custom_nodes/ComfyUI-MOSS-LLLite/tests/test_nodes.py."""

import importlib.util
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import comfy.cli_args
comfy.cli_args.args.cpu = True
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.patcher_extension
import comfy.ldm.moss_image.model
import safetensors.torch
import torch
from torch import nn

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("moss_tile_node_test", PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
nodes = sys.modules[spec.name + ".nodes"]
architecture = sys.modules[spec.name + ".model"]


def fixture():
    backbone = comfy.ldm.moss_image.model.MOSSImage(
        dim=32, ffn_dim=64, num_heads=2, num_layers=2, text_dim=16,
        dtype=torch.float32, device="cpu", operations=nn,
    )
    base = nn.Module()
    base.diffusion_model = backbone
    base.model_sampling = comfy.model_sampling.ModelSamplingDiscreteFlow()
    model = comfy.model_patcher.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    control = architecture.MossTileControl(32, 2, 8, condition_dim=8)
    holder = comfy.model_patcher.ModelPatcher(control, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    vae = Mock()
    vae.encode.return_value = torch.randn(1, 128, 2, 3)
    image = torch.rand(1, 32, 48, 3)
    return model, holder, vae, image


class NodeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def test_zero_returns_exact_model_without_encoding(self):
        model, control, vae, image = fixture()
        self.assertIs(nodes.MossTileControlApply.execute(model, control, vae, image, 0).result[0], model)
        vae.encode.assert_not_called()
        self.assertFalse(model.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL))

    def test_reject_pending_legacy_projection_before_encoding(self):
        model, control, vae, image = fixture()
        model.add_patches({"diffusion_model.video_embedder.weight": ("set", (torch.randn(32, 384),))})
        self.assertEqual(model.model.diffusion_model.video_embedder.weight.shape[1], 128)
        with self.assertRaisesRegex(ValueError, "legacy concat tile LoRA"):
            nodes.MossTileControlApply.execute(model, control, vae, image)
        vae.encode.assert_not_called()

    def test_apply_after_legacy_clone_has_resized_shared_projection(self):
        model, control, vae, image = fixture()
        key = "diffusion_model.video_embedder.weight"
        legacy = model.clone()
        legacy.add_patches({key: ("set", (torch.randn(32, 384),))})
        legacy.patch_weight_to_device(key, device_to=torch.device("cpu"))
        self.assertEqual(model.model.diffusion_model.video_embedder.weight.shape, (32, 384))
        self.assertEqual(model.backup[key].weight.shape, (32, 128))

        result = nodes.MossTileControlApply.execute(model, control, vae, image).result[0]

        vae.encode.assert_called_once()
        self.assertIsNot(result, model)
        self.assertEqual(model.model.diffusion_model.video_embedder.weight.shape, (32, 384))
        with self.assertRaisesRegex(ValueError, "legacy concat tile LoRA"):
            nodes.MossTileControlApply.execute(legacy, control, vae, image)

    def test_residual_allocations_are_outside_backbone_graph(self):
        _, control, _, _ = fixture()
        controller = nodes.MossTilePatch(control, torch.randn(1, 128, 2, 3), 1, 1, 0)
        block = nodes.MossTileBlockPatch(controller, 0, None)
        x = torch.randn(1, 8, 32)
        condition = torch.randn(1, 6, 8)
        paused = False
        @contextmanager
        def pause():
            nonlocal paused
            paused = True
            try:
                yield
            finally:
                paused = False
        def base(args):
            self.assertFalse(paused)
            return {"img": args["img"]}
        forward = control.model.blocks[0].forward
        def residual(*args):
            self.assertTrue(paused)
            return forward(*args)
        with patch.object(nodes.comfy.model_prefetch, "pause_malloc_graph", pause), patch.object(control.model.blocks[0], "forward", residual):
            out = block({"img": x, "transformer_options": {nodes.CONTROL_KEY: {controller: condition}}}, {"original_block": base})
        self.assertFalse(paused)
        self.assertTrue(torch.equal(out["img"], forward(x, condition, 1)))

    def test_schedule_batch_and_patch_composition(self):
        model, control, vae, image = fixture()
        before = {k: v.clone() for k, v in model.model.diffusion_model.state_dict().items()}
        def previous(args, extra):
            out = extra["original_block"](args)
            return {"img": out["img"] + 0.25}
        model.set_model_patch_replace(previous, "dit", "single_block", 0)
        result = nodes.MossTileControlApply.execute(model, control, vae, image, 0.5, 0.2, 0.8).result[0]
        self.assertIsNot(result, model)
        self.assertEqual(result.get_additional_models_with_key(nodes.CONTROL_KEY), [control])
        self.assertIs(model.model_options["transformer_options"]["patches_replace"]["dit"][("single_block", 0)], previous)
        wrapper = result.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL)[0]
        block = result.model_options["transformer_options"]["patches_replace"]["dit"][("single_block", 0)]
        x = torch.randn(2, 8, 32)
        options = {"sigmas": torch.tensor([0.5])}
        observed = {}
        def execute(latent, time, context, transformer_options, **kwargs):
            observed.update(transformer_options)
            return block({"img": x, "transformer_options": transformer_options},
                         {"original_block": lambda args: {"img": args["img"]}})["img"]
        out = wrapper(execute, torch.randn(2, 128, 2, 3), torch.ones(2), None, options)
        condition = observed[nodes.CONTROL_KEY][wrapper]
        self.assertEqual(condition.shape, (2, 6, 8))
        self.assertTrue(torch.equal(out, control.model.blocks[0](x + 0.25, condition, 0.5)))
        self.assertNotIn(nodes.CONTROL_KEY, options)
        options["sigmas"] = torch.tensor([0.99])
        self.assertTrue(torch.equal(wrapper(execute, torch.randn(2, 128, 2, 3), torch.ones(2), None, options), x + 0.25))
        self.assertFalse(hasattr(wrapper, "condition"))
        for key, value in model.model.diffusion_model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_loader_roundtrip_and_reject_legacy(self):
        _, control, _, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tile.safetensors"
            state = {architecture.PREFIX + k: v for k, v in control.model.state_dict().items()}
            safetensors.torch.save_file(state, path, metadata=control.model.metadata())
            with patch.object(nodes.folder_paths, "get_full_path_or_raise", return_value=str(path)):
                loaded = nodes.MossTileControlLoader.execute("tile.safetensors").result[0]
            for key, value in control.model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.model.state_dict()[key]))
            safetensors.torch.save_file({"diffusion_model.video_embedder.set_weight": torch.ones(32, 384)}, path, metadata={"format": "pt"})
            with patch.object(nodes.folder_paths, "get_full_path_or_raise", return_value=str(path)):
                with self.assertRaises(ValueError):
                    nodes.MossTileControlLoader.execute("tile.safetensors")

    def test_cfg_batch_preserves_control_image_order(self):
        _, control, _, _ = fixture()
        latent = torch.randn(2, 128, 2, 3)
        wrapper = nodes.MossTilePatch(control, latent, 1, 1, 0)
        def execute(x, t, context, options):
            return options[nodes.CONTROL_KEY][wrapper]
        condition = wrapper(execute, torch.zeros(4, 128, 2, 3), None, None)
        expected = control.model.conditioner(latent)
        self.assertTrue(torch.equal(condition[:2], expected))
        self.assertTrue(torch.equal(condition[2:], expected))

    def test_real_moss_forward_uses_patches_and_zero_schedule(self):
        model, control, vae, image = fixture()
        result = nodes.MossTileControlApply.execute(model, control, vae, image, 0.5, 0.2, 0.8).result[0]
        backbone = model.model.diffusion_model
        x, t, context = torch.randn(1, 128, 2, 3), torch.tensor([0.5]), torch.randn(1, 4, 16)
        def forward(patcher, sigma):
            options = comfy.patcher_extension.copy_nested_dicts(patcher.model_options["transformer_options"])
            options["sigmas"] = torch.tensor([sigma])
            options["wrappers"] = patcher.wrappers
            return backbone(x, t, context, transformer_options=options)
        # Exercise the differentiable CPU attention path; CUDA inference is checked separately.
        with patch.object(comfy.model_management, "in_training", True):
            baseline = forward(model, 0.99)
            outside = forward(result, 0.99)
            active = forward(result, 0.5)
        self.assertTrue(torch.equal(baseline, outside))
        self.assertFalse(torch.equal(baseline, active))
        self.assertEqual(active.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
