"""CPU regression for the keyframe posterior adapter's RNG handoff.

Extract the entry's wrapper without importing/initializing a native GPU engine.
The test supplies the same posterior API with Diffusers' CPU Gaussian class.
"""
import ast
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch


class DiagonalGaussianDistribution:
    def __init__(self, moments):
        self.mean, self.logvar = moments.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator=None):
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * noise


def test_keyframe_sampling_uses_the_callers_generator():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'InfiniLM/examples/minimax_h3.py').read_text())
    installer = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == 'install_component_profilers'
    )

    class Profiler:
        def __init__(self, *args):
            pass

        def wrap(self, *args):
            pass

    class Model:
        use_3d_conv = True
        calls = 0

        def _adaptive_encode(self, pixels):
            self.calls += 1
            return torch.cat((torch.ones(1, 24, 1, 2, 3), torch.zeros(1, 24, 1, 2, 3)), dim=1)

        def trim_code(self, latents, count):
            return latents[:, :, :count]

    class Vae:
        def __init__(self):
            self.model = Model()

        def parameters(self):
            return iter([torch.ones(1)])

        def encode(self, *args):
            raise AssertionError('Expected the adapter to replace this method')

        def decode(self, *args):
            pass

    class Component:
        pass

    conditioner = Component()
    conditioner.model = conditioner
    scope = dict(torch=torch, TimingReport=object, Any=object, ComponentProfiler=Profiler)
    exec(compile(ast.Module(body=[installer], type_ignores=[]), '<vae-adapter-test>', 'exec'), scope)
    fake_bundle = ModuleType('video_vae')
    fake_distribution = ModuleType('video_vae.vae_module')
    fake_distribution.DiagonalGaussianDistribution = DiagonalGaussianDistribution
    vae = Vae()
    with patch.dict('sys.modules', {'video_vae': fake_bundle, 'video_vae.vae_module': fake_distribution}):
        scope['install_component_profilers'](
            object(), text_encoder=conditioner, transformer=Component(),
            vae=vae, audio_vae=Component(),
        )
        posterior = vae.encode(torch.ones(1, 3, 1, 32, 32), return_dict=False)[0]
        global_state = torch.get_rng_state().clone()
        first = posterior.sample(generator=torch.Generator().manual_seed(42))
        assert torch.equal(global_state, torch.get_rng_state())
        torch.randn(500)  # An unrelated component may consume the global RNG.
        second = posterior.sample(generator=torch.Generator().manual_seed(42))
        other_seed = posterior.sample(generator=torch.Generator().manual_seed(43))
        assert torch.equal(first, second)
        assert not torch.equal(first, other_seed)
        assert first.shape == (1, 24, 1, 2, 3)
        assert torch.equal(posterior.mode(), torch.ones_like(first))
        assert vae.model.calls == 1
