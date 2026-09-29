"""Pre-training equivalence and numerical checks for the new conditioning models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "work"), str(ROOT / "followup")]

import torch
import orbitlab as o
from common import state_digest
from models import from_variant, guided
from p0 import dump, sha


def main():
    out = HERE / "reports/p1_model_preflight_v1.json"
    if out.exists():
        report = json.loads(out.read_text())
        assert report["models_source_sha256"] == sha(HERE / "models.py")
        print(json.dumps(report, ensure_ascii=False))
        return
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(770151)
    z = torch.randn(16, 4, 16, generator=generator)
    t = torch.rand(16, generator=generator)
    labels = torch.tensor([[i % 4, i % 6] for i in range(16)])
    torch.manual_seed(123)
    old = o.Velocity(16, True)
    torch.manual_seed(123)
    new = from_variant("F0")
    initial_identical = all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    baseline_max_abs = float((old(z, t, labels) - new(z, t, labels)).abs().max())
    models = {}
    for variant in ("F0", "F1", "F2", "F3"):
        torch.manual_seed(81000)
        model = from_variant(variant)
        c4 = float((model(o.rho(z, 1), t, labels) - o.rho(model(z, t, labels), 1)).abs().max())
        blank = torch.full_like(labels, -1)
        unconditional_finite = bool(torch.isfinite(model(z, t, blank)).all())
        guided_finite = bool(torch.isfinite(guided(model, z, t, labels, 2)).all())
        y = labels.clone()
        if variant in ("F2", "F3"):
            y[[0, 5, 9]] = -1
        model(z, t, y).square().mean().backward()
        gradients_finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        film_grad_nonzero = None
        if variant in ("F1", "F3"):
            film_grad_nonzero = float(model.film1.weight.grad.abs().sum() + model.film2.weight.grad.abs().sum()) > 0
        models[variant] = {"parameters": sum(p.numel() for p in model.parameters()),
                           "initial_state_sha256": state_digest(model),
                           "c4_max_abs": c4, "unconditional_finite": unconditional_finite,
                           "guided_finite": guided_finite, "gradients_finite": gradients_finite,
                           "film_gradient_nonzero": film_grad_nonzero}
    matched_concat_init = models["F0"]["initial_state_sha256"] == models["F2"]["initial_state_sha256"]
    matched_film_init = models["F1"]["initial_state_sha256"] == models["F3"]["initial_state_sha256"]
    parameter_increase = models["F1"]["parameters"] / models["F0"]["parameters"] - 1
    passed = bool(initial_identical and baseline_max_abs == 0 and matched_concat_init and matched_film_init and
                  parameter_increase < .10 and
                  all(r["c4_max_abs"] < 1e-5 and r["unconditional_finite"] and r["guided_finite"] and
                      r["gradients_finite"] and r["film_gradient_nonzero"] is not False for r in models.values()))
    report = {"passed": passed, "baseline_initial_weights_identical": initial_identical,
              "baseline_output_max_abs": baseline_max_abs,
              "matched_concat_initializations": matched_concat_init,
              "matched_film_initializations": matched_film_init,
              "film_parameter_increase_fraction": parameter_increase,
              "models": models, "models_source_sha256": sha(HERE / "models.py"),
              "torch_version": torch.__version__, "device": "cpu",
              "scope": "Implementation and numerical preflight, not a generated-image success result."}
    dump(out, report)
    if not passed:
        raise AssertionError("Conditioning model preflight failed")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
