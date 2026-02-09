from types import SimpleNamespace

from protein_design_hub.analysis.mutation_scanner import MutationScanner


def test_mutation_scanner_adds_ost_comprehensive_metrics(monkeypatch, tmp_path):
    import protein_design_hub.evaluation.composite as composite

    class DummyEvaluator:
        def __init__(self, metrics):
            self.metrics = metrics

        def evaluate(self, model_path, reference_path=None):
            return SimpleNamespace(metadata={})

        def evaluate_comprehensive(self, model_path, reference_path):
            return {"global": {"lddt": 0.91, "rmsd_ca": 1.25}}

    monkeypatch.setattr(composite, "CompositeEvaluator", DummyEvaluator)

    scanner = MutationScanner(
        predictor="esmfold_api",
        evaluation_metrics=[],
        run_openstructure_comprehensive=True,
    )
    scanner._metrics_available = False

    model = tmp_path / "model.pdb"
    ref = tmp_path / "ref.pdb"
    model.write_text("MODEL\n")
    ref.write_text("MODEL\n")

    metrics = scanner.calculate_biophysical_metrics(model, ref, evaluation_metrics=[])

    assert "extra_metrics" in metrics
    assert metrics["extra_metrics"]["ost_comprehensive"]["global"]["lddt"] == 0.91
    assert metrics["extra_metrics"]["ost_comprehensive"]["global"]["rmsd_ca"] == 1.25


def test_mutation_scanner_skips_ost_metrics_when_disabled(monkeypatch, tmp_path):
    import protein_design_hub.evaluation.composite as composite

    class DummyEvaluator:
        def __init__(self, metrics):
            self.metrics = metrics

        def evaluate(self, model_path, reference_path=None):
            return SimpleNamespace(metadata={})

        def evaluate_comprehensive(self, model_path, reference_path):
            raise AssertionError("evaluate_comprehensive should not be called")

    monkeypatch.setattr(composite, "CompositeEvaluator", DummyEvaluator)

    scanner = MutationScanner(
        predictor="esmfold_api",
        evaluation_metrics=[],
        run_openstructure_comprehensive=False,
    )
    scanner._metrics_available = False

    model = tmp_path / "model.pdb"
    ref = tmp_path / "ref.pdb"
    model.write_text("MODEL\n")
    ref.write_text("MODEL\n")

    metrics = scanner.calculate_biophysical_metrics(model, ref, evaluation_metrics=[])

    assert metrics.get("extra_metrics", {}).get("ost_comprehensive") is None
