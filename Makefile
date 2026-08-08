.PHONY: verify audit build_splits ablation latency test phase1 handoff reproduce reproduce-full app bundle

# --- Reproduce the reported headline score (challenge brief deliverable #2) -------------
# Verifies the committed model against report/TECHNICAL_REPORT.md section 7 and exits
# non-zero on mismatch. Requires data/raw (see README) and data/splits.
reproduce:
	python scripts/reproduce.py

# Same, but rebuilds splits and retrains from data/raw first.
reproduce-full:
	python scripts/reproduce.py --retrain

# --- Phase 1 data pipeline --------------------------------------------------------------
verify:
	python src/data/verify_snapshot.py

audit:
	python src/data/audit_snapshot.py

build_splits:
	python src/data/build_splits.py

ablation:
	python src/eval/ablation.py

latency:
	python src/latency/harness.py

test:
	python -m pytest

handoff:
	python scripts/handoff_check.py

phase1: verify audit build_splits ablation latency test handoff
	@echo "Phase 1 pipeline completed successfully!"

# --- Deliverables -----------------------------------------------------------------------
bundle:
	python -m src.deliverables.results_bundle

app:
	streamlit run app/streamlit_app.py
