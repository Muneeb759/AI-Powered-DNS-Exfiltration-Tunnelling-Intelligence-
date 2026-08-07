.PHONY: verify audit build_splits ablation latency test phase1 handoff

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
