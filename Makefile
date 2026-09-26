PY ?= python3
MODEL = results/07_final_model/ensemble_raw_activity_max_unweighted.cbm

.PHONY: all test selection model evaluation indels calibration figures clean

all: selection model evaluation indels calibration figures

test:
	$(PY) -m pytest tests -q

## model selection — training partition only, nothing below touches the test set
selection:
	$(PY) scripts/02_representation.py
	$(PY) scripts/03_fm_combination.py
	$(PY) scripts/04_classifier_comparison.py
	$(PY) scripts/05_hyperparameters.py
	$(PY) scripts/06_per_tool_calibration.py

## the one frozen model, and the out-of-fold predictions it is calibrated on
model:
	$(PY) scripts/07_train_final_model.py

## everything below applies that model; it is never refitted
evaluation:
	$(PY) scripts/08_evaluate.py --cohort internal --model $(MODEL)
	$(PY) scripts/08_evaluate.py --cohort external --model $(MODEL)
	$(PY) scripts/09_screening.py
	$(PY) scripts/10_evo2_size.py

indels:
	$(PY) scripts/07_train_final_model.py --dataset indel_train --name indel
	$(PY) scripts/08_evaluate.py --cohort indel --model results/07_final_model/ensemble_indel.cbm
	$(PY) scripts/08_evaluate.py --cohort indel_external --model results/07_final_model/ensemble_indel.cbm

calibration:
	$(PY) scripts/11_acmg_calibration.py
	$(PY) scripts/12_reclassification.py
	$(PY) scripts/12_reclassification.py --model results/07_final_model/ensemble_indel.cbm \
		--variant-type indel --name reclassification_indel
	$(PY) scripts/15_predictive_values.py --oof results/07_final_model/oof_raw_unweighted.npz \
		--model $(MODEL)

figures:
	$(PY) scripts/13_figures.py --model results/07_final_model/ensemble_indel.cbm
	$(PY) scripts/16_shap.py --model $(MODEL)
	$(PY) scripts/17_importance.py --model $(MODEL)
	$(PY) scripts/17_importance.py --model results/07_final_model/ensemble_indel.cbm \
		--dataset indel_test --name importance_indel

clean:
	rm -rf results/*
