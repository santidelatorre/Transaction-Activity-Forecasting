import json
import time
import numpy as np
import pandas as pd
from ubs_recurrence.data import ROOT,LABELS,aligned_target
from ubs_recurrence.evaluation import record
from ubs_recurrence.decision import apply_bias,optimize_bias


MEMBERS=["robust_augTrue_droptextFalse_noise", "robust_augTrue_droptextFalse_noise_soft", "robust_augTrue_droptextFalse_noise_soft_xgb"]


def main():
    start=time.perf_counter();sets={}
    ids=None
    for scenario in ["original","valid_like","test_like"]:
        probabilities=[]
        for member in MEMBERS:
            d=pd.read_csv(ROOT/f"outputs/experiments/{member}_{scenario}_s42/predictions.csv")
            if ids is None:ids=d.client_id.to_numpy()
            assert np.array_equal(ids,d.client_id.to_numpy())
            p=d[["p_"+f for f in LABELS]].to_numpy();probabilities.append(p/p.sum(axis=1,keepdims=True))
        sets[scenario]=np.mean(probabilities,axis=0)
        y=aligned_target(ids)
        record("ensemble_equal3_"+scenario,ids,y,sets[scenario],runtime=time.perf_counter()-start,metadata={"hypothesis":"Different candidate assignment and boosting algorithms have complementary errors","features":MEMBERS,"model":"Equal probability blend","parameters":{"weights":[1/3]*3},"seed":42,"protocol":"Fixed equal blend of genuinely OOF base predictions; same client folds; scenario="+scenario,"status":"evaluated","conclusion":"Fixed ensemble before official holdout access"})
    bias,history=optimize_bias([sets["valid_like"],sets["test_like"]],y)
    config={"members":MEMBERS,"weights":[1/3]*3,"seed":42,"bias":bias.tolist(),"bias_training_scenarios":["valid_like","test_like"],"search_history":history,"caveat":"Bias fitted on train-side OOF labels; calibrated development F1 is not an independent OOF estimate. Official holdout remains untouched."}
    (ROOT/"reports/frozen_candidate_v1.json").write_text(json.dumps(config,indent=2))
    for scenario,p in sets.items():
        record("ensemble_equal3_biasfit_"+scenario,ids,y,apply_bias(p,bias),runtime=time.perf_counter()-start,split="oof_decision_fit_not_independent",metadata={"hypothesis":"Training-OOF class biases can improve macro-F1 decision allocation","features":MEMBERS,"model":"Equal blend plus OOF-fitted log biases","parameters":config,"seed":42,"protocol":"Decision biases fitted on training OOF; these scores are development fit diagnostics, not unbiased validation","status":"development_only","conclusion":"Judge generalization on sealed official holdout, not these tuned scores"})
    print("Frozen biases",bias,flush=True)


if __name__=="__main__":main()
