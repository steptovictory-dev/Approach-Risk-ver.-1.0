"""Binary GoA classification: fixed logistic L2 baseline, no tuning."""
from train_primary import main
if __name__=='__main__':
    print('Method: supervised binary classification; logistic L2 C=1; no aircraft metadata.',flush=True)
    main(aircraft=False,prefix='05_no_aircraft',folder='no_aircraft_models_v1')
    print('Method: supervised binary classification; logistic L2 C=1; 24h gap mask.',flush=True)
    main(gap=True,prefix='05_gap',folder='gap_models_v1')
