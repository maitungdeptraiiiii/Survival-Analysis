import pandas as pd
from lifelines import CoxPHFitter

df = pd.read_csv("cohort_hf_survival_90d.csv")
df['gender_male'] = df['gender'].map({'F': 0, 'M': 1})

for var in ['anchor_age', 'gender_male', 'cci_without_hf']:
    df_model = df[['time_to_event_90d', 'death_90d', var]].dropna().copy()
    
    cph = CoxPHFitter()
    cph.fit(df_model, duration_col='time_to_event_90d', event_col='death_90d')
    
    print(f"\n===== {var} =====")
    cph.print_summary()