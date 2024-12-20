"""
This script explores trends in AR(1) and variance for varying values in the Expressed Population Signal (EPS) in
the individual series and varying AR(1) in the background environmental signal. 
"""
from astropy import stats
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample
import sys
sys.path.append('../../lib/')
from regimeshifts import ews

def create_inc_n_chron(ts_len=300,min_n=10,max_n=30,min_len=60,last_year=2022,std=1):
    """
    Creates a dataframe with independent white-noise time-series introduced successively
    """
    var_sample_depth = np.round(np.linspace(min_n,max_n,ts_len-min_len),0)
    var_sample_depth = np.concatenate([var_sample_depth,np.tile(max_n,min_len)])
    sample_depth_ts = pd.Series(index=np.arange(last_year-ts_len,last_year),data=var_sample_depth)
    ind_synth_ts = []
    for j,sd in enumerate(np.unique(var_sample_depth)):
        len_ind_ts = len(sample_depth_ts[sample_depth_ts>=sd])
        start_year = sample_depth_ts[sample_depth_ts==sd].idxmax()
        n_ind_meas = min_n if j == 0 else 1
        for rep in range(n_ind_meas):
            ind_ts = pd.Series(index=range(start_year,start_year+len_ind_ts),data=np.random.normal(0,std,len_ind_ts))
            ind_synth_ts.append(ind_ts)
    chron_synth = pd.concat(ind_synth_ts, axis=1, sort=False)
    return chron_synth

def create_synth_ind_meas(like_df,ar1c=0,noise_power=1,std=1):
    """
    Creates synthetic independent measurements from a random environmental signal
    like_df: Pandas dataframe
            Dataframe containing the structure of the independent measurements
    ar1c: float
          AR(1) coefficient to model the environmental signal 
    noise_power: float
               Amount of white noise added to each independent series
    std: float
        Standard deviation of the environmental signal and white noise added to individual series
    """
    ind_synth_ts = []
    arma_noise = arma_generate_sample(ar=[1,-ar1c],ma=[1],nsample=len(like_df),scale=std,burnin=3)
    env_ts = pd.Series(index=like_df.index,data=arma_noise) ## Environmental signal
    for shell in like_df.columns:
        orig_ts = like_df[shell].dropna()
        white_noise = pd.Series(index=orig_ts.index,data=np.random.normal(0,std,len(orig_ts)))
        synth_shell = env_ts.loc[orig_ts.index] + (white_noise * noise_power)
        ind_synth_ts.append(synth_shell)
    chron_synth = pd.concat(ind_synth_ts, axis=1, sort=False)
    return chron_synth

def ews_vs_eps(df_like,rel_noise_eps_df,repetitions=10,std=1,bW=35,wL=None):
    """
    Obtains the AR(1) and variance series from an ensemble of synthetic measurements
    """
    
    ar1cs = np.round(np.arange(-0.9,1,0.1),1)
    if not wL:
        wL = int(len(df_like)/2)
    eps_values = np.round(np.arange(0,1.05,0.05),2) ## Standard EPS values
    eps_rel = round(rel_noise_eps_df.median(axis=1) * 20) / 20
    ## Obtaining the relationship between EPS and noise power for each particular dataset
    eps_rel_int = eps_rel.reset_index().rename(columns={0:'EPS'}).groupby(['AR1c','EPS']).mean()['noise_power'].unstack().interpolate(axis=1).round(1)
    idx_ts = pd.MultiIndex.from_product([ar1cs,eps_values,df_like.index],names=['AR1c', 'EPS','Year'])
    idx_c = pd.MultiIndex.from_product([ar1cs,eps_values],names=['AR1c', 'EPS'])
    ar1_ts = pd.DataFrame(index=idx_ts,columns=list(range(repetitions)))
    var_ts = pd.DataFrame(index=idx_ts,columns=list(range(repetitions)))
    kv_ar1_df = pd.DataFrame(index=idx_c,columns=list(range(repetitions)))
    kv_var_df = pd.DataFrame(index=idx_c,columns=list(range(repetitions)))
    corr_ar1_var_df = pd.DataFrame(index=idx_c,columns=list(range(repetitions)))
    for ar1c in ar1cs:
        ### Relationship between Noise power and EPS for each AR(1) value
        rel_eps_np = eps_rel_int.loc[ar1c].dropna()
        rel_eps_np = rel_eps_np[rel_eps_np.index>=eps_values[0]]        
        for eps,noise_power in rel_eps_np.iteritems():
            for i in range(repetitions):
                synth_df = create_synth_ind_meas(df_like,ar1c=ar1c,noise_power=noise_power,std=std)
                avg_synth_ts = ews.Ews(synth_df.apply(stats.biweight_location, axis = 1,c=9,ignore_nan=True))
                ar1 = avg_synth_ts.ar1(detrend=True,wL=wL,bW=bW)
                var = avg_synth_ts.var(detrend=True,wL=wL,bW=bW)
                kv_ar1_df.loc[(ar1c,eps),i] = ar1.kendall                
                kv_var_df.loc[(ar1c,eps),i] = var.kendall
                corr_ar1_var_df.loc[(ar1c,noise_power),i] = ar1[0].corr(var[0])
                ar1_ts.loc[(ar1c,eps),i] = ar1.values
                var_ts.loc[(ar1c,eps),i] = var.values             
    
    return ar1_ts, var_ts, kv_ar1_df, kv_var_df, corr_ar1_var_df

#### Reading the available individual shell measurements

ice_shells = pd.read_csv('output/raw_ind_shells_ICE_PB.csv',index_col=0) ##Iceland
iom_shells = pd.read_csv('output/raw_ind_shells_IS_PB.csv',index_col=0)

### Reading dataframes containing the relationship between EPS and Noise power

iom_rel_eps = pd.read_csv('output/eps_vs_noise_IS_PB_n300.csv',index_col=[0,1])
ice_rel_eps = pd.read_csv('output/eps_vs_noise_ICE_PB_n300.csv',index_col=[0,1])
rel_eps_10_30 = pd.read_csv('output/eps_vs_noise_n_10_to_30_n300.csv',index_col=[0,1])
rel_eps_1_10 = pd.read_csv('output/eps_vs_noise_n_1_to_10_n300.csv',index_col=[0,1])

start_year = 1750
ice_shells = ice_shells[ice_shells.index>=start_year]
iom_shells = iom_shells[iom_shells.index>=start_year]

len_ts = 250

n_1_to_10 = create_inc_n_chron(ts_len=len_ts,min_n=1,max_n=10,last_year=2023)
n_10_to_30 = create_inc_n_chron(ts_len=len_ts,min_n=10,max_n=30,last_year=2023)


repetitions = 1000


rel_eps_np_dfs = {'IS_PB':iom_rel_eps,
                   'ICE_PB':ice_rel_eps,
                   'n_10_to_30':rel_eps_10_30,
                   'n_1_to_10':rel_eps_1_10}

for chid, df_ind_ts in test_ind_series.items():
    print(chid)
    ## Reading dataframe with the relationship between Noise Power and EPS
    rel_df = rel_eps_np_dfs[chid]    
    ar1_ts, var_ts, kv_ar1_df, kv_var_df, corr_ar1_var_df = ews_vs_eps(df_ind_ts,rel_df,repetitions=repetitions)
    print('\tSaving csvs')
    ar1_ts.to_csv(f'output/ar1_vs_eps_ts_{chid}.csv')
    var_ts.to_csv(f'output/var_vs_eps_ts_{chid}.csv')
    kv_ar1_df.to_csv(f'output/ar1_vs_eps_kv_{chid}.csv')
    kv_var_df.to_csv(f'output/var_vs_eps_kv_{chid}.csv')
    corr_ar1_var_df.to_csv(f'output/eps_corr_var_ar1_{chid}.csv')