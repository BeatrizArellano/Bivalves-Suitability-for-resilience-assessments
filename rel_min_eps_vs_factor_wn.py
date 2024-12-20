"""
This script explores the relationship between the noise power of independent measurements and
metrics such as EPS and rbar
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample


def get_rbar(df,wL=50,method='pearson',min_obs=10):
    """
    Gets the mean interseries correlation between all series from different individuals
    """
    time = df.index.values
    rbar = pd.Series(index=time)
    for t in np.arange(time[0],time[-wL]):
        corrs = df.loc[t:t+wL].corr(method=method,min_periods=min_obs)
        #corrs = corrs.replace({1: np.nan}) 
        ## Removes the diagonal of the correlation matrix
        np.fill_diagonal(corrs.values, np.nan)
        ## JUst preserving the uuper triangle of the correlation matrix
        corrs = corrs.where(np.triu(np.ones(corrs.shape)).astype(np.bool))
        #corrs_arr = corrs.to_numpy().flatten() #np.nanmean(corrs_arr)
        rbar.loc[t+wL] = corrs.mean().mean() ### Gets the mean correlation
    return rbar

def get_eps(df,wL=50,min_obs=10):
    """
    Estimates the Expressed Population Signal using the average number of individuals per year
    """
    time = df.index.values
    rbar = pd.Series(index=time)
    eps = pd.Series(index=time)
    for t in np.arange(time[0],time[-wL]):
        roll_df = df.loc[t:t+wL]
        corrs = roll_df.corr(min_periods=min_obs)
        np.fill_diagonal(corrs.values, np.nan)
        ## JUst preserving the uuper triangle of the correlation matrix
        corrs = corrs.where(np.triu(np.ones(corrs.shape)).astype(np.bool))
        rbar = corrs.mean().mean()
        ### obtaining the average number of trees for each year
        #N = len(roll_df.count()[roll_df.count()>=min_obs])
        N = np.round(roll_df.count(axis=1).mean(),0)
        #eps.loc[t+wL] = N * rbar / ((N-1)*rbar+1)
        eps.loc[t+wL] = (N * np.abs(rbar)) / (((N-1)*np.abs(rbar))+1)
    return eps

def get_snr(df,wL=50,min_obs=10):
    """
    Estimates the signal to noise ratio
    """
    time = df.index.values
    rbar = pd.Series(index=time)
    snr = pd.Series(index=time)
    for t in np.arange(time[0],time[-wL]):
        roll_df = df.loc[t:t+wL]
        corrs = roll_df.corr(min_periods=min_obs)
        np.fill_diagonal(corrs.values, np.nan)
        ## JUst preserving the uuper triangle of the correlation matrix
        corrs = corrs.where(np.triu(np.ones(corrs.shape)).astype(np.bool))
        rbar = corrs.mean().mean()
        #N = len(roll_df.count()[roll_df.count()>=min_obs])
        N = np.round(roll_df.count(axis=1).mean(),0)
        snr.loc[t+wL] = N * rbar / (1-rbar)
    return snr

def create_inc_n_chron(ts_len=300,min_n=10,max_n=30,min_len=60,last_year=2022,std=1):
    """
    Creates a dataframe with independing white-noise time-series introduced successively
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

def get_min_eps_vs_ar1_noise(df_like,repetitions=10,max_power_wn=8,res_power_wn=0.5,std=1,min_obs=30):
    """
    Gets the minimum value of rbar, EPS and SNS for a given combination of power noise and AR(1) coefficient
    """
    noise_factors = np.round(np.concatenate([np.arange(0,1,0.1),np.arange(1,max_power_wn,res_power_wn)]),1)
    ar1cs = np.round(np.arange(-0.9,1,0.1),1)
    idx = pd.MultiIndex.from_product([ar1cs,noise_factors],names=['AR1c', 'noise_power'])
    rbars = pd.DataFrame(index=idx,columns=list(range(repetitions)))
    epss = pd.DataFrame(index=idx,columns=list(range(repetitions)))
    #snrs = pd.DataFrame(index=idx,columns=list(range(repetitions)))
    for ar1c in ar1cs:
        for noise_power in noise_factors:
            i = 0
            while i < repetitions:
                try:
                    print(f'AR1: {ar1c}, Noise: {noise_power}, {i}',end='\r')
                    synth_df = create_synth_ind_meas(df_like,ar1c=ar1c,noise_power=noise_power,std=std)
                    rbars.loc[(ar1c,noise_power),i] = get_rbar(synth_df,min_obs=min_obs).min()
                    epss.loc[(ar1c,noise_power),i] = get_eps(synth_df,min_obs=min_obs).min()
                    #snrs.loc[(ar1c,noise_power),i] = get_snr(synth_df,min_obs=min_obs).min()
                    i += 1
                except:
                    continue
    return rbars, epss#, snrs

#### Reading the available individual shell measurements

ice_shells = pd.read_csv('output/raw_ind_shells_ICE_PB.csv',index_col=0) ##Iceland
iom_shells = pd.read_csv('output/raw_ind_shells_IS_PB.csv',index_col=0)

start_year = 1750
ice_shells = ice_shells[ice_shells.index>=start_year]
iom_shells = iom_shells[iom_shells.index>=start_year]

len_ts = 250
n_1_to_10 = create_inc_n_chron(ts_len=len_ts,min_n=1,max_n=10,last_year=2023)
n_10_to_30 = create_inc_n_chron(ts_len=len_ts,min_n=10,max_n=30,last_year=2023)
n_10_to_50 = create_inc_n_chron(ts_len=len_ts,min_n=10,max_n=50,last_year=2023)

repetitions = 300

output_path = 'output/'

test_ind_series = {'IS_PB':iom_shells,
               'ICE_PB':ice_shells,
                'n_1_to_10':n_1_to_10,
               'n_10_to_30':n_10_to_30}

for chid, df_ind_ts in test_ind_series.items():
    print('\n'+chid)
    rbar,eps = get_min_eps_vs_ar1_noise(df_ind_ts,repetitions=repetitions)
    print('\nSaving csvs')
    rbar.to_csv(f'output/rbar_vs_noise_{chid}_n{repetitions}.csv')
    eps.to_csv(f'output/eps_vs_noise_{chid}_n{repetitions}.csv')
    #snr.to_csv(f'output/snr_vs_noise_{chid}.csv')