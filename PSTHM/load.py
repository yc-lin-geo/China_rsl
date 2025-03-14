#----------------------Define Functions---------------------------
import numpy as np
import torch
import pandas as pd
import os
import torch
import zipfile
from scipy import interpolate

def load_local_rsl_data(file):
    '''
    A function to load rsl data from a csv file, this csv 
    file should be presented in the same format as NJ_CC.csv in data folder

    ---------Inputs---------
    file: str, the path to the csv file

    ---------Outputs---------
    X: torch.tensor, the age of rsl data
    y: torch.tensor, the rsl data
    y_sigma: torch.tensor, one sigma uncertainty of rsl data
    x_sigma: torch.tensor, one uncertainty of age data
    '''

    #load data
    data = pd.read_csv(file)
    rsl = data['RSL']
    rsl_2sd =( data['RSLer_up_2sd']+data['RSLer_low_2sd'])/2 #average up and low 2std
    rsl_age = -(data['Age']-1950) #convert age from BP to CE
    rsl_age_2sd = (data['Age_low_er_2sd']+data['Age_up_er_2sd'])/2 #average up and low 2std
    rsl_lon = data['Longitude']
    rsl_lat = data['Latitude']

    #convert RSL data into tonsors
    X = torch.tensor(rsl_age).flatten() #standardise age
    y = torch.tensor(rsl).flatten()
    y_sigma = torch.tensor(rsl_2sd/2).flatten()
    x_sigma = torch.tensor(rsl_age_2sd/2).flatten()

    return X,y,y_sigma,x_sigma,rsl_lon,rsl_lat

def load_regional_rsl_data(file,CE=False):
    '''
    A function to load rsl data from a csv file, this csv 
    file should be presented in the same format as US_Atlantic_Coast_for_ESTGP.csv in data folder

    ---------Inputs---------
    file: str, the path to the csv file

    ---------Outputs---------
    marine_limiting: a list containing marine limiting data (details below)
    SLIP: a list containing sea-level index point data
    terrestrial limiting: a list containing terrestrial limiting data

    data within each list:
    X: torch.tensor, the age of rsl data
    y: torch.tensor, the rsl data
    y_sigma: torch.tensor, one sigma uncertainty of rsl data
    x_sigma: torch.tensor, one uncertainty of age data
    lon: numpy.array, 
    rsl_region: a number indicating the region where data locates at
    '''

    #load data
    data = pd.read_csv(file)
    rsl = data['RSL']
    rsl_2sd =( data['RSLer_up_2sd']+data['RSLer_low_2sd'])/2 #average up and low 2std
    if CE:
        rsl_age = -(data['Age']-1950) #convert age from BP to CE
    else:
        rsl_age = data['Age']
    rsl_age_2sd = (data['Age_low_er_2sd']+data['Age_up_er_2sd'])/2 #average up and low 2std
    rsl_lon = data['Longitude']
    rsl_lat = data['Latitude']
    rsl_region = data['Region.1']
    rsl_limiting = data['Limiting']
    marine_index, SLIP_index, terrestrial_index = rsl_limiting==-1, rsl_limiting==0, rsl_limiting==1

    #convert RSL data into tonsors
    X = torch.tensor(rsl_age).flatten() #standardise age
    y = torch.tensor(rsl).flatten()
    y_sigma = torch.tensor(rsl_2sd/2).flatten()
    x_sigma = torch.tensor(rsl_age_2sd/2).flatten()
    
    marine_limiting = [X[marine_index],y[marine_index],y_sigma[marine_index],
                       x_sigma[marine_index],rsl_lon[marine_index].values,rsl_lat[marine_index].values, rsl_region[marine_index].values]

    SLIP = [X[SLIP_index],y[SLIP_index],y_sigma[SLIP_index],
                       x_sigma[SLIP_index],rsl_lon[SLIP_index].values,rsl_lat[SLIP_index].values, rsl_region[SLIP_index].values]

    marine_limiting = [X[terrestrial_index],y[terrestrial_index],y_sigma[terrestrial_index],
                      x_sigma[terrestrial_index],rsl_lon[terrestrial_index].values,rsl_lat[terrestrial_index].values, rsl_region[terrestrial_index].values]

    return marine_limiting, SLIP, marine_limiting

def load_PSMSL_data(data_folder,min_lat=25,max_lat=50,min_lon=-90,max_lon=-60,min_time_span=100,latest_age=2000):
    '''
    A function to load annual sea-level data from PSMSL (https://psmsl.org/), note the site file should be copy and pasted into the data folder.
    We have filtered out data with -99999 value, missing data (with 'Y' indicated) or any flagged data (i.e., the fourth column is not 0).
    ---------Inputs---------
    data_folder: str, the path to the data folder
    min_lat: float, the minimum latitude of the region of interest
    max_lat: float, the maximum latitude of the region of interest
    min_lon: float, the minimum longitude of the region of interest
    max_lon: float, the maximum longitude of the region of interest

    ---------Outputs---------
    US_AT_data: a list containing US Atlantic coast data 
    '''
    if data_folder[-1]!='/':
        data_folder = data_folder+'/'
    if len(os.listdir(data_folder))<5:
        with zipfile.ZipFile(data_folder+'TG_data.zip', 'r') as zip_ref:
            zip_ref.extractall(data_folder)
    site_file = pd.read_table(data_folder+'/filelist.txt',delimiter=';',header=None,)
    US_AT_index = (site_file.iloc[:,1]>=min_lat) & (site_file.iloc[:,1]<=max_lat) & (site_file.iloc[:,2]>=min_lon) & (site_file.iloc[:,2]<=max_lon)
    US_AT_site = site_file.iloc[:,0][US_AT_index].values
    US_AT_lat = site_file.iloc[:,1][US_AT_index].values
    US_AT_lon = site_file.iloc[:,2][US_AT_index].values

    #generate US Atlantic coast data
    US_AT_data = None
    for i,p in enumerate(US_AT_site):
        if US_AT_data is None:
            US_AT_data = pd.read_table(data_folder+str(p)+'.rlrdata',delimiter=';',header=None)
            US_AT_data[4] = US_AT_lat[i]
            US_AT_data[5] = US_AT_lon[i]
        else:
            tmp = pd.read_table(data_folder+str(p)+'.rlrdata',delimiter=';',header=None)
            tmp[4] = US_AT_lat[i]
            tmp[5] = US_AT_lon[i]
            US_AT_data = pd.concat([US_AT_data,tmp],ignore_index=True)
    
    data_filter = US_AT_data.iloc[:,1]!=-99999
    data_filter_2 = US_AT_data.iloc[:,2]=='N'
    data_filter_3 = US_AT_data.iloc[:,3]==0
    US_site_coord = np.unique(US_AT_data.iloc[:,4:],axis=0)
    data_filter_4 = np.zeros(len(data_filter_3),dtype=bool)
    data_filter_5 = np.zeros(len(data_filter_3),dtype=bool)
    new_rsl = np.zeros(len(data_filter_3))
    for i in range(len(US_site_coord)):
        site_index = np.sum(US_AT_data.iloc[:,4:],axis=1) == np.sum(US_site_coord[i])
        if np.max(US_AT_data[site_index].iloc[:,0])-np.min(US_AT_data[site_index].iloc[:,0])>=min_time_span:
            data_filter_4[site_index] = True
        if np.max(US_AT_data[site_index].iloc[:,0])>=latest_age:
            data_filter_5[site_index] = True
        new_rsl[site_index] = US_AT_data[site_index].iloc[:,1]- US_AT_data[site_index].iloc[-1,1]
    US_AT_data.iloc[:,1] = new_rsl
    data_filter_all = data_filter & data_filter_2 & data_filter_3 & data_filter_4 & data_filter_5
    US_AT_data = US_AT_data[data_filter_all]
    
    return US_AT_data

def rsl_data_pred(total_rsl_site,obs_age,site_index,pred_age=np.arange(12000,-100,-1000)):
    '''
    A function to predict rsl for each data point given the GIA prediction for each sites

    Inputs:

    ----------------------------
    total_rsl_site: 3D array, The GIA prediction for each sites, the shape should be [number of ice models, number of earth models, number of time steps]
    obs_age: 1D array, The age of the observation data
    site_index: 1D array, The index of the GIA prediction corresponding to each data point
    pred_age: 1D array, The age of the GIA prediction

    Outputs:

    ----------------------------
    tota_rsl_data: 3D array, The GIA prediction for each data point, the shape should be [number of ice models, number of earth models, number of data points]
    '''
    
    tota_rsl_data = np.zeros([*total_rsl_site.shape[:2],len(obs_age)])
    for i2 in range(total_rsl_site.shape[0]):
        for i3 in range(total_rsl_site.shape[1]):
            for i in range(len(obs_age)):
                pred_x,pred_y = pred_age, total_rsl_site[i2,i3,site_index[i],:]
                pred_interp = interpolate.interp1d(pred_x,pred_y)
                tota_rsl_data[i2,i3,i] = pred_interp(obs_age[i])
    return tota_rsl_data

def cal_rsl_misfit(total_rsl_site,obs_x,obs_y,obs_x_uncert,obs_y_uncert,site_index,pred_age=np.arange(12000,-100,-1000)):
    '''
    A function to calculate rsl data-model misfit based on misfit function from Love et al., 2016
    
    Inputs:

    ----------------------------------
    total_rsl_site: 3D array, The GIA prediction for each sites, the shape should be [number of ice models, number of earth models, number of time steps]
    obs_x: 1D array, The age of the observation data
    obs_y: 1D array, The RSL of the observation data
    obs_x_uncert: 1D array, The 1 sigma age uncertainty of the observation data
    obs_y_uncert: 1D array, The 1 sigma RSL uncertainty of the observation data
    site_index: 1D array, The index of the GIA prediction corresponding to each data point
    pred_age: 1D array, The age of the GIA prediction

    Outputs:

    ----------------------------------
    toto_misfit: 2D array, The misfit for each ice and earth model, the shape should be [number of ice models, number of earth models]
    toto_misfit_all: 3D array, The misfit for each ice and earth model for each data point, the shape should be [number of ice models, number of earth models, number of data points]
    '''
    toto_misfit = np.zeros([*total_rsl_site.shape[:2]])
    toto_misfit_all =  np.zeros([*total_rsl_site.shape[:2],len(obs_x)])

    for i2 in range(total_rsl_site.shape[0]):
        for i3 in range(total_rsl_site.shape[1]):
            output_mis = np.zeros(len(obs_x))
            for i in range(len(rsl_coord)):
                obs_age,obs_rsl = obs_x[i],obs_y[i]
                obs_xerr,obs_yerr = obs_x_uncert[i],obs_y_uncert[i]
                pred_x,pred_y = pred_age, total_rsl_site[i2,i3,site_index[i],:]
                pred_interp = interpolate.interp1d(pred_x,pred_y)
                min_interp_age = obs_age-3*obs_xerr
                if min_interp_age<0: min_interp_age = 0 #GIA prediction only go to 0 BP
                interp_age = torch.arange(min_interp_age,obs_age+3*obs_xerr,1)
                
                pred_y = torch.tensor(pred_interp(interp_age))

                mis_age = (interp_age-obs_age)**2
                mis_rsl = (pred_y-obs_rsl)**2
                min_index = np.argmin(mis_age+mis_rsl)
                output_mis[i] = np.sqrt(mis_rsl[min_index]+mis_age[min_index])
            toto_misfit[i2,i3] = np.mean(output_mis)
            toto_misfit_all[i2,i3] = output_mis
    return toto_misfit,toto_misfit_all
def read_rsl_out(input_file,num_time_step):
    rsl_all,site_coord = [],[]
    rsl_out = open(input_file,'r')
    rsl_out = rsl_out.readlines()
    rsl_time = np.array(rsl_out[0].split(),dtype = float)
    rsl_out = rsl_out[1:]
    for i in range(len(rsl_out)):
        if i%(num_time_step+1) ==0:
            site_coord.append(np.array([float(i) for i in rsl_out[i].split()[1:3]]))
        else:
            rsl_all+=[float(i) for i in rsl_out[i].split()]
    return np.array(rsl_all).reshape([int(len(rsl_all)/num_time_step),num_time_step]),np.array(site_coord),rsl_time
