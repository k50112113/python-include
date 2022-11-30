import numpy as np
import os
import random
import copy

def ACF_DIRECT(Data,time,dt_index_range,dt_index_max):
    Data = np.array(Data)
    ACF = []
    Intg_ACF = []
    
    dt_list = time[:dt_index_max]
    dim = len(Data[0])
    
    quantity = 0.0
    cntt = 0
    for start_index in range(0,len(Data),dt_index_range):
        print("t0 = %f"%(time[start_index]))
        now_lim_index = start_index+dt_index_range
        if now_lim_index < len(Data):
            Cor = np.zeros((len(dt_list)))
            cntt += 1
            for dt_index in range(len(dt_list)):
                tmp = np.zeros(dim)
                cnt = 0
                t_index = start_index
                while t_index+dt_index < now_lim_index:
                    tmp += Data[t_index]*Data[t_index+dt_index]
                    cnt += 1
                    t_index += 1
                Cor[dt_index] = np.mean(tmp/float(cnt))
            sum = 0
            ACF.append(Cor)
            Intg_ACF.append([])
            for dt_index in range(len(dt_list)):
                sum += Cor[dt_index]
                quantity = (sum-Cor[0]/2.0-Cor[dt_index]/2.0)
                Intg_ACF[len(Intg_ACF)-1].append(quantity)
        else:
            break
    print("Number of profiles = %d"%(cntt))

    return dt_list,ACF,Intg_ACF
    
    
def ACF_FFT(Data,time,dt_index_range,dt_index_max): #See Tildesley Page 280
    #Data: N x m
    ACF = []
    Intg_ACF = []
    dt_list = time[:dt_index_max]
    dim = Data.shape[1]
    quantity = 0.0
    cntt = 0
    for start_index in range(0,Data.shape[0],dt_index_range):
        #print("t0 = %f"%(time[start_index]))
        now_lim_index = start_index+dt_index_range
        if now_lim_index < Data.shape[0]:
            Cor = np.zeros(dt_list.shape[0])
            cntt += 1
            for i in range(dim):
                tmpData = Data[start_index:now_lim_index][:,i]
                l = tmpData.shape[0]
                tmpfft = np.fft.fft(np.concatenate((tmpData,np.zeros(l)),axis=0))
                c = np.fft.ifft(tmpfft*np.conjugate(tmpfft)).real
                corr = c[:len(c)//2]
                corr /= np.array([l-x for x in range(l)])
                Cor += corr[:dt_index_max]
            Cor /= dim 
            intg = 0
            ACF.append(Cor)
            Intg_ACF.append([])
            for dt_index in range(len(dt_list)):
                intg += Cor[dt_index]
                quantity = (intg-Cor[0]/2.0-Cor[dt_index]/2.0)
                Intg_ACF[len(Intg_ACF)-1].append(quantity)
                
    print("Number of profiles = %d"%(cntt))
    return np.array(dt_list),np.array(ACF),np.array(Intg_ACF)