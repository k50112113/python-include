import numpy as np
import math
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

def intgr(x,y,n1,n2,lbox):
    inty = [0.]
    dr = x[1]-x[0]
    sum = 0
    for index in range(1,len(x)):
        sum += y[index]*x[index]**2*dr
        inty.append(sum)
    inty = np.array(inty)*4*np.pi*(n1*n2/lbox**3)
    return inty

def compute_cn(x,y,n1,n2,lbox,first_shell):
    return np.interp(first_shell, x, intgr(x,y,n1,n2,lbox))

def find_local_extrema_creep_cubicspline(start,step,end,direction,f):
    now = start
    maxitr = 1000
    while f(now,1)*f(now+step*direction,1) > 0 and maxitr > 0:
        maxitr -= 1
        if maxitr == 0 or now+step*direction > end:
            return -1
        now += step*direction
    return find_local_extrema_bisection(now,now+step*direction,f)

def find_local_extrema_bisection_cubicspline(left,right,f,maxitr=1000):
    mid = (left+right)/2
    while abs(f(mid,1)) > 1E-6 and maxitr > 0:
        maxitr -= 1
        if maxitr == 0:
            return -1
        if f(mid,1)*f(left,1) < 0:
            right = mid
        else:
            left = mid
        mid = (left+right)/2
    return mid    

def find_coordination_shell(r_,gr_,lim,sg_window=5,show=False):
    if sg_window > 0 and sg_window % 2 == 1:
        sg_ = savgol_filter(gr_, sg_window, 3)
        gr_cs_ = CubicSpline(r_, sg_)
    else:
        gr_cs_ = CubicSpline(r_, gr_)
    cutoff = find_local_extrema_bisection_cubicspline(lim[0],lim[1],gr_cs_)
    
    return cutoff, gr_cs_

def get_significant_value(value,significant_digits=1,round_method="round"):
    k = 0
    while abs(value*(10**k)) < 10**significant_digits and value != 0:
        k += 1
    if round_method == "round":
        return round(value*(10**k))/(10**k)
    elif round_method == "floor":
        return math.floor(value*(10**k))/(10**k)
    elif round_method == "ceil":
        return math.ceil(value*(10**k))/(10**k)

def get_best_colorbar_range(vmin,vmax,significant_digits=1):
    return get_significant_value(vmin,significant_digits=significant_digits,round_method="ceil"),\
           get_significant_value(vmax,significant_digits=significant_digits,round_method="floor")

def write_slurm(filename,ntasks_per_node=1,cpu_per_task=1,time=1,partition="secondary",\
                job_name="",output="",error="",mail_user="scl6@illinois.edu",mail_type="FAIL",\
                command=""):
    if job_name == "": job_name=filename
    if output == "": output=filename+'.out'
    if error == "": error=filename+'.err'
    with open(filename,"w",newline='\n') as fout:
        fout.write('''#!/bin/bash
#
#SBATCH --ntasks-per-node=%s
#SBATCH --cpus-per-task=%s
#SBATCH --time=%s:00:00
#SBATCH --partition=%s
#SBATCH --mail-user=%s
#SBATCH --mail-type=%s
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --error=%s
#####################################
cd $SLURM_SUBMIT_DIR
'''%(ntasks_per_node,cpu_per_task,time,partition,mail_user,mail_type,job_name,output,error))
        fout.write(command)

def find_file(input_dir,filename,filetype='f'):
    # from subprocess import Popen, PIPE
    # p = Popen(['find',input_dir,'-name',filename], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    # output, err = p.communicate()
    # output_str = ""
    # for a in output:
    #     output_str+=chr(a)
    # file_list = output_str.split('\n')
    # file_list.pop()
    # return file_list
    import os
    # print("find %s -name %s"%(input_dir,filename))
    file_list = os.popen("find %s -type %s -name %s"%(input_dir,filetype,filename)).read()
    file_list = file_list.split('\n')
    file_list.pop()
    return file_list

def mole_fraction(component1, component2, molefraction2):
    component = np.append(component1, component2*molefraction2/(1-molefraction2))
    return component

def make_xyz(coord, type, filename):
    with open(filename,"w") as fout:
        for i in range(coord.shape[0]):
            fout.write('''     %d
 i =        %d, time =        %.3f, E =        0
'''%(coord.shape[1],i,i))
            for j in range(coord.shape[1]):
                fout.write("%s "%(type[j]))
                fout.write("%f %f %f\n"%(coord[i][j][0],coord[i][j][1],coord[i][j][2]))