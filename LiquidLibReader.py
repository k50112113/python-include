import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.optimize import leastsq

def smooth(x,y,n_points=1000,sg_filter=True,sg_window=101):
    xx = np.linspace(x.min(),x.max(),n_points)
    cs = CubicSpline(x, y)
    if sg_filter:
        sg = savgol_filter(cs(xx), sg_window, 3)
        return xx,sg
    else:
        return xx,cs(xx)
        
def sg_smooth(x,y,sg_window=11):
    return x,savgol_filter(y, sg_window, 3)
        
def read(filename,quantity="",target_k=0,target_k_index=-1):
    
    with open(filename,"r") as fin:
        aline = fin.readline()
        
        if quantity == "":
            if "Self intermediate scattering function" in aline:
                quantity="fskt"
            elif "Collective intermediate scattering function" in aline:
                quantity="fkt"
            
        aline = fin.readline()
        
        if "S(k)" in aline and quantity == "":
            quantity = "sk"
        elif "g(r)" in aline and quantity == "":
            quantity = "gr"
        elif "r2(t)" in aline and quantity == "":
            quantity = "r2t"
        elif "alpha_2" in aline and quantity == "":
            quantity = "alpha_2"
        elif "chi_4" in aline and quantity == "":
            quantity = "chi_4"
        
            
        
        if quantity == "gr" or quantity == "sk" or quantity == "r2t" or quantity == "alpha_2" or quantity == "chi_4":
            x_ = []
            y_ = []
            for aline in fin:
                if "#" not in aline:
                    linelist = aline.strip().split()
                    x_.append(float(linelist[0]))
                    y_.append(float(linelist[1]))
            
            return np.array(x_), np.array(y_), quantity
        elif quantity == "fskt" or quantity == "fkt":
            k_ = []
            t_ = []
            f_ = []
            aline = fin.readline()
            while "#" not in aline:
                k_.append(float(aline.strip()))
                aline = fin.readline()
            for aline in fin:
                linelist = aline.strip().split()
                t_.append(float(linelist[0]))
                f_.append(np.array([float(i) for i in linelist[1:]]))
            
            k_ = np.array(k_)
            t_ = np.array(t_)
            f_tk = np.array(f_)
            f_kt = f_tk.transpose()
            if target_k > 0 and target_k_index > -1:
                print("Error: Both target_k and target_k_index specified.")
                exit()
            elif target_k > 0:
                f_kt_fitk = []
                gaussian = lambda p, x: p[0]*np.exp(-p[1]*(x**2))
                error_gaussian  = lambda p, x, y : y - gaussian(p, x)
                fit_coeff = []
                for a_f_tk in f_tk:
                    init  = [1, 0.001]
                    coeff = leastsq(error_gaussian, init, args=(k_, a_f_tk))[0]
                    f_kt_fitk.append(gaussian(coeff[:],target_k))
                return t_, np.array(f_kt_fitk), quantity                
            elif target_k_index > -1:
                return k_[target_k_index], t_, f_kt[target_k_index], quantity
            else:
                return k_, t_, f_kt, quantity
            
            
            