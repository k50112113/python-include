import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import sys

class ZLabPlot:
    def __init__(self, margin_ratio = 0.15, axsize = 0.7, wspace = 0.3, hspace = 0.3, rcmap = {}):
        #plt.rcParams["figure.figsize"] = figsize
        #plt.subplots_adjust(bottom = margin_ratio, top = margin_ratio + axsize, left = margin_ratio, right = margin_ratio + axsize, wspace = wspace, hspace = hspace)
        self.subplot_map = {}
        self.plot_data_map = {}
        self.default_framewidth = 1.5
        self.default_fontsize = 'large'
        plt.rcParams['axes.titlesize'] = 'large' # 'medium' in matplotlibrc
        plt.rcParams['xtick.major.size']  = plt.rcParams['ytick.major.size']  = 9  
        plt.rcParams['xtick.minor.size']  = plt.rcParams['ytick.minor.size']  = 6    
        plt.rcParams['xtick.major.width'] = plt.rcParams['ytick.major.width'] = 2  
        plt.rcParams['xtick.minor.width'] = plt.rcParams['ytick.minor.width'] = 2  
        plt.rcParams['xtick.major.pad']   = plt.rcParams['ytick.major.pad']   = 10  
        plt.rcParams['xtick.minor.pad']   = plt.rcParams['ytick.minor.pad']   = 10
        plt.rcParams['xtick.labelsize']   = plt.rcParams['ytick.labelsize']   = self.default_fontsize # 'medium' in matplotlibrc

        for a_key in rcmap:
            print("Manually set %s to %s"%(a_key, rcmap[a_key]))
            plt.rcParams[a_key] = rcmap[a_key]

        self.default_color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.number_of_default_colors = len(self.default_color_list)

        self.xlabel_map = {\
#Radial Distribution Function
"PDF": r'$r\ \mathrm{(Å)}$',
"RDF": r'$r\ \mathrm{(Å)}$',
"gr": r'$r\ \mathrm{(Å)}$',
#Angular Distribution Function
"ADF": r'$\theta\ \mathrm{(degree)}$',
#Structure Factor
"sk": r'$Q\ \mathrm{(Å^{-1})}$',
"sq": r'$Q\ \mathrm{(Å^{-1})}$',
#Mean Squared Displacment
"msd": r'$t\ \mathrm{(ps)}$',
"r2t": r'$t\ \mathrm{(ps)}$',
"mmsd":r'$〈r^2〉(t)\ \mathrm{(nm^2)}$',
"mr2t":r'$〈r^2〉(t)\ \mathrm{(nm^2)}$',
#Autocorrelation Functions
"vacf": r'$t\ \mathrm{(ps)}$',    #Velocity
"eacf": r'$t\ \mathrm{(ps)}$',    #Electrical Current
"hacf": r'$t\ \mathrm{(ps)}$',    #Heat Flux
"sacf": r'$t\ \mathrm{(ps)}$',    #Stress
"fskt": r'$t\ \mathrm{(ps)}$',    #Self-intermediate scattering function
"fkt": r'$t\ \mathrm{(ps)}$',     #Collective-intermediate scattering function
"alpha_2": r'$t\ \mathrm{(ps)}$', #Non-Gaussian parameter
"chi_4": r'$t\ \mathrm{(ps)}$'}   #Four-Point
        self.ylabel_map = {\
#Radial Distribution Function
"PDF": r'$g(r)$',
"RDF": r'$g(r)$',
"gr": r'$g(r)$',
#Angular Distribution Function
"ADF": r'$ADF(\theta)$',
#Structure Factor
"sk": r'$S(Q)$',
"sq": r'$S(Q)$',
#Mean Squared Displacment
"msd":r'$〈r^2〉(t)\ \mathrm{(nm^2/s)}$',
"r2t":r'$〈r^2〉(t)\ \mathrm{(nm^2/s)}$',
"mmsd":r'$〈r^2〉(t)\ \mathrm{(nm^2/s)}$',
"mr2t":r'$〈r^2〉(t)\ \mathrm{(nm^2/s)}$',
#Autocorrelation Functions
"vacf": r'$\frac{〈v(t)v(0)〉/〈v(0)^2〉}$',                          #Velocity
"eacf": r'$〈J(t)J(0)〉/〈J(0)^2〉$',                                 #Electrical Current
"hacf": r'$\frac{〈J(t)J(0)〉/〈J(0)^2〉}$',                          #Heat Flux
"sacf": r'$\frac{〈\tau_{ij}(t)\tau_{ij}(0)〉}{〈\tau_{ij}(0)^2〉}$', #Stress
"fskt": r'$F_s(Q,t)$',                                               #Self-intermediate scattering function
"fkt": r'$F(Q,t)$',                                                  #Collective-intermediate scattering function
"alpha_2": r'$\alpha_2(t)$',                                         #Non-Gaussian parameter
"chi_4": r'$\chi_4(t)$'}                                             #Four-Point

    def setupFigure(figsize=(10,8), axsize=.7):
        '''
        rcParams['text.usetex'] = True
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Computer Modern Roman'] + rcParams['font.serif']
        rcParams['font.size'] = 10.0
        rcParams['legend.fontsize'] = 'medium'
        '''
        self.fig = plt.figure(figsize=figsize)
        self.ax = fig.add_axes([0.65*(1-axsize), 0.65*(1-axsize), axsize, axsize])
        #mpl.rcParams['font.sans-serif'] = "Arial"
        ax.spines["top"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["right"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.tick_params(axis="x",which ="major",length=9,width=2,labelsize=DefaultTickFontSize, pad=10)
        ax.tick_params(axis="y",which ="major",length=9,width=2,labelsize=DefaultTickFontSize, pad=10)
        ax.tick_params(axis="x",which ="minor",length=6,width=2,labelsize=DefaultTickFontSize, pad=10)
        ax.tick_params(axis="y",which ="minor",length=6,width=2,labelsize=DefaultTickFontSize, pad=10)
        ax.tick_params(axis="both",direction="in",which ="both",top=True,right=True)
        return fig, ax

    def get_Color_from_RGB(self, RGB):
        #RGB = (int, int, int)
        return "#%02x%02x%02x"%RGB

    def get_color_series(self, color_index, cmap):
        #color_index = list of int
        #cmap = string
        color_rgb = []
        cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(color_index))
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap)) #mpl.colormaps[cmap]
        color_rgb = [scalarMap.to_rgba(c) for c in color_index]
        return color_rgb

    # def gradient_image(ax, extent=(0, 1, 0, 1), direction=0, cmap_range=(0, 1), **kwargs):
    #     phi = direction * np.pi / 2
    #     v = np.array([np.cos(phi), np.sin(phi)])
    #     X = np.array([[v @ [1, 0], v @ [1, 1]],
    #                 [v @ [0 ,0], v @ [0, 1]]])
    #     a, b = cmap_range
    #     X = a + (b - a) / X.max() * X
    #     im = ax.imshow(X, extent=extent, interpolation='bicubic', vmin=0, vmax=1, **kwargs)
    #     return im

    def add_subplot(self, subplot_name = "0", subplot_spec = 111, \
                          plottitle = None,\
                          framewidth = None, twinx = False):
        subplot_name_ = str(subplot_name)
        if self.subplot_map.get(subplot_name_) is not None:
            print("subplot name has already been used.")
        else:
            # self.subplot_map[subplot_name_] = plt.subplot(subplot_spec, position= [margin_ratio*(1-axsize), margin_ratio*(1-axsize), axsize, axsize])
            self.subplot_map[subplot_name_] = plt.subplot(subplot_spec)
            self.subplot_map[subplot_name_].set_title(label = plottitle, pad = 10)
            # if framewidth is None: framewidth = self.default_framewidth
            # self.subplot_map[subplot_name_].spines["top"].set_linewidth(framewidth)
            # self.subplot_map[subplot_name_].spines["left"].set_linewidth(framewidth)
            # self.subplot_map[subplot_name_].spines["right"].set_linewidth(framewidth)
            # self.subplot_map[subplot_name_].spines["bottom"].set_linewidth(framewidth)
            self.plot_data_map[subplot_name_] = []
            # self.subplot_map[subplot_name_].set_facecolor('white')

            if twinx == True:
                self.subplot_map[subplot_name_+"-t"] = self.subplot_map[subplot_name_].twinx()
                self.plot_data_map[subplot_name_+"-t"] = []

    def add_data(self, data_x, data_y, subplot_name = "0", legend = None, \
                       xlim = None, ylim = None, xlog = False, ylog = False, xstart = None, xinc = None, ystart = None, yinc = None, \
                       xlabel = None, ylabel = None, \
                       color = None, lw = None, ls = None, ms = None, msize = None, mf = None, cmap = None, max_cmap_index = None, \
                       legendtitle = None, legendtitle_fontsize = None, legend_fontsize = None, tick_fontsize = None, label_fontsize = None, hide_xtick = False, hide_ytick = False, \
                       ncol = 1, legend_location = None, \
                       twinx = False, plottype = None):
        #lw    = line width
        #ls    = line style https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
        #ms    = marker style https://matplotlib.org/stable/api/markers_api.html
        #msize = marker size
        #mf    = marker fill style {'full', 'left', 'right', 'bottom', 'top', 'none'}
        #color = https://matplotlib.org/stable/gallery/color/named_colors.html
        #cmap  = https://matplotlib.org/stable/tutorials/colors/colormaps.html
        
        ##########################Initialize
        subplot_name_ = str(subplot_name)
        if self.subplot_map.get(subplot_name_) is None:
            print("ZLabPlot error: subplot not found.")
            exit()
        if twinx == True:
            subplot_name_ += "-t"
            if self.subplot_map.get(subplot_name_) is None:
                print("ZLabPlot error: subplot specified has no twinx.")
                exit()
        ax = self.subplot_map[subplot_name_]
        if legendtitle_fontsize is None: legendtitle_fontsize = self.default_fontsize
        if legend_fontsize      is None: legend_fontsize      = self.default_fontsize
        if tick_fontsize        is None: tick_fontsize        = self.default_fontsize
        if label_fontsize       is None: label_fontsize       = self.default_fontsize

        ##########################Padding features
        data_len = len(data_x)
        legend_ = self.feature_padding(data_len, legend, None)
        if type(lw) == float or type(lw) == int:       lw_    = self.feature_padding(data_len, [], lw)
        else:                                          lw_    = self.feature_padding(data_len, lw, None)
        if type(ls) == str or type(ls) == tuple:       ls_    = self.feature_padding(data_len, [], ls)
        else:                                          ls_    = self.feature_padding(data_len, ls, None)
        if type(ms) == str or type(ms) == int:         ms_    = self.feature_padding(data_len, [], ms)
        else:                                          ms_    = self.feature_padding(data_len, ms, None)
        if type(msize) == float or type(msize) == int: msize_ = self.feature_padding(data_len, [], msize)
        else:                                          msize_ = self.feature_padding(data_len, msize, None)
        if type(mf) == str:                            mf_    = self.feature_padding(data_len, [], mf)
        else:                                          mf_    = self.feature_padding(data_len, mf, None)           

        ##########################Setting colors
        if cmap is None:
            if type(color) == tuple or type(color) == str or type(color) == int: color_ = self.feature_padding(data_len, [], color)
            else:                                                                color_ = self.feature_padding(data_len, color, None)
            color_tmp = []
            for c in color_:
                if type(c) == tuple:
                    if len(c) == 3:   color_tmp.append(self.get_Color_from_RGB(c))
                    elif len(c) == 4: color_tmp.append(c)
                elif type(c) == int: color_tmp.append(self.default_color_list[c%self.number_of_default_colors])
                else: color_tmp.append(c)      
            color_ = color_tmp
        else:
            color_tmp = []
            if max_cmap_index is None:
                max_cmap_index = 0
                if color is not None:
                    max_cmap_index = 0
                    for c in color:
                        if type(c) == int: max_cmap_index = max(max_cmap_index, c)
                else:
                    max_cmap_index = data_len
            cNorm  = mpl.colors.Normalize(vmin = 0, vmax = max_cmap_index)
            scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap))
            if color is None: color_tmp = [scalarMap.to_rgba(i) for i in range(data_len)]
            else:
                if len(color) != data_len:
                    print("ZLabPlot error: the length of color array specified is not consistent with number of curves provided.")
                    exit()
                for c in color:
                    if type(c) == int: color_tmp.append(scalarMap.to_rgba(c))  
                    else: color_tmp.append(c)
            color_ = color_tmp[:]

        ##########################Adding data
        for index in range(data_len):
            thisline = ax.plot(data_x[index], data_y[index], label=legend_[index])
            self.plot_data_map[subplot_name_].append(thisline[0])
            if lw_[index] is not None: self.plot_data_map[subplot_name_][-1].set_linewidth(lw_[index])
            if ls_[index] is not None: self.plot_data_map[subplot_name_][-1].set_linestyle(ls_[index])
            if ms_[index] is not None: self.plot_data_map[subplot_name_][-1].set_marker(ms_[index])
            if msize_[index] is not None: self.plot_data_map[subplot_name_][-1].set_markersize(msize_[index])
            if mf_[index] is not None: self.plot_data_map[subplot_name_][-1].set_fillstyle(mf_[index])
            #self.plot_data_map[subplot_name_][-1].set_markerfacecolor(markerface_[index])
            if color_[index] is not None: self.plot_data_map[subplot_name_][-1].set_color(color_[index])

        ##########################Setting legend box
        if legend is not None: ax.legend(title = legendtitle, ncol = ncol, labelspacing = 0.5, frameon = False, loc = legend_location)
        
        ##########################Setting axis scale, limit, and ticks
        if xlog: ax.set_xscale('log')
        if ylog: ax.set_yscale('log')
        if xlim is not None and xlim[0] is not None and xlim[1] is not None and xlim[0] >= xlim[1]: print("ZLabPlot error: custom x-axis limit error.")
        if ylim is not None and ylim[0] is not None and ylim[1] is not None and ylim[0] >= ylim[1]: print("ZLabPlot error: custom y-axis limit error.")
        if xlim is not None: ax.set_xlim(xlim)
        else:                xlim = ax.get_xlim()
        if ylim is not None: ax.set_ylim(ylim)
        else:                ylim = ax.get_ylim()
        if xstart is not None and xinc is not None:
            xticks = self.custom_ticks(self, xstart, xinc, xlog, xlim)
            ax.set_xticks(xticks)
        if ystart is not None and yinc is not None:
            yticks = self.custom_ticks(self, ystart, yinc, ylog, ylim)
            ax.set_yticks(yticks)
        
        # if bg_grad == True: gradient_image(ax, transform=ax.transAxes, extent=(*xlim,*ylim), cmap=bgcm, aspect='auto')
        
        if twinx == True:
            ax.tick_params(axis = "both", which ="both", bottom = not hide_xtick, top = not hide_xtick, left = False,          right = not hide_ytick)
        elif self.subplot_map.get(subplot_name_+"-t") is not None:     
            ax.tick_params(axis = "both", which ="both", bottom = not hide_xtick, top = not hide_xtick, left = not hide_ytick, right = False)
        else:
            ax.tick_params(axis = "both", which ="both", bottom = not hide_xtick, top = not hide_xtick, left = not hide_ytick, right = not hide_ytick)

        #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        #rc('font',**{'family':'serif','serif':['Palatino']})
        # plt.rc('text', usetex=True)

        # plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']
        # plt.rcParams['legend.fontsize'] = 'medium'

        if plottype is not None:
            if xlabel is None: xlabel = self.xlabel_map[plottype]
            if ylabel is None: ylabel = self.ylabel_map[plottype]
        ax.set_xlabel(xlabel, fontsize = label_fontsize)
        ax.set_ylabel(ylabel, fontsize = label_fontsize)

    def custom_ticks(self, tick_start, tick_inc, log, lim):
        i = 0
        ticks = []
        if log == False:
            while tick_start+tick_inc*i <= lim[1]:
                ticks.append(ystart+yinc*i)
                i += 1
        else:
            while tick_start+tick_inc**i <= lim[1]:
                ticks.append(ystart+yinc**i)
                i += 1
        return ticks

    def feature_padding(self, data_len, feature, pad_obj):
        if feature is None:           feature_  = [pad_obj]*data_len
        elif len(feature) < data_len: feature_  = feature + [pad_obj]*(data_len-len(feature))
        else:                         feature_  = feature[:]
        return feature_

    def show(self):
        plt.show()

    def save(self, filename = 'zlabplot.png', dpi = 'figure', transparent = True):
        plt.rcParams['savefig.transparent'] = transparent
        plt.savefig(filename, dpi = dpi)

    def clear(self):
        self.subplot_map = {}
        self.plot_data_map = {}
        plt.close('all')

# def plot(x_,y_,plottype=None,legendtitle="",plottitle="",
# yerr_=[],legend_=[],linestyle_=None,markerstyle_=None,markerface_=None,markersize_=None,lw_=None,color_=None,log_fg=(False,False),
# xlabel=None,ylabel=None,xminortick=2,yminortick=4,
# legend_fontsize=ZLab_DefaultFontSize,legendtitle_fontsize=ZLab_DefaultFontSize,label_fontsize=ZLab_DefaultFontSize,plottitle_fontsize=ZLab_DefaultFontSize,tick_fontsize=ZLab_DefaultFontSize,tickx_fontsize=None,ticky_fontsize=None,figsize=(10,8),axsize=.7,margin_ratio=0.5,legend_column=1,legend_location=0,
# xlim=None,ylim=None,xstart=None,ystart=None,xinc=None,yinc=None,bg_grad=False,cmap='',maxcolor=None):
#     default_color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
#     bgcolors = [(1, 1, 1), (float(226)/float(255), float(235)/float(255), float(244)/float(255))]
#     #bgcolors = [(1, 1, 1), (float(223)/float(255), float(223)/float(255), float(223)/float(255))]
#     bgcm = mpl.colors.LinearSegmentedColormap.from_list("standard",bgcolors,N=256)
#     if cmap == '':
#         if color_:
#             color_tmp = []
#             for i in color_:
#                 if type(i) == tuple:
#                     if len(i) == 3:
#                         color_tmp.append(get_Color_from_RGB(i))
#                     elif len(i) == 4:
#                         color_tmp.append(i)
#                 elif type(i) == int:
#                     color_tmp.append(default_color_list[i])
#                 else:
#                     color_tmp.append(i)      
#             color_ = color_tmp[:] 
#     else:
#         color_tmp = []
#         if not maxcolor:
#             maxcolor = 0
#             if color_:
#                 for i in color_:
#                     if type(i) == int: maxcolor = max(maxcolor, i)
#         cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(x_) if not color_ else maxcolor)
#         scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap))
#         if color_:
#             for i in color_:
#                 if type(i) == int: color_tmp.append(scalarMap.to_rgba(i))  
#                 else: color_tmp.append(i)
#         else:
#             for i in range(len(x_)):
#                 color_tmp.append(scalarMap.to_rgba(i))  
#         color_ = color_tmp[:]


#     LegendFontSize = int(legend_fontsize)
#     PlottitleFontSize = int(plottitle_fontsize)
#     LabelFontSize = int(label_fontsize)
#     TickFontSize = int(tick_fontsize)
#     if tickx_fontsize == None: TickXFontSize = TickFontSize
#     else: TickXFontSize = int(tickx_fontsize)
#     if ticky_fontsize == None: TickYFontSize = TickFontSize
#     else: TickYFontSize = int(ticky_fontsize)

#     FrameWidth = 1.5
    
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_axes([margin_ratio*(1-axsize), margin_ratio*(1-axsize), axsize, axsize])
#     ax.set_title(label=plottitle, fontsize=PlottitleFontSize, pad=10)
#     ax.spines["top"].set_linewidth(FrameWidth)
#     ax.spines["left"].set_linewidth(FrameWidth)
#     ax.spines["right"].set_linewidth(FrameWidth)
#     ax.spines["bottom"].set_linewidth(FrameWidth)
    
#     if type(lw_) == float or type(lw_) == int:
#         lw_ = [lw_]*len(x_)
    
#     if type(markerstyle_) == str:
#         markerstyle_ = [markerstyle_]*len(x_)
    
#     if type(markersize_) == float or type(markersize_) == int:
#         markersize_ = [markersize_]*len(x_)
        
#     if type(markerface_) == str:
#         markerface_ = [markerface_]*len(x_)

#     for index in range(len(x_)):
        
#         if len(legend_) > 0 and index < len(legend_):
#             legend = "%s"%(legend_[index])
#         else:
#             legend = ""
        
#         if len(yerr_)==0 or len(yerr_[index])==0:
#             thisline = ax.plot(x_[index],y_[index],label=legend)
#         else:
#             thisline = ax.errorbar(x_[index],y_[index],yerr_[index],lw=2,fmt='-o',elinewidth=2, capsize=3, markersize=9,label=legend)   
#             if color_!=None and index < len(color_):
#                 thisline[1][1].set_color(color_[index])
#                 thisline[1][0].set_color(color_[index])
#                 thisline[2][0].set_color(color_[index])

#         if lw_!=None and index < len(lw_):
#             thisline[0].set_linewidth(lw_[index])
#         if linestyle_!=None and index < len(linestyle_):
#             thisline[0].set_linestyle(linestyle_[index])
#         if markerstyle_!=None and index < len(markerstyle_):
#             thisline[0].set_marker(markerstyle_[index])
#         if markersize_!=None and index < len(markersize_):
#             thisline[0].set_markersize(markersize_[index])
#         if markerface_!=None and index < len(markerface_):
#             #thisline[0].set_markerfacecolor(markerface_[index])
#             thisline[0].set_fillstyle(markerface_[index])
#         if color_!=None and index < len(color_):
#             thisline[0].set_color(color_[index])

            
#     ax.legend(title=legendtitle,title_fontsize=legendtitle_fontsize,fontsize=LegendFontSize,ncol=legend_column, labelspacing=0.5, frameon=False, loc=legend_location)
#     if log_fg[0] == True:    
#         ax.set_xscale('log')
#     if log_fg[1] == True:    
#         ax.set_yscale('log')
    
#     if xlim!=None:
#         ax.set_xlim(xlim)
#         if xinc!=None:
#             xticks = []
#             if log_fg[0] == True:
#                 i = 0
#                 newxlim=[xlim[0],xlim[1]]
#                 for j in [0,1]:
#                     if isinstance(np.log10(xlim[j]), int)==False:
#                         if xlim[j] >= 1:
#                             newxlim[j]=10**(int(np.log10(xlim[j]))+1)
#                         else:
#                             newxlim[j]=10**(int(np.log10(xlim[j])))    
#                 while xstart*xinc**i <= newxlim[1]:
#                     xticks.append(xstart*xinc**i)
#                     i += 1
#             else:
#                 i = 0
#                 while xstart+xinc*i <= xlim[1]:
#                     xticks.append(xstart+xinc*i)
#                     i += 1
#             ax.set_xticks(xticks)
    
#     if ylim!=None:
#         ax.set_ylim(ylim)
#         if yinc!=None:
#             yticks = []
#             if log_fg[1] == True:
#                 ax.set_yscale('log')
#                 i = 0
#                 newxlim=[ylim[0],ylim[1]]
#                 for j in [0,1]:
#                     if isinstance(np.log10(ylim[j]), int)==False:
#                         if ylim[j] >= 1:
#                             newxlim[j]=10**(int(np.log10(ylim[j]))+1)
#                         else:
#                             newxlim[j]=10**(int(np.log10(ylim[j])))   
#                 while ystart*yinc**i <= newxlim[1]:
#                     yticks.append(ystart*yinc**i)
#                     i += 1
#             else:
#                 i = 0
#                 while ystart+yinc*i <= ylim[1]:
#                     yticks.append(ystart+yinc*i)
#                     i += 1
#             ax.set_yticks(yticks)
    
#     if bg_grad == True:
#         if not xlim: xx = ax.get_xlim()
#         else: xx = xlim
#         if not ylim: yy = ax.get_ylim()
#         else: yy = ylim
#         extent = (*xx,*yy)
#         gradient_image(ax, transform=ax.transAxes, extent=extent, cmap=bgcm, aspect='auto')
            
#     '''
#     if log_fg[0] == False:    
#         ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(xminortick))
#     else:
#         minxticks = []
#         for atick in xticks:
#             for j in range(1,10):
#                 minxticks.append(atick/10*float(j))
#         ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(minxticks))
#         ax.xaxis.set_minor_formatter(mpl.ticker.LogFormatter(base=xinc, labelOnlyBase=True))
#     if log_fg[1] == False:         
#         ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(yminortick))
#     else:
#         minxticks = []
#         for atick in yticks:
#             for j in range(1,10):
#                 minxticks.append(atick/10*float(j))
#         ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator(minxticks))
#         ax.yaxis.set_minor_formatter(mpl.ticker.LogFormatter(base=yinc, labelOnlyBase=True))
#     '''
#     widthx  = 2 if TickXFontSize > 0 else 0
#     widthy  = 2 if TickYFontSize > 0 else 0
#     lengthx = 9 if TickXFontSize > 0 else 0
#     lengthy = 9 if TickYFontSize > 0 else 0
#     ax.tick_params(axis="x",which ="major",length=lengthx,width=widthx,labelsize=TickXFontSize, pad=10)
#     ax.tick_params(axis="y",which ="major",length=lengthy,width=widthy,labelsize=TickYFontSize, pad=10)
#     lengthx = 6 if TickXFontSize > 0 else 0
#     lengthy = 6 if TickYFontSize > 0 else 0
#     ax.tick_params(axis="x",which ="minor",length=lengthx,width=widthx,labelsize=TickXFontSize, pad=10)
#     ax.tick_params(axis="y",which ="minor",length=lengthy,width=widthy,labelsize=TickYFontSize, pad=10)
#     ax.tick_params(axis="both",direction="in",which ="both",top=True,right=True)
    
#     #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#     #rc('font',**{'family':'serif','serif':['Palatino']})
#     #rc('text', usetex=True)
#     if plottype!=None:
#         if xlabel == None:
#             xlabel=xlabel_dic[plottype]
#         if ylabel == None:
#             ylabel=ylabel_dic[plottype]
        
#     ax.set_xlabel(xlabel, fontsize=LabelFontSize)
#     ax.set_ylabel(ylabel, fontsize=LabelFontSize)
#     return fig, ax, plt
#     #plt.tight_layout()
#     #plt.show()
