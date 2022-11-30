import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import sys


DefaultTickFontSize = 26 
DefaultLegendFontSize = 26 
DefaultLabelFontSize = 26 
DefaultPlottitleFontSize = 26
DefaultLegendtitleFontSize = 26
DefaultFontSize = 26
def setupFigure(figsize=(10,8), axsize=.7):
    '''
    rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman'] + rcParams['font.serif']
    rcParams['font.size'] = 10.0
    rcParams['legend.fontsize'] = 'medium'
    '''
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.65*(1-axsize), 0.65*(1-axsize), axsize, axsize])
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

def get_Color_from_RGB(RGB):
    return "#%02x%02x%02x"%RGB

def gradient_image(ax, extent=(0, 1, 0, 1), direction=0, cmap_range=(0, 1), **kwargs):
    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    X = np.array([[v @ [1, 0], v @ [1, 1]],
                  [v @ [0 ,0], v @ [0, 1]]])
    a, b = cmap_range
    X = a + (b - a) / X.max() * X
    im = ax.imshow(X, extent=extent, interpolation='bicubic', vmin=0, vmax=1, **kwargs)
    return im

def get_color_series(color_,cmap):
        color_tmp = []
        cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(color_))
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap))
        for i in color_:
            color_tmp.append(scalarMap.to_rgba(i))  
        return color_tmp


def plot(x_,y_,plottype=None,legendtitle="",plottitle="",
yerr_=[],legend_=[],linestyle_=None,markerstyle_=None,markerface_=None,markersize_=None,lw_=None,color_=None,log_fg=(False,False),
xlabel=None,ylabel=None,xminortick=2,yminortick=4,
legend_fontsize=DefaultLegendFontSize,legendtitle_fontsize=DefaultLegendtitleFontSize,label_fontsize=DefaultLabelFontSize,plottitle_fontsize=DefaultPlottitleFontSize,tick_fontsize=DefaultTickFontSize,tickx_fontsize=None,ticky_fontsize=None,figsize=(10,8),axsize=.7,margin_ratio=0.5,legend_column=1,legend_location=0,
xlim=None,ylim=None,xstart=None,ystart=None,xinc=None,yinc=None,bg_grad=False,cmap='',maxcolor=None):
    default_color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    xlabel_dic = {\
"PDF": r'$r\ \mathrm{(Å)}$',
"RDF": r'$r\ \mathrm{(Å)}$',
"gr": r'$r\ \mathrm{(Å)}$',

"ADF": r'$\theta\ \mathrm{(degree)}$',

"sk": r'$Q\ \mathrm{(Å^{-1})}$',
"sq": r'$Q\ \mathrm{(Å^{-1})}$',

"msd": r'$t\ \mathrm{(ps)}$',
"r2t": r'$t\ \mathrm{(ps)}$',

"vacf": r'$t\ \mathrm{(ps)}$',

"eacf": r'$t\ \mathrm{(ps)}$', #electrical current

"sacf": r'$t\ \mathrm{(ps)}$',

"fskt": r'$t\ \mathrm{(ps)}$',

"fkt": r'$t\ \mathrm{(ps)}$',

"alpha_2": r'$t\ \mathrm{(ps)}$',

"chi_4": r'$t\ \mathrm{(ps)}$',
}
    ylabel_dic = {\
"PDF": r'$g(r)$',
"RDF": r'$g(r)$',
"gr": r'$g(r)$',

"ADF": r'$ADF(\theta)$',

"sk": r'$S(Q)$',
"sq": r'$S(Q)$',

"msd":r'$〈r^2〉(t)\ \mathrm{(nm^2)}$',
"r2t":r'$〈r^2〉(t)\ \mathrm{(nm^2)}$',
"mmsd":r'$〈r^2〉(t)\ \mathrm{(nm^2)}$',
"mr2t":r'$〈r^2〉(t)\ \mathrm{(nm^2)}$',

"vacf": r'$\frac{〈v(t)v(0)〉/〈v(0)^2〉}$',

"eacf": r'$〈J(t)J(0)〉/〈J(0)^2〉$', #electrical current

"hacf": r'$\frac{〈J(t)J(0)〉/〈J(0)^2〉}$', #heat flux

"sacf": r'$\frac{〈\tau_{ij}(t)\tau_{ij}(0)〉}{〈\tau_{ij}(0)^2〉}$', #shear stress

"fskt": r'$F_s(Q,t)$',

"fkt": r'$F(Q,t)$',

"alpha_2": r'$\alpha_2(t)$',

"chi_4": r'$\chi_4(t)$',
}
    bgcolors = [(1, 1, 1), (float(226)/float(255), float(235)/float(255), float(244)/float(255))]
    #bgcolors = [(1, 1, 1), (float(223)/float(255), float(223)/float(255), float(223)/float(255))]
    bgcm = mpl.colors.LinearSegmentedColormap.from_list("standard",bgcolors,N=256)
    if cmap == '':
        if color_:
            color_tmp = []
            for i in color_:
                if type(i) == tuple:
                    if len(i) == 3:
                        color_tmp.append(get_Color_from_RGB(i))
                    elif len(i) == 4:
                        color_tmp.append(i)
                elif type(i) == int:
                    color_tmp.append(default_color_list[i])
                else:
                    color_tmp.append(i)      
            color_ = color_tmp[:] 
    else:
        color_tmp = []
        if not maxcolor:
            maxcolor = 0
            if color_:
                for i in color_:
                    if type(i) == int: maxcolor = max(maxcolor, i)
        cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(x_) if not color_ else maxcolor)
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap))
        if color_:
            for i in color_:
                if type(i) == int: color_tmp.append(scalarMap.to_rgba(i))  
                else: color_tmp.append(i)
        else:
            for i in range(len(x_)):
                color_tmp.append(scalarMap.to_rgba(i))  
        color_ = color_tmp[:]


    LegendFontSize = int(legend_fontsize)
    PlottitleFontSize = int(plottitle_fontsize)
    LabelFontSize = int(label_fontsize)
    TickFontSize = int(tick_fontsize)
    if tickx_fontsize == None: TickXFontSize = TickFontSize
    else: TickXFontSize = int(tickx_fontsize)
    if ticky_fontsize == None: TickYFontSize = TickFontSize
    else: TickYFontSize = int(ticky_fontsize)

    FrameWidth = 1.5
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([margin_ratio*(1-axsize), margin_ratio*(1-axsize), axsize, axsize])
    ax.set_title(label=plottitle, fontsize=PlottitleFontSize, pad=10)
    ax.spines["top"].set_linewidth(FrameWidth)
    ax.spines["left"].set_linewidth(FrameWidth)
    ax.spines["right"].set_linewidth(FrameWidth)
    ax.spines["bottom"].set_linewidth(FrameWidth)
    
    if type(lw_) == float or type(lw_) == int:
        lw_ = [lw_]*len(x_)
    
    if type(markerstyle_) == str:
        markerstyle_ = [markerstyle_]*len(x_)
    
    if type(markersize_) == float or type(markersize_) == int:
        markersize_ = [markersize_]*len(x_)
        
    if type(markerface_) == str:
        markerface_ = [markerface_]*len(x_)

    for index in range(len(x_)):
        
        if len(legend_) > 0 and index < len(legend_):
            legend = "%s"%(legend_[index])
        else:
            legend = ""
        
        if len(yerr_)==0 or len(yerr_[index])==0:
            thisline = ax.plot(x_[index],y_[index],label=legend)
        else:
            thisline = ax.errorbar(x_[index],y_[index],yerr_[index],lw=2,fmt='-o',elinewidth=2, capsize=3, markersize=9,label=legend)   
            if color_!=None and index < len(color_):
                thisline[1][1].set_color(color_[index])
                thisline[1][0].set_color(color_[index])
                thisline[2][0].set_color(color_[index])

        if lw_!=None and index < len(lw_):
            thisline[0].set_linewidth(lw_[index])
        if linestyle_!=None and index < len(linestyle_):
            thisline[0].set_linestyle(linestyle_[index])
        if markerstyle_!=None and index < len(markerstyle_):
            thisline[0].set_marker(markerstyle_[index])
        if markersize_!=None and index < len(markersize_):
            thisline[0].set_markersize(markersize_[index])
        if markerface_!=None and index < len(markerface_):
            #thisline[0].set_markerfacecolor(markerface_[index])
            thisline[0].set_fillstyle(markerface_[index])
        if color_!=None and index < len(color_):
            thisline[0].set_color(color_[index])

            
    ax.legend(title=legendtitle,title_fontsize=legendtitle_fontsize,fontsize=LegendFontSize,ncol=legend_column, labelspacing=0.5, frameon=False, loc=legend_location)
    if log_fg[0] == True:    
        ax.set_xscale('log')
    if log_fg[1] == True:    
        ax.set_yscale('log')
    
    if xlim!=None:
        ax.set_xlim(xlim)
        if xinc!=None:
            xticks = []
            if log_fg[0] == True:
                i = 0
                newxlim=[xlim[0],xlim[1]]
                for j in [0,1]:
                    if isinstance(np.log10(xlim[j]), int)==False:
                        if xlim[j] >= 1:
                            newxlim[j]=10**(int(np.log10(xlim[j]))+1)
                        else:
                            newxlim[j]=10**(int(np.log10(xlim[j])))    
                while xstart*xinc**i <= newxlim[1]:
                    xticks.append(xstart*xinc**i)
                    i += 1
            else:
                i = 0
                while xstart+xinc*i <= xlim[1]:
                    xticks.append(xstart+xinc*i)
                    i += 1
            ax.set_xticks(xticks)
    
    if ylim!=None:
        ax.set_ylim(ylim)
        if yinc!=None:
            yticks = []
            if log_fg[1] == True:
                ax.set_yscale('log')
                i = 0
                newxlim=[ylim[0],ylim[1]]
                for j in [0,1]:
                    if isinstance(np.log10(ylim[j]), int)==False:
                        if ylim[j] >= 1:
                            newxlim[j]=10**(int(np.log10(ylim[j]))+1)
                        else:
                            newxlim[j]=10**(int(np.log10(ylim[j])))   
                while ystart*yinc**i <= newxlim[1]:
                    yticks.append(ystart*yinc**i)
                    i += 1
            else:
                i = 0
                while ystart+yinc*i <= ylim[1]:
                    yticks.append(ystart+yinc*i)
                    i += 1
            ax.set_yticks(yticks)
    
    if bg_grad == True:
        if not xlim: xx = ax.get_xlim()
        else: xx = xlim
        if not ylim: yy = ax.get_ylim()
        else: yy = ylim
        extent = (*xx,*yy)
        gradient_image(ax, transform=ax.transAxes, extent=extent, cmap=bgcm, aspect='auto')
            
    '''
    if log_fg[0] == False:    
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(xminortick))
    else:
        minxticks = []
        for atick in xticks:
            for j in range(1,10):
                minxticks.append(atick/10*float(j))
        ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(minxticks))
        ax.xaxis.set_minor_formatter(mpl.ticker.LogFormatter(base=xinc, labelOnlyBase=True))
    if log_fg[1] == False:         
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(yminortick))
    else:
        minxticks = []
        for atick in yticks:
            for j in range(1,10):
                minxticks.append(atick/10*float(j))
        ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator(minxticks))
        ax.yaxis.set_minor_formatter(mpl.ticker.LogFormatter(base=yinc, labelOnlyBase=True))
    '''
    widthx  = 2 if TickXFontSize > 0 else 0
    widthy  = 2 if TickYFontSize > 0 else 0
    lengthx = 9 if TickXFontSize > 0 else 0
    lengthy = 9 if TickYFontSize > 0 else 0
    ax.tick_params(axis="x",which ="major",length=lengthx,width=widthx,labelsize=TickXFontSize, pad=10)
    ax.tick_params(axis="y",which ="major",length=lengthy,width=widthy,labelsize=TickYFontSize, pad=10)
    lengthx = 6 if TickXFontSize > 0 else 0
    lengthy = 6 if TickYFontSize > 0 else 0
    ax.tick_params(axis="x",which ="minor",length=lengthx,width=widthx,labelsize=TickXFontSize, pad=10)
    ax.tick_params(axis="y",which ="minor",length=lengthy,width=widthy,labelsize=TickYFontSize, pad=10)
    ax.tick_params(axis="both",direction="in",which ="both",top=True,right=True)
    
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #rc('font',**{'family':'serif','serif':['Palatino']})
    #rc('text', usetex=True)
    if plottype!=None:
        if xlabel == None:
            xlabel=xlabel_dic[plottype]
        if ylabel == None:
            ylabel=ylabel_dic[plottype]
        
    ax.set_xlabel(xlabel, fontsize=LabelFontSize)
    ax.set_ylabel(ylabel, fontsize=LabelFontSize)
    return fig, ax, plt
    #plt.tight_layout()
    #plt.show()
