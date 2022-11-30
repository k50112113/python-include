
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class Plot2D:
    def __init__(self, CV, pe, force, eigenvalues, eigenvectors):
        nbin = round(CV.shape[0]**(1/CV.shape[1]))
        cv1 = CV[:,0].reshape(nbin,nbin)
        cv2 = CV[:,1].reshape(nbin,nbin)
        eig1 = eigenvalues[:,0].reshape(nbin,nbin)
        eig2 = eigenvalues[:,1].reshape(nbin,nbin)
        p = pe.reshape(nbin,nbin)
        eiv1 = eigenvectors[:,0,:]
        eiv2 = eigenvectors[:,1,:]
        
        fig, ax = self.plot_basic(cv1, cv2, p, 'jet')
        # self.add_arrows(ax, CV, force, eiv1, eiv2)
        plt.savefig("pe.pdf")
        plt.savefig("pe.png")
        plt.close('all')
        # fig, ax = self.plot_basic(cv1, cv2, eig1, 'bwr', colors.TwoSlopeNorm(vcenter=0))
        # plt.savefig("eig1.pdf")
        # plt.savefig("eig1.png")
        # plt.close('all')
        # fig, ax = self.plot_basic(cv1, cv2, eig2, 'bwr', colors.TwoSlopeNorm(vcenter=0))
        # plt.savefig("eig2.pdf")
        # plt.savefig("eig2.png")
        # plt.close('all')

    def plot_basic(self, cv1, cv2, quantity, cmap, norm=None):
        fig = plt.figure(figsize=(12,12))
        axsize = .7
        ax = fig.add_axes([.7*(1-axsize), .5*(1-axsize), axsize, axsize])
        ax.tick_params(axis="x",which ="major",length=9,width=2,labelsize=24, pad=10)
        ax.tick_params(axis="y",which ="major",length=9,width=2,labelsize=24, pad=10)
        ax.tick_params(axis="x",which ="minor",length=6,width=2,labelsize=24, pad=10)
        ax.tick_params(axis="y",which ="minor",length=6,width=2,labelsize=24, pad=10)
        ax.set_xlabel("$X$",fontsize=26)
        ax.set_ylabel("$Y$",fontsize=26)
        ax.spines["top"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["right"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.tick_params(axis="both",direction="in",which ="both",top=True,right=True)
        surf = ax.contourf(cv1, cv2, quantity, cmap = cmap, levels=500, extend='both', norm=norm)
        for c in surf.collections: c.set_edgecolor("face")
        cb = fig.colorbar(surf, ax=ax, pad=0.08, extend='both')
        cb.ax.tick_params(labelsize=20)
        # if colorbar_title != "": cb.set_label(colorbar_title, fontsize=20)
        return fig, ax
    
    def add_arrows(self, ax, CV, force, eiv1, eiv2):
        scale = 0.1
        for i in range(len(CV)):
            ax.arrow(CV[i][0],CV[i][1],eiv1[i][0]*scale*0.1,eiv1[i][1]*scale*0.1,color='k', width=0.00002*scale, head_width=0.005*scale, head_length=0.005*scale)
            ax.arrow(CV[i][0],CV[i][1],eiv2[i][0]*scale*0.1,eiv2[i][1]*scale*0.1,color='white', width=0.00002*scale, head_width=0.005*scale, head_length=0.005*scale)
            ax.arrow(CV[i][0],CV[i][1],force[i][0]*scale*0.05,force[i][1]*scale*0.05,color='purple', width=0.00002*scale, head_width=0.005*scale, head_length=0.005*scale)