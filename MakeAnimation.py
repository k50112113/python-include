#import numpy as np
#import matplotlib as mpl
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
import CollectiveVariable as CLVR
import numpy as np

def animate_rotate(i,dummy,ax,CV,colorcode,cmap,vmin,vmax,elev):
    azim = i*4
    ax.view_init(elev=elev, azim=azim)
    direction = np.array([np.cos(azim/180*np.pi),np.sin(azim/180*np.pi),0.0])
    zorder = np.inner(CV,direction)
    arg = np.argsort(zorder)
    ax.clear()
    ax.tick_params(axis="x",which ="major",length=9,width=2,labelsize=24, pad=10)
    ax.tick_params(axis="y",which ="major",length=9,width=2,labelsize=24, pad=10)
    ax.tick_params(axis="x",which ="minor",length=6,width=2,labelsize=24, pad=10)
    ax.tick_params(axis="y",which ="minor",length=6,width=2,labelsize=24, pad=10)
    ax.tick_params(axis="z",which ="major",length=9,width=2,labelsize=24, pad=10)
    ax.tick_params(axis="z",which ="minor",length=6,width=2,labelsize=24, pad=10)
    ax.set_xlabel("$CV_1$",fontsize=26,labelpad=26)
    ax.set_ylabel("$CV_2$",fontsize=26,labelpad=26)
    ax.set_zlabel("$CV_3$",fontsize=26,labelpad=26)
    ax.set_xlim3d([-1.1, 1.1])
    ax.set_ylim3d([-1.1, 1.1])
    ax.set_zlim3d([-1.1, 1.1])
    ax.set_xticks([-1.0,-0.5,0.0,0.5,1.0])
    ax.set_yticks([-1.0,-0.5,0.0,0.5,1.0])
    ax.set_zticks([-1.0,-0.5,0.0,0.5,1.0])
    ax.scatter(CV[arg][:,0],CV[arg][:,1],CV[arg][:,2],s=1,c=colorcode[arg],cmap=cmap,vmin=vmin,vmax=vmax)
    print(i,end="\r")

def make_animation_rotate(fig,ax,filename,CV,colorcode,cmap,vmin,vmax,elev=13,fps=15,nframe=90):
    dummy=1
    ani = animation.FuncAnimation(fig, animate_rotate, fargs=(dummy,ax,CV,colorcode,cmap,vmin,vmax,elev), frames=nframe, interval=20, blit=False, save_count=50)
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(filename+'.mp4', writer=writer)

def animate_rotate_cont(i,dummy,ax,elev):
    ax.view_init(elev=elev, azim=i*4)
    print(i,end="\r")

def make_animation_rotate_cont(fig,ax,filename,elev=13,fps=15,nframe=90):
    dummy=1
    ani = animation.FuncAnimation(fig, animate_rotate_cont, fargs=(dummy,ax,elev), frames=nframe, interval=20, blit=False, save_count=50)
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(filename+'.mp4', writer=writer) 

def animate_move(i,data,ax,title,dim,timescale):
    #ax.view_init(elev=10., azim=i)
    if dim == 2:
        ax.set_offsets(data[i])
    elif dim == 3:
        ax._offsets3d = (data[i][:,0], data[i][:,1], data[i][:,2])
    #im.set_array(colorcode[i*10])
    #im.changed()
    title.set_text('Time = %dfs'%(i*timescale))
    print(i,end='\r')
    return ax,

def make_animation_move(data,title,nframe,filename,timescale=1,fps=30):
    dim = data.shape[2]
    fig, ax = CLVR.setupFigure(dim=dim)
    title_ = ax.set_title(title, fontsize=40)
    color = 'k'
    if dim==3:
        ax.set_xlim3d([-1.1, 1.1])
        ax.set_ylim3d([-1.1, 1.1])
        ax.set_zlim3d([-1.1, 1.1])
        im = ax.scatter(data[0][:,0],data[0][:,1],data[0][:,2],s=5,c=color)
    else:
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        im = ax.scatter(data[0][:,0],data[0][:,1],s=5,c=color)
        plt.tight_layout()
    ani = animation.FuncAnimation(fig, animate_move, fargs=(data,im,title_,dim,timescale), frames=nframe, interval=20, blit=False, save_count=50)
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(filename+'.mp4', writer=writer)
    