import numpy as np

def point_to_density_3d(bin, vol, point_data, density=True):
    hist, edges = np.histogramdd(point_data, bins = (bin[0], bin[1], bin[2]), range = ((vol[0],vol[1]),(vol[2],vol[3]),(vol[4],vol[5])), density=density)
    
    xmin = vol[0] + (vol[1]-vol[0])/bin[0]/2
    xmax = vol[1] - (vol[1]-vol[0])/bin[0]/2
    ymin = vol[2] + (vol[3]-vol[2])/bin[1]/2
    ymax = vol[3] - (vol[3]-vol[2])/bin[1]/2
    zmin = vol[4] + (vol[5]-vol[4])/bin[2]/2
    zmax = vol[5] - (vol[5]-vol[4])/bin[2]/2
    spacing = np.array([edges[0][1]-edges[0][0],edges[1][1]-edges[1][0],edges[2][1]-edges[2][0]])
    xo, yo, zo = np.ogrid[xmin:xmax:bin[0]*1j,ymin:ymax:bin[1]*1j,zmin:zmax:bin[2]*1j]
    xm, ym, zm = np.mgrid[xmin:xmax:bin[0]*1j,ymin:ymax:bin[1]*1j,zmin:zmax:bin[2]*1j]                
    return hist, xo, yo, zo, xm, ym, zm, spacing


def point_to_density_2d(bin, vol, point_data, density=True):
    hist, edges = np.histogramdd(point_data, bins = (bin[0], bin[1]), range = ((vol[0],vol[1]),(vol[2],vol[3])), density=density)
    
    xmin = vol[0] + (vol[1]-vol[0])/bin[0]/2
    xmax = vol[1] - (vol[1]-vol[0])/bin[0]/2
    ymin = vol[2] + (vol[3]-vol[2])/bin[1]/2
    ymax = vol[3] - (vol[3]-vol[2])/bin[1]/2
    
    spacing = np.array([edges[0][1]-edges[0][0],edges[1][1]-edges[1][0],1])
    xo, yo = np.ogrid[xmin:xmax:bin[0]*1j,ymin:ymax:bin[1]*1j]
    xm, ym = np.mgrid[xmin:xmax:bin[0]*1j,ymin:ymax:bin[1]*1j]                
    return hist, xo, yo, xm, ym, spacing

def tovtk(data,spacing,origin,filename):
    from tvtk.common import configure_input
    from tvtk.api import tvtk
    # if len(data.shape) == 2:
    #     spacing = np.append(spacing,1)
    #     origin = np.append(origin,0)
    #     shape = list(data.shape)+[1]
    # else:
    #     shape = list(data.shape)
    # data = np.random.random((10,10,10))
    #print(dir(tvtk.XMLStructuredDataWriter))
    data = data.astype('f')
    dims = data.shape
    # x, y, z = ogrid[origin[0]:origin[0]+spacing[0]*dims[0]:dims[0]*1j,
    #                 origin[1]:origin[1]+spacing[1]*dims[1]:dims[1]*1j,
    #                 origin[2]:origin[2]+spacing[2]*dims[2]:dims[2]*1j]
    # x, y, z = [t.astype('f') for t in (x, y, z)]
    spoints = tvtk.StructuredPoints(spacing=spacing, origin=origin, dimensions=dims)
    s = data.transpose().copy()
    spoints.point_data.scalars = np.ravel(s)
    #spoints.point_data.scalars = np.ravel(data,order='F')
    spoints.point_data.scalars.name = 'scalars'


    # Writes legacy ".vtk" format if filename ends with "vtk", otherwise
    # this will write data using the newer xml-based format.
    # write_data(spoints, filename+'.vti')
    # Uncomment the next two lines to save the dataset to a VTK XML file.
    writer = tvtk.XMLImageDataWriter(file_name=filename+'.vti')
    #writer = tvtk.XMLStructuredGridWriter(file_name=filename+'.vti')
    # for a in dir(spoints):
    #     print(a)
    configure_input(writer, spoints) # <== will work
    writer.write()