import numpy as np

def read_binary(filename):
    output = []
    with open(filename, "rb") as fin:
        fin.readline()
        for aline in fin:
            tmp = ""
            output.append([])
            for achar in aline:
                if achar == ord('\n'):
                    output[-1].append(tmp)
                    tmp = ""
                    break
                elif achar==ord(','):
                    output[-1].append(tmp)
                    tmp = ""
                else:
                    if achar <= 3:
                        tmp = str(achar)
                    else:
                        tmp += chr(achar)
    return output

def readttkcriticalpairs(prefix,csv_or_txt="txt"):
    dim = 3
    critical_pair_ = []
    pair_type_ = []
    ttk_vertexscalarfield = []
    with open("%s.%s"%(prefix,csv_or_txt)) as fin:
        fin.readline()
        for aline in fin:
            linelist = aline.strip().split(',')
            critical_pair_.append([float(i) for i in linelist[5:5+dim]])
            pair_type_.append(int(linelist[1]))
            ttk_vertexscalarfield.append(int(linelist[0]))
    critical_pair_ = np.array(critical_pair_)
    ttk_vertexscalarfield = np.array(ttk_vertexscalarfield)
    pair_type_ = np.array(pair_type_)
    if critical_pair_[2].all() == 0:
        dim = 2
        critical_pair_ = critical_pair_[:,:2]
    return critical_pair_, pair_type_, dim, ttk_vertexscalarfield

def readttkmorsesmale(prefix,csv_or_txt="txt",sep_interval=2):
    dim = 3
    ms_critical_point_  = [] # (N x 2) coordinate of each critical point
    ms_critical_type_   = [] # (N x 1) 2: max, 1: saddle, 0: min
    ms_critical_cellid_ = [] # (N x 1) id of each critical point
    ttk_vertexscalarfield = []
    '''
    ms_max_point_       = [] # (N x 1) coordinate of each maximum point
    ms_sad_point_       = [] # (N x 1) coordinate of each saddle point
    ms_max_cellid_      = [] # (N x 1) id of each maximum point
    ms_sad_cellid_      = [] # (N x 1) id of each saddle point
    '''

    ms_1sep_point_         = [] # (M x 2) coordinate of each 1-separation point
    ms_1sep_cellid_        = [] # (M x 1) id of each 1-separation point
    ms_1sep_celldim        = [] # (M x 1) 3: body, 2: face, 1: edge, 0: point
    ms_1sep_sep_type_      = [] # (M x 1) dim-1: accending separation, 0: decending separation
    ms_1sep_startid_       = [] # (K x 1) source id of each 1-separation point
    ms_1sep_id_            = [] # (K x 1) id of each 1-separation point
    ms_1sep_endid_         = [] # (K x 1) destination id of each 1-separation point

    output = read_binary("%s-critical-points.%s"%(prefix,csv_or_txt))
    for linelist in output:
        if len(linelist) > 0:
            ms_critical_point_.append([float(i) for i in linelist[6:6+dim]])
            ms_critical_type_.append(int(linelist[0]))
            ms_critical_cellid_.append(int(linelist[1]))
            ttk_vertexscalarfield.append(int(linelist[4]))
       
    ms_critical_point_ = np.array(ms_critical_point_)
    ms_critical_type_ = np.array(ms_critical_type_)
    ms_critical_cellid_ = np.array(ms_critical_cellid_)
    ttk_vertexscalarfield = np.array(ttk_vertexscalarfield)
    if ms_critical_point_[2].all() == 0:
        dim = 2
        ms_critical_point_ = ms_critical_point_[:,:2]

    output = read_binary("%s-1-sep-points.%s"%(prefix,csv_or_txt))
    for linelist in output:
        if len(linelist) > 0:
            ms_1sep_point_.append([float(i) for i in linelist[3:3+dim]])
            ms_1sep_cellid_.append(int(linelist[2]))
            ms_1sep_celldim.append(int(linelist[1]))
                
    output = read_binary("%s-1-sep-cells.%s"%(prefix,csv_or_txt))
    for linelist in output:
        if len(linelist) > 0:
            ms_1sep_startid_.append(int(linelist[0]))
            ms_1sep_endid_.append(int(linelist[1]))
            ms_1sep_id_.append(int(linelist[2]))
            ms_1sep_sep_type_.append(int(linelist[3]))

    sep_point_ = []                   # (W x m x 2) W = the total number of separation lines (accending + decending)
    sep_celldim = []
    sep_point_start_end_type_ = []    # (W x 2) (source id, destination id, separation type)
    startid = -1
    endid = -1
    sepid = -1
    offset = 0
    for index in range(len(ms_1sep_sep_type_)):
        if sepid != ms_1sep_id_[index]:
            sepid = ms_1sep_id_[index]
            startid = ms_1sep_startid_[index]
            endid = ms_1sep_endid_[index]
            atype = ms_1sep_sep_type_[index]
            sep_point_start_end_type_.append([startid,endid,atype])
            sep_point_.append([])
            sep_celldim.append([])
            offset += 1
        if ms_1sep_celldim[index+offset] >= 0:
            sep_point_[len(sep_point_)-1].append(ms_1sep_point_[index+offset])
            sep_celldim[len(sep_celldim)-1].append(ms_1sep_celldim[index+offset])    

    for index in range(len(sep_point_)-1,-1,-1):
        _,_,atype = sep_point_start_end_type_[index]
        if atype == -1:
            del sep_point_[index]
            del sep_point_start_end_type_[index]

    '''
    ms_max_point_ = np.array(ms_max_point_)
    ms_sad_point_ = np.array(ms_sad_point_)
    ms_max_cellid_ = np.array(ms_max_cellid_)
    ms_sad_cellid_ = np.array(ms_sad_cellid_)
    '''
    ms_1sep_point_ = np.array(ms_1sep_point_)
    ms_1sep_cellid_ = np.array(ms_1sep_cellid_)
    ms_1sep_startid_ = np.array(ms_1sep_startid_)
    ms_1sep_endid_ = np.array(ms_1sep_endid_)

    for sep_index in range(len(sep_point_)):
        sep_point_[sep_index] = np.array(sep_point_[sep_index])
        if sep_point_[sep_index].shape[0] > 2:
            sep_point_[sep_index] = sep_point_[sep_index][1::sep_interval]
    ms_critical_cellid_tmp_ = list(ms_critical_cellid_)
    for sep_index in range(len(sep_point_)):
        sep_point_start_end_type_[sep_index][0] = ms_critical_cellid_tmp_.index(sep_point_start_end_type_[sep_index][0])
        sep_point_start_end_type_[sep_index][1] = ms_critical_cellid_tmp_.index(sep_point_start_end_type_[sep_index][1])
    return ms_critical_point_,ms_critical_type_, sep_point_, sep_point_start_end_type_, sep_celldim, ttk_vertexscalarfield
    
    
    
    