import numpy as np
from numpy.fft import fft
import os
import scipy.ndimage.filters as nd_filters
import cv2

def matlab_factorial(l):
    res = []
    for elt in l:
        if not len(res):
            if elt == 0:
                res.append(1)
            else:
                res.append(elt)
        else:
            res.append(elt * l[-1])
    return res


class OFilter:
    def __init__(self, order, mask_size, mode='symmetric'):
        self.order = order
        self.mask_size = mask_size
        self.mode = mode
        
    def local_filter(self, x):
        x.sort()
        return x[self.order]

    def ordfilt2(self, A):
        return nd_filters.generic_filter(np.pad(A, 1, self.mode), self.local_filter, size=(self.mask_size, self.mask_size))
        

def bf_filter(x,bfdata):
    # filtering bank
    #
    # x      is a 2d matrix
    # bfdata is the filter bank, it is a structure with the following elements:
    #  .number     number of filters, 
    #  .bf         a 3d matrix with the filters, 
    #  .factor     factor of the filters. 
    #
    n_filters, bf, factor = bfdata['number'], bfdata['bf'], bfdata['factor']
    filtered = np.zeros((x.shape[0], x.shape[1], n_filters))
    for idx in range(n_filters):
        filtered[:,:,idx] = factor[idx] * cv2.filter2D(x, -1, np.conj(bf[:,:,idx]))
    return filtered

def const_bf(SZ,ORDER):
    #Compute the simple CHT functions
    #  bfdata = const_bf(SZ,ORDER)
    # 
    #    SZ is the size of patch
    # ORDER is the number of basis functions
    #
    # bfdata is a structure with the following elements:
    #  .number     number of basis functions, 
    #  .orders     a matrix with 2 columns, where the first column indicates n and the second column m, 
    #  .bf         a 3d matrix with the basis functions, 
    #  .factor     factor for ortonormal basis functions. 
    #

    NM = np.concatenate(np.zeros((ORDER,1)), np.arange(ORDER),axis=1)
    BF = np.zeros((SZ, SZ, NM.shape[0]))
    WF = np.zeros((NM.shape[0], 1))
    
    X, Y = np.meshgrid(np.arange(1,SZ+1), np.arange(1,SZ+1))
    rho   =  np.sqrt( (2.*Y - SZ - 1) ** 2 + (2.*X - SZ - 1) ** 2 ) / SZ
    theta = np.arctan2( -(2.*Y - SZ - 1),(2*X - SZ -1) )
    mask  = rho<=1
    cnt   = mask.sum()
    for idx in range(NM.shape[0]):
        m = NM[idx,1]
        BF[:,:,idx] = mask * np.exp(1j * m * theta)
        WF[idx ]    = np.sqrt(1. / cnt)

    # ---- Package the whole thing into a dictionnary
    bfdata = {}
    bfdata['maxorder'] = ORDER
    bfdata['orders']   = NM
    bfdata['bf']       = BF
    bfdata['factor']   = WF

    return bfdata, mask

def quadfilt_err(x,V):
    err = cv2.filter2D(x ** 2, V[:,:,0], 'symmetric') - ( \
        cv2.filter2D(x, -1, V[:,:,1], 'symmetric') ** 2 + \
        cv2.filter2D(x, -1, V[:,:,2], 'symmetric') ** 2 + \
        cv2.filter2D(x, -1, V[:,:,3], 'symmetric') ** 2 )
    return err

def DLFerror(mpf_y,mpf_x,rd):
# Compute the squared error of dense linear fitting
#   e = DLFerror(mpf_y,mpf_x,rd)
# 
#    mpf_x   column indexes;
#    mpf_y   row indexes;
#    rd      radius of circular neighborhood.
#


    EE_mvf,AA_mvf,VV_mvf = mvfAffineMtx2Error(rd)
    e = quadfilt_err(mpf_y,VV_mvf) + quadfilt_err(mpf_x,VV_mvf)
    return e
    
def FMTpl_bf(SZ=24, freq_m=np.arange(5), radiusNum=26, anglesNum=32, freq_n=np.arange(26), radiusMin=np.sqrt(2)):
    # Comupte the Fourier-Mellin Transform functions with log-polar resample
    #   bfdata = FMTpl_bf(SZ, freq_m, radiusNum, anglesNum, freq_n, radiusMin);
    # 
    #         SZ is the size of patch
    #     freq_m is a vector which indicates m
    #  radiusNum is the number of sampling points along the radius
    #  anglesNum is the number of sampling points along the circumferences
    #     freq_n is a vector which indicates n
    #  radiusMin is the minimun radius
    # 
    #  bfdata is a structure with the following elements:
    #   .number     number of basis functions, 
    #   .orders     a matrix with 2 columns, where the first column indicates n and the second column m, 
    #   .bf         a 3d matrix with the basis functions, 
    #   .factor     factor for ortonormal basis functions, 
    #   .freq0      is the frequency step.
    #   
    
    num_freq_m = np.prod(freq_m.shape)
    num_freq_n = np.prod(freq_n.shape)
    
    gf = lambda x,y : max(0,1-abs(x)) * max(0,1-abs(y)) # linear
    # gf = lambda x,y :  rectFunc(x).*rectFunc(y)        # nearest
    # gf = lambda x,y :  np.exp(-(x**2 + y**2) / 2) / (2*np.pi)  # gaussian
    
    X,Y = np.meshgrid(np.arange(1,SZ+1),np.arange(1,SZ+1))
    Y = (2.*Y - SZ - 1) / 2
    X = (2.*X - SZ - 1) / 2
    
    radiusMax = (SZ - 1)/2
    radiusPoints = (2 ** (np.linspace(np.log2(radiusMin),np.log2(radiusMax),radiusNum)))
    
    BF = np.zeros(SZ, SZ, anglesNum, radiusNum)
    for idxA in range(anglesNum):
        A  = (idxA-1) * 360 / anglesNum
        for idxR in range(radiusNum):
            R = radiusPoints(idxR)
            pX = X - R*np.cosd(A)
            pY = Y - R*np.sind(A)
            J = gf(pX,pY)
            # J = J/J.sum()
            BF[:,:,idxA,idxR] = J

    BF = fft(BF,axis=2) / anglesNum
    BF = fft(BF,axis=3)
    BF = BF[:, :, freq_m, freq_n%radiusNum]
    
    n,m = np.meshgrid(freq_n,freq_m)
    WF = np.ones(num_freq_m,num_freq_n)

    bfdata = {}
    bfdata['number']   = num_freq_m*num_freq_n
    bfdata['orders']   = np.concatenate(n,m,axis=1)
    bfdata['bf']       = BF
    bfdata['factor']   = WF
    bfdata['radiusStep']  = np.log(radiusPoints[-1]) - np.log(radiusPoints[-2])
    bfdata['freq0']       = 1 / (radiusNum*bfdata['radiusStep'])
    return bfdata

def PCT_bf(SZ=16, ORDER=3):
    # Comupte the Polar Cosine Transform functions
    #   bfdata = PCT_bf(SZ,NM)
    #  
    #     SZ is the size of patch
    #     NM is a matrix with 2 columns, where the first column indicates n and the second column m.
    # 
    #  bfdata is a structure with the following elements:
    #   .number     number of basis functions, 
    #   .orders     a matrix with 2 columns, where the first column indicates n and the second column m, 
    #   .bf         a 3d matrix with the basis functions, 
    #   .factor     factor for ortonormal basis functions. 
    # 
    
    if np.isscalar(ORDER):
        n,m = np.meshgrid(np.arange(ORDER),np.arange(ORDER))
        NM  = np.concatenate([m.reshape(-1,1),n.reshape(-1,1)],axis=1)
    else:
        NM = ORDER

    ORDER = max(NM[:,0])
  
    BF = np.zeros(SZ,SZ,NM.shape[0])
    WF = np.zeros(NM.shape[0],1)

    X,Y = np.meshgrid(np.arange(1,SZ+1),np.arange(1,SZ+1))
    rho2  = ((2*Y - SZ - 1)**2 + (2*X - SZ - 1)**2) / SZ**2
    theta = np.arctan2(-(2*Y - SZ - 1), (2*X - SZ - 1))
    mask  = rho2<=1
    cnt   = mask.sum()
    for idx in range(NM.shape[0]):
        n = NM(idx,0)
        m = NM(idx,1)
        
        BF[:,:,idx] = mask * np.cos(np.pi*n*rho2) * np.exp(1j*m*theta)
        WF[idx] = (((n>0)+1) / cnt)
    
    bfdata = {}
    bfdata['number'] = NM.shape[0]
    bfdata['orders'] = NM
    bfdata['bf']     = BF
    bfdata['factor'] = WF
    return bfdata

def  PCTp_bf(SZ=16, ORDER=3, radiusNum=26, anglesNum=32):
    # Comupte the Polar Cosine Transform functions with polar resample
    #   bfdata = PCTp_bf(SZ, NM, radiusNum, anglesNum);
    # 
    #         SZ is the size of patch
    #         NM is a matrix with 2 columns, where the first column indicates n and the second column m.
    #  radiusNum is the number of sampling points along the radius
    #  anglesNum is the number of sampling points along the circumferences
    # 
    #  bfdata is a structure with the following elements:
    #   .number     number of basis functions, 
    #   .orders     a matrix with 2 columns, where the first column indicates n and the second column m, 
    #   .bf         a 3d matrix with the basis functions, 
    #   .factor     factor for ortonormal basis functions. 
    # 
        
    if np.isscalar(ORDER):
        [n,m] = np.meshgrid(np.arange(ORDER),np.arange(ORDER))
        NM  = np.concatenate([m.reshape(-1,1),n.reshape(-1,1)],axis=1)
    else:
        NM  = ORDER

    ORDER = NM[:,0].max()
  
    BF = np.zeros(SZ,SZ,NM.shape[0])
    WF = np.zeros(NM.shape[0],1)
    for idx in range(NM.shape[0]):
        n = NM[idx,0]
        WF[idx] = (n>0)+1
    
    gf = lambda x,y : np.max(0,1-np.abs(x)) * np.max(0,1-np.abs(y)) # bi-linear

    X,Y = np.meshgrid(np.arange(1,SZ+1),np.arange(1,SZ+1))
    Y = (2.*Y - SZ - 1) / 2
    X = (2.*X - SZ - 1) / 2
    
    radiusMax = (SZ-1)/2
    radiusMin = 0 #(mod(SZ+1,2))/2;
    radiusPoints = np.linspace(radiusMin,radiusMax,radiusNum)
    
    for idxA in range(anglesNum):
        A  = (idxA-1) * 360 / anglesNum
        for idxR in range(radiusNum):
            R = radiusPoints(idxR)
            pX = X - R * np.cosd(A)
            pY = Y - R * np.sind(A)
            J = gf(pX,pY)
            R = 2*R/SZ
            for idx in range(NM.shape[0]):
                n = NM[idx,0]
                m = NM[idx,1]
                Rad = np.cos(np.pi*n*R*R) * (np.cosd(m*A) + 1j*np.sind(m*A))
                BF[:,:,idx] = BF[:,:,idx] + (J @ Rad) @ R

    WF = WF / np.sqrt(np.sum(np.sum(np.abs(BF[:,:,0])**2)))
    
    bfdata = {}
    bfdata['number'] = NM.shape[0]
    bfdata['orders'] = NM
    bfdata['bf']     = BF
    bfdata['factor'] = WF
    return bfdata

def ZM_bf(SZ=16, ORDER=5):
    # Comupte the Zernike momente functions
    #   bfdata = ZM_bf(SZ,ORDER)
    #   bfdata = ZM_bf(SZ,NM)
    #  
    #     SZ is the size of patch
    #  ORDER is the order of Zernike momente
    #     NM is a matrix with 2 columns, where the first column indicates n and the second column m.
    # 
    #  bfdata is a structure with the following elements:
    #   .number     number of basis functions, 
    #   .orders     a matrix with 2 columns, where the first column indicates n and the second column m, 
    #   .bf         a 3d matrix with the basis functions, 
    #   .factor     factor for ortonormal basis functions. 
    # 
    
    if np.isscalar(ORDER):
        NM  = ZM_orderlist(ORDER)
    else:
        NM  = ORDER
    
    ORDER = NM[:,0].max()
  
    BF = np.zeros((SZ,SZ,NM.shape[0]))
    WF = np.zeros((NM.shape[0],1))
    
    F = matlab_factorial(np.arange(NM[:,0].max()+1))

    X,Y = np.meshgrid(np.arange(1,SZ+1),np.arange(1,SZ+1))
    rho  = np.sqrt((2*Y - SZ - 1)**2 + (2*X - SZ - 1)**2) / SZ
    theta = np.arctan2(-(2*Y - SZ - 1), (2*X - SZ - 1))
    mask  = rho<=1
    cnt   = mask.sum()
    for idx in range(NM.shape[0]):
        n = int(NM[idx,0])
        m = NM[idx,1]

        Rad = np.zeros(rho.shape)
        tu = int(np.floor((n+abs(m))/2))
        td = int(np.floor((n-abs(m))/2))
        for s in range(td+1):
           c = ( ((-1)**s) * F[n-s] ) / (F[s] * F[tu-s] * F[td-s])
           Rad = Rad + c * (rho ** (n-2*s))
    
        BF[:,:,idx] = mask * Rad * np.exp(1j*m*theta)
        WF[idx] = np.sqrt((n+1) / cnt)
    
    bfdata = {}
    bfdata['number'] = NM.shape[0]
    bfdata['orders'] = NM
    bfdata['bf']     = BF
    bfdata['factor'] = WF
    return bfdata

def ZM_orderlist(ORDER):
    # Create the moment indices 
    #   NM = ZM_orderlist(ORDER)
    #  
    #  NM is a matrix with 2 columns, where the first column indicates n and the second column m.
    # 
	NM = np.zeros((1,2))
	for n in range(ORDER):
		for m in range(n):
			if not (n-abs(m))%2:
			    NM = np.concatenate([NM, np.array([n, m]).reshape(1,2)], axis=0); return NM

def  ZMp_bf(SZ=16, ORDER=5, radiusNum=26, anglesNum=32):
    # Compute the Zernike momente functions with polar resample
    #   bfdata = ZMp_bf(SZ, NM, radiusNum, anglesNum);
    # 
    #         SZ is the size of patch
    #         NM is a matrix with 2 columns, where the first column indicates n and the second column m.
    #  radiusNum is the number of sampling points along the radius
    #  anglesNum is the number of sampling points along the circumferences
    # 
    #  bfdata is a structure with the following elements:
    #   .number     number of basis functions, 
    #   .orders     a matrix with 2 columns, where the first column indicates n and the second column m, 
    #   .bf         a 3d matrix with the basis functions, 
    #   .factor     factor for ortonormal basis functions. 
    # 
    
    if np.isscalar(ORDER):
        NM  = ZM_orderlist(ORDER)
    else:
        NM  = ORDER

    ORDER = NM[:,0].max()
    F = matlab_factorial(np.arange(ORDER+1))
    
    coef = np.zeros(NM.shape[0],ORDER)
    WF   = np.zeros(NM.shape[0],1)
    for idx in range(NM.shape[0]):
        n = int(NM[idx,1])
        m = NM[idx,2]
        tu = int(np.floor((n+abs(m))/2))
        td = int(np.floor((n-abs(m))/2))
        for s in range(td+1):
           coef[idx,s+1] = ( ((-1)**s) * F[n-s] ) / (F[s] * F[tu-s] * F[td-s])
        WF[idx] = np.sqrt(n+1)
    
    gf = lambda x,y : np.max(0,1-np.abs(x)) * np.max(0,1-np.abs(y)) # bi-linear
    
    X,Y = np.meshgrid(np.arange(1,SZ+1),np.arange(1,SZ+1))
    Y = (2.*Y - SZ - 1) / 2
    X = (2.*X - SZ - 1) / 2
    
    radiusMax = (SZ-1)/2
    radiusMin = 0
    radiusPoints = np.linspace(radiusMin,radiusMax,radiusNum)
    
    BF = np.zeros(SZ,SZ,NM.shape[0])

    for idxA in range(anglesNum):
        A  = (idxA-1) * 360 / anglesNum
        for idxR in range(radiusNum):
            R = radiusPoints(idxR)
            pX = X - R * np.cosd(A)
            pY = Y - R * np.sind(A)
            J = gf(pX,pY)
            R = 2*R/SZ
            for idx in range(NM.shape[0]):
                n = NM[idx,0]
                m = NM[idx,1]
                Rad = 0
                td = int(np.floor((n-np.abs(m))/2))
                for s in range(td+1):
                    Rad = Rad + coef(idx,s+1) * (R ** (n-2*s))
                Rad = Rad * (np.cosd(m*A) + 1j*np.sind(m*A))
                BF[:,:,idx] = BF[:,:,idx] + (J @ Rad) @ R

    WF = WF / np.sqrt(np.sum(np.sum(np.abs(BF[:,:,0])**2)))
    
    bfdata = {}
    bfdata['number'] = NM.shape[0]
    bfdata['orders'] = NM
    bfdata['bf']     = BF
    bfdata['factor'] = WF
    return bfdata

    
def MPFdual(mpf_y,mpf_x,mask):
    # Mirroring of the regions.
    #  mask = MPFdual(mpf_y,mpf_x,mask)
    #    mpf_x   column indexes;
    #    mpf_y   row indexes;
    #    mask    selective mask, with same dimensions of mpf_x
    # 

    Nr, Nc = mpf_y.shape 
    if mask.sum()>0:
        ind = np.argwhere(mask==True).reshape(-1,1)
        ind = (np.round(mpf_y(ind)) + Nr*np.round(mpf_x(ind))+1)
        ind[ind>(Nr*Nc)] = [] # marche ?
        ind[ind<1] = [] # marche ?
        mask[ind] = True # marche ?
    return mask

def MPFregularize(mpf_y,mpf_x,DD_med,NN_med):
    # regularize offsets field by order-statistic filtering
    #  [mpf_y,mpf_x] = MPFregularize(mpf_y,mpf_x,DD_med,NN_med)
    # 
    #    mpf_x   column indexes,
    #    mpf_y   row indexes,
    #    DD_med  the structuring element,
    #    NN_med  order of order-statistic filtering.
    # 
    Nr, Nc = mpf_y.shape
    xp, yp = np.meshgrid(np.arange(Nc),np.arange(Nr))
    mpf_y = OFilter(mpf_y-yp,NN_med,DD_med,'symmetric') + yp # marche pas...
    mpf_x = OFilter(mpf_x-xp,NN_med,DD_med,'symmetric') + xp # marche pas...

    return mpf_y,mpf_x

def MPFspacedist2(mpf_y,mpf_x):
    # Compute the square of spatial distance
    #  dist2 = MPFspacedist2(mpf_y,mpf_x)
    # 
    #    mpf_x   column indexes;
    #    mpf_y   row indexes;
    # 

    Nr, Nc = mpf_y.shape
    xp, yp = np.meshgrid(np.arange(Nc),np.arange(Nr))
    dist2 = (mpf_y-yp)**2 + (mpf_x-xp)**2
    return dist2

# checked
def mvfAffineMtx2Error(r):

    X,Y = np.meshgrid(np.arange(-r,r+1),np.arange(-r,r+1))
    D = (X**2 + Y**2) <= (r**2)
    D = D.reshape(-1,1)
    M = D @ D.T
    x = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1), np.ones(np.prod(X.shape),1)], axis=1).T
    x = x * np.tile(D.T,[3,1])
    B = X.T @ np.linalg.inv(X @ X.T) # pinv(x)
    A = B @ x
    A = M * (A+A.T)/2
    E = np.diag(D) - A
    
    V,S = np.linalg.eig(A)
    V = V[:,np.argwhere(np.diag(S>0.5))]
    V = np.concatenate([D,V], axis=1)
    V = V.reshape(2*r+1, 2*r+1, 4)

    return E,A,V

def padarray_both(mtx, padding , mode):
    # mtx = padarray_both(mtx, padding , type)
    #  padding = [u l d r];
    #
    mtx = np.pad(mtx,((padding[0],padding[2]),(padding[1],padding[3])), constant_values=mode)
    return mtx

def genDisk(radius):
    # generate a disk-shaped structuring element
    #  
    X,Y = np.meshgrid(np.arange(-radius,radius+1),np.arange(-radius,radius+1))
    D = (X**2 + Y**2) <= (radius**2)
    n = np.sum(D)
    return D,n
    
def getFalseColoredResult(value,gt):

    if type(value)==str:
        value = cv2.imread(value)>0 # ça marche?
    if type(gt)==str:
        gt = cv2.imread(gt)
    
    map = np.array([[106, 106, 106], [235, 125, 125], [180, 180, 180], [195, 177, 164], [255, 255, 255], [155, 230, 203]]) / 255.
    gt  = gt>0 + gt==np.max(gt)
    col = (2*gt + value).astype(np.uint8)
    return col,map

def getFmeasure(value,gt):
    # compute the F-measure
    #  [FM,ret] = getFmeasure(map,gt)
    # 

    if type(value)==str:
        value = cv2.imread(value)>0 # ça marche?
    if type(gt)==str:
        gt = cv2.imread(gt)

    gt    = gt.reshape(-1,1)
    value = value.reshape(-1,1)
    gt1   = gt==max(gt)
    gt0   = gt==0

    measure =	{}
    measure['N_TP'] = np.logical_or(value,gt1).sum()
    measure['N_TN'] =  np.logical_or(not(value),gt0).sum()
    measure['N_FP'] =  np.logical_or(value,gt0).sum()
    measure['N_FN'] =  np.logical_or(not(value),gt1).sum()

    measure['FM'] = 2* measure['N_TP'] / (measure['N_TP']+measure['N_FP']+measure['N_TP']+measure['N_FN'])
    measure['TPR'] = measure['N_TP'] / (measure['N_TP']+measure['N_FN']) # = sensitivity = recall
    measure['TNR'] = measure['N_TN'] / (measure['N_TN']+measure['N_FP']) # = specificity
    measure['FNR'] = measure['N_FN'] / (measure['N_TP']+measure['N_FN']) # = 1-sensitivity 
    measure['FPR'] = measure['N_FP'] / (measure['N_TN']+measure['N_FP']) # = 1-specificity = false alarm rate
    measure['PPV'] = measure['N_TP'] / (measure['N_TP']+measure['N_FP']) # = precision  
    measure['NPV'] = measure['N_TN'] / (measure['N_TN']+measure['N_FN']) 
    
    FM = measure['FM']
    return FM,measure

# def displayMPF(img,mpf_x,mpf_y,bsize,mask):
#     # display the offsets field
#     # displayMPF(img,mpf_x,mpf_y,bsize[,mask])
#     #   img     the image;
#     #   mpf_x   column indexes;
#     #   mpf_y   row indexes;
#     #   bsize   sampling;
#     #   mask    selective mask, with same dimensions of mpf_x
#     #
# 
#     rowImg, colImg, K = img.shape
#     rowMpf, colMpf    = mpf_x.shape
#     rowBlock = rowImg-rowMpf + 1
#     colBlock = colImg-colMpf + 1
# 
#     [xp, yp] = np.meshgrid(np.arange(rowMpf),np.arange(rowMpf))
#     mvf_x = mpf_x - xp
#     mvf_y = mpf_y - yp
#     xp = xp + np.ceil((colBlock+1)/2)
#     yp = yp + np.ceil((rowBlock+1)/2)
# 
#     # Takes a vector for block
#     mvf_x = mvf_x[:bsize[0]:rowMpf,:bsize[2]:colMpf]
#     mvf_y = mvf_y[:bsize[0]:rowMpf,:bsize[2]:colMpf]
#     xp = xp[:bsize[0]:rowMpf,:bsize[1]:colMpf]
#     yp = yp[:bsize[0]:rowMpf,:bsize[1]:colMpf]
# 
#     if  os.path.exists('mask','var') and not(isempty(mask)):
#         mask = mask(1:bsize(1):rowMpf,1:bsize(2):colMpf)
#         xp = xp(mask)
#         yp = yp(mask)
#         mvf_x = mvf_x(mask)
#         mvf_y = mvf_y(mask)
# 
#     # show figure
#     h = imshow(img,'Parent',gca()); hold on
#     vh = quiver(xp,yp,mvf_x,mvf_y, 0,'r')
#     set(vh, 'Linewidth', 2,'ShowArrowHead','off')
#     hold off
#     axis off
#     return vh,h

