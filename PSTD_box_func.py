from numpy import *
#from scipy import *
from scipy.fft import fft, ifft

aa=fft(ones((500,2)),axis=1)

def PML(PMLcells,ampmax,rho):
    # PML coefficients in free air
    alphaPMLp = ampmax*pow(arange(0.5,PMLcells-0.5+1.,1.)/PMLcells,4)  # attenuation coefficients of PMLcells for pressure
    alphaPMLu = rho*ampmax*pow(arange(0.,PMLcells+1.,1.)/PMLcells,4)   # attenuation coefficients of PMLcells for velocity

    return alphaPMLp, alphaPMLu

def Rmatrices(rho1,rho2,rho):
    Zn1 = rho1/rho
    Rlw1 = (Zn1-1.)/(Zn1+1.)
    Rlw2 = (pow(Zn1,-1.)-1.)/(pow(Zn1,-1.)+1.)
    Tlw1 = (2.*Zn1)/(Zn1+1)
    Tlw2 = (2.*pow(Zn1,-1.))/(pow(Zn1,-1.)+1.)
    Zn2 = rho2/rho
    Rrw1 = (Zn2-1.)/(Zn2+1.)
    Rrw2 = (pow(Zn2,-1.)-1.)/(pow(Zn2,-1.)+1.)
    Trw1 = (2.*Zn2)/(Zn2+1.)
    Trw2 = (2.*pow(Zn2,-1.))/(pow(Zn2,-1.)+1.)

    # reflection and transmission matrix for pressure
    Rmatrix = array([[Rlw1, Rlw2],[Rrw1, Rrw2],[Tlw1, Tlw2],[Trw1, Trw2]])
    # reflection and transmission matrix for velocity
    Rmatrixvel = array([[-Rlw1, -Rlw2],[-Rrw1, -Rrw2],[Tlw2, Tlw1],[Trw2, Trw1]])

    return Rmatrix, Rmatrixvel

def kcalc(dx,N,cw,PMLcellsghost):
    # wavenumber discretization
    kmax = pi/dx
    dk = kmax/(N/2)               # wave number discretization
    k = concatenate((arange(0,kmax+dk,dk),  arange(kmax-dk,0,-dk)), axis=None)   
    jfact = 1j*concatenate((ones((1,round(N/2)+1)), -ones((1,round(N/2)-1))), axis=None)

    dkgh = kmax/pow(2,(ceil(log2(cw/2+PMLcellsghost))))          
    kcy =  concatenate((arange(0,kmax+dkgh,dkgh),  arange(kmax-dkgh,0,-dkgh)))   
    jfactcy = 1j*concatenate((ones((1,round(pow(2,(ceil(log2(cw/2+PMLcellsghost)))))+1)), -ones((1,round(pow(2,(ceil(log2(cw/2+PMLcellsghost)))))-1))), axis=None)

    dkgh2 = kmax/pow(2,(ceil(log2(PMLcellsghost))))     
    kgh =  concatenate((arange(0,kmax+dkgh2,dkgh2),  arange(kmax-dkgh2,0,-dkgh2)), axis=None)  
    jfactgh = 1j*concatenate((ones((1,round(pow(2,(ceil(log2(PMLcellsghost)))))+1)), -ones((1,round(pow(2,(ceil(log2(PMLcellsghost)))))-1))), axis=None)
 
    return k, jfact, kcy, jfactcy, kgh, jfactgh

def spatderp3(p,xfactcy,xfactgh,N1,N2,Rmatrix,pos1,size1,pos2,size3,loc,PMLcells):
    # spatial derivative across three media
    #       1         2       3
    # -----------|---------|--------
    #   size1   pos1     pos2  size3

    # N2 = dimension of fft
    # N1 = # of ffts that are applied
    # Rmatrix = [Rl1 Rl2;Rr1 Rr2;Tl1 Tl2;Tr1 Tr2];   % reflection and transmission matrix

    size2 = pos2-pos1
    Lp = zeros((N1.shape[0],size1+size2+size3))

    Acorr=array([[11,     8,     4,     3,    4,     4],
    [25,    16,    10,     8,     9,     8],
    [39,    24,    16,    13,    14,    12],
    [53,    32,    22,    18,    19,    16],
    [67,    40,    28,    23,    24,    20],
    [81,    48,    34,    28,    29,    24],
    [95,    56,    40,    33,    34,    28],
    [109,    64,    46,    38,    39,    32],
    [123,    72,   52,   43,    44,   36,]])

    (Acorr[:,0]<PMLcells/2.).nonzero()
    B = Acorr.compress((Acorr[:,0]<PMLcells/2.).flat)
    if B>= 1:
        alfa = B.shape[0]
    else:
        alfa = 1

    A = array([exp(-alfa*log(10)*pow(arange(-PMLcells/2,PMLcells/2+1)/(PMLcells/2),6))]).reshape(-1,1)
    Gtemp = A
    Gtemp[0:round(PMLcells/2.)]
    G = ones((max(size1+1,size3+1),N1.shape[0]))
    G[0:round(PMLcells/2.),0:N1.shape[0]] = Gtemp[0:round(PMLcells/2.)]*ones((1,N1.shape[0]))
   
    if loc == 1: #% calculation for variable node staggered with boundary
        # pos1 is dx/2 left from boundary 1
        Ktemp=fft(concatenate((p[:,pos1-size1:pos1], Rmatrix[2,0]*p[:,pos1:pos1+size1]*(G[size1-1::-1,0:N1.shape[0]].transpose())\
                               +Rmatrix[0,1]*p[:,pos1-1::-1]), axis=1),int(pow(2,ceil(log2(2*size1)))), axis=1)
        Ltemp = ifft((ones((N1.shape[0],1))*xfactgh[0:int(pow(2,ceil(log2(2*size1))))]*Ktemp),int(pow(2,ceil(log2(2*size1)))), axis=1)
        Lp[0:N1.shape[0],0:size1] =  real(Ltemp[0:N1.shape[0],0:size1])  # Lp in medium 1
        
        Ktemp = fft(concatenate((Rmatrix[2,1]*p[:,pos1-size1:pos1]+Rmatrix[0,0]*p[:,pos1+size1-1:pos1-1:-1]*G[0:size1,0:N1.shape[0]].transpose(), \
                                 p[:,pos1:pos2], \
                                 Rmatrix[3,1]*p[:,pos2:pos2+size3]+Rmatrix[1,0]*p[:,pos2-1:pos2-size3-1:-1]*G[size3-1::-1,0:N1.shape[0]].transpose()), axis=1),int(N2), axis=1)
        Ltemp = ifft((ones((N1.shape[0],1))*xfactcy[0:round(N2)]*Ktemp),round(N2), axis=1)
        Lp[0:N1.shape[0],size1:size1+size2] =  real(Ltemp[0:N1.shape[0],size1:size1+size2]) # Lp in medium 2

 
        Ktemp = fft(concatenate((Rmatrix[3,0]*p[:,pos2-size3:pos2]*(G[0:size3,0:N1.shape[0]].transpose())+Rmatrix[1,1]*p[:,pos2+size3-1:pos2-1:-1], \
           p[:,pos2:pos2+size3]), axis=1),int(pow(2,ceil(log2(2*size3)))), axis=1)
        Ltemp = ifft((ones((N1.shape[0],1))*xfactgh[0:int(pow(2,ceil(log2(2*size3))))]*Ktemp),int(pow(2,ceil(log2(2*size3)))), axis=1)
        Lp[0:N1.shape[0],size1+size2:size1+size2+size3] =  real(Ltemp[0:N1.shape[0],size3:2*size3]) # Lp in medium 3
  
    elif loc == 2:
        # pos1 is at boundary 1, with size1-1 velocity nodes left from boundary 1
        # pos2 is at boundary 2, with size3 velocity nodes right from boundary 2

        Ktemp = fft(concatenate((p[:,pos1-size1:pos1], Rmatrix[2,0]*p[:,pos1:pos1+size1-1]*(G[size1-2::-1,0:N1.shape[0]].transpose())\
                                 +Rmatrix[0,1]*p[:,pos1-2::-1]), axis=1),int(pow(2,ceil(log2(2*size1)))), axis=1)
        Ltemp = ifft((ones((N1.shape[0],1))*xfactgh[0:int(pow(2,ceil(log2(2*size1))))]*Ktemp),int(pow(2,ceil(log2(2*size1)))), axis=1)
        Lp[0:N1.shape[0],0:size1] =  real(Ltemp[0:N1.shape[0],0:size1])  # Lp in medium 1
        
        Ktemp = fft(concatenate((Rmatrix[2,1]*p[:,pos1-size1:pos1-1]+Rmatrix[0,0]*p[:,(pos1+size1-2):pos1-1:-1]*(G[0:size1-1,0:N1.shape[0]].transpose()), \
                                 p[:,pos1-1:pos2], \
                                 Rmatrix[3,1]*p[:,pos2:pos2+size3]+Rmatrix[1,0]*p[:,pos2-2:pos2-size3-2:-1]*G[size3-1::-1,0:N1.shape[0]].transpose()), axis=1),int(N2), axis=1)
        Ltemp = ifft((ones((N1.shape[0],1))*xfactcy[0:round(N2)]*Ktemp),int(N2), axis=1)
        Lp[0:N1.shape[0],size1:size1+size2] =  real(Ltemp[0:N1.shape[0],size1:size1+size2]) # Lp in medium 2
        
        Ktemp = fft(concatenate((Rmatrix[3,0]*p[:,pos2-size3-1:pos2]*(G[0:size3+1,0:N1.shape[0]].transpose())+Rmatrix[1,1]*p[:,pos2+size3-1:pos2-2:-1], \
                                 p[:,pos2:pos2+size3]), axis=1),int(pow(2,ceil(log2(2*size3)))), axis=1)
        Ltemp = ifft((ones((N1.shape[0],1))*xfactgh[0:int(pow(2,ceil(log2(2*size3))))]*Ktemp),int(pow(2,ceil(log2(2*size3)))), axis=1)
        Lp[0:N1.shape[0],size1+size2:size1+size2+size3] =  real(Ltemp[0:N1.shape[0],size3+1:2*size3+1]) # Lp in medium 3

    return Lp
