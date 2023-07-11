import numpy as np
from numpy import *
from scipy import *
import matplotlib.pyplot as plt
from PSTD_box_func import *
from PSTD_box_input import *

# Extended PSTD method to solve sound propagation
#
# 2D propagation in a box
#   o box boundaries are rigid or have a finite impedance
#   o box boundaries are located at velocity nodes
#   o box boundaries are locally reacting
#
#Created by Maarten Hornikx, 2023-07, based on a Python script prepared for openPSTD in 2012


#--------------------------------------------------------------------------
# fixed input
rho = 1.2				# density of air
c1 = 340.				# sound speed of air
dxv = array([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.]) ###???What is this?
(dxv<c1/freqmax/2.).nonzero()  ###???What is this?
dxtemp = dxv.compress((dxv<c1/freqmax/2.).flat) ###???What is this?
# spatial discretization
dx = dxtemp[-1]
dz = dx

Nxtot = round(xdist/dx)+2*PMLcells     # Number of computational points in x direction
Nztot = round(zdist/dz)+2*PMLcells     # Number of computational points in z direction

# Number of computational points in x and z direction for the box (Nx, Nz)
# and the subdomains representing the boundaries (left, right, lower,
# upper)

Nxleft = PMLcells
Nxright = PMLcells
Nxlower = round(xdist/dx)
Nxupper = round(xdist/dx)
Nzleft = round(zdist/dz)
Nzright = round(zdist/dz)
Nzlower = PMLcells
Nzupper = PMLcells
Nx = round(xdist/dx)
Nz = round(zdist/dz)

# source position in numbers of samples, relative to center of grid
sx = spos[0]/dx
sz = spos[1]/dz

# receiver positions in numbers of samples
rx =  PMLcells+Nx/2.+1./2.+rpos[:,0]/dx
rz =  PMLcells+Nz/2.+1./2.+rpos[:,1]/dz

fs = c1/dx              # spatial sample frequency
b = 3*pow(10,-6)*pow(fs,2)      # normalized source bandwidth
tfactRK = 0.5          # CFL number RK-scheme

# PML data
ampmax = 2.*pow(10.,4.)     # maximum value of PML attenuation


alphaPMLp, alphaPMLu =PML(PMLcells,ampmax,rho)

#--------------------------------------------------------------------------
# CALCULATION SECTION
#--------------------------------------------------------------------------

# impedances corresponding to the absorption coefficients
# using alfa = 1-((Z-1)/(Z+1))^2

Zleft = -((math.sqrt(1.-alfaleft)+1.)/(math.sqrt(1.-alfaleft)-1))
Zright = -((math.sqrt(1.-alfaright)+1.)/(math.sqrt(1.-alfaright)-1))
Zlower = -((math.sqrt(1.-alfalower)+1.)/(math.sqrt(1.-alfalower)-1))
Zupper = -((math.sqrt(1.-alfaupper)+1.)/(math.sqrt(1.-alfaupper)-1))

# matrices of reflection coefficients for calculation

if Zleft <= 1000:
    rholeft = rho*Zleft
else:
    rholeft = 1e200

if Zright <= 1000:
    rhoright = rho*Zright
else:
    rhoright = 1e200

Rmatrixhor,Rmatrixhorvel= Rmatrices(rholeft,rhoright,rho) #calls the function Rmatrices

if Zlower <= 1000:
    rholower = rho*Zlower
else:
    rholower = 1e200

if Zupper <= 1000:
    rhoupper = rho*Zupper
else:
    rhoupper = 1e200
    
Rmatrixvert,Rmatrixvertvel= Rmatrices(rholower,rhoupper,rho)


# offset receiver positions from grid
dxstagg = rx-floor(rx)
dzstagg = rz-floor(rz)

# temporal discretization
dtRK = tfactRK*dx/c1 # time step
TRK = round(calctime/dtRK) # number of time steps

# coefficients in Runge Kutta 6 method (acc. to Bogey and Bailly 2004)

alfa = zeros((6,1)) #absorption coefficient at different frequencies?
alfa[5] = 1.
alfa[4] = 1./2.
alfa[3] = 0.165919771368/alfa[4]
alfa[2] = 0.040919732041/(alfa[3]*alfa[4])
alfa[1] = 0.007555704391/(alfa[2]*alfa[3]*alfa[4])
alfa[0] = 0.000891421261/(alfa[1]*alfa[2]*alfa[3]*alfa[4])

# calculation of numerical wave numbers
# kxrig = wave number discretization in x-direction for lower and upper material
# kxbox = wave number discretization in x-direction in box
# kxmat = wave number discretization in x-direction for left and right boundaries
# kzrig = wave number discretization in z-direction for left and right material
# kzbox = wave number discretization in z-direction in box
# kzmat = wave number discretization in z-direction for lower and upper boundaries
kxrig,jfactxrig,kxbox,jfactxbox,kxmat,jfactxmat = kcalc(dx,2*Nx,Nx,PMLcells) #calls the function kcalc
kzrig,jfactzrig,kzbox,jfactzbox,kzmat,jfactzmat = kcalc(dz,2*Nz,Nz,PMLcells)

# initialization data
u0left = zeros((Nzleft,Nxleft)) #u is horizontal velocity
w0left = zeros((Nzleft,Nxleft)) #w is vertical velocity
p0left = zeros((Nzleft,Nxleft)) #p is pressure
px0left = zeros((Nzleft,Nxleft))
pz0left = zeros((Nzleft,Nxleft))

u0right = zeros((Nzright,Nxright))
w0right = zeros((Nzright,Nxright))
p0right = zeros((Nzright,Nxright))
px0right = zeros((Nzright,Nxright))
pz0right = zeros((Nzright,Nxright))

u0lower = zeros((Nzlower,Nxlower))
w0lower = zeros((Nzlower,Nxlower))
p0lower = zeros((Nzlower,Nxlower))
px0lower = zeros((Nzlower,Nxlower))
pz0lower = zeros((Nzlower,Nxlower))

u0upper = zeros((Nzupper,Nxupper))
w0upper = zeros((Nzupper,Nxupper))
p0upper = zeros((Nzupper,Nxupper))
px0upper = zeros((Nzupper,Nxupper))
pz0upper = zeros((Nzupper,Nxupper))

u0 = zeros((Nz,Nx))
w0 = zeros((Nz,Nx))
p0 = zeros((Nz,Nx))
px0 = zeros((Nz,Nx))
pz0 = zeros((Nz,Nx))

u0 = zeros((Nz,Nx))
w0 = zeros((Nz,Nx))
p0 = zeros((Nz,Nx))
px0 = zeros((Nz,Nx))
pz0 = zeros((Nz,Nx))

# initial pressure distribution:
# matrix of distances from grid points to source position, normalized to center
    
sdist = 1j*abs(sz*dz-arange(-dz*(Nz-1.)/2.,dz*Nz/2.,dz)).reshape(-1,1)*ones((1,Nx))+\
    ones((Nz,1))*abs(sx*dz-arange(-dx*(Nx-1)/2.,dx*Nx/2.,dx))
# initial values p0
p0 = exp(-b*pow(abs(sdist),2))

# initial values of px0 and pz0
px0 = pow(cos(angle(sdist)),2.)*p0
pz0 = pow(sin(angle(sdist)),2.)*p0

##
## matrices constructed for staggered grid
##
xfactprig = exp(jfactxrig*kxrig*dx/2.)*jfactxrig*kxrig #what is rig? box? mat?
xfactmrig = exp(-jfactxrig*kxrig*dx/2.)*jfactxrig*kxrig
xfactpbox = exp(jfactxbox*kxbox*dx/2.)*jfactxbox*kxbox
xfactmbox = exp(-jfactxbox*kxbox*dx/2.)*jfactxbox*kxbox
xfactpmat = exp(jfactxmat*kxmat*dx/2.)*jfactxmat*kxmat
xfactmmat = exp(-jfactxmat*kxmat*dx/2.)*jfactxmat*kxmat

zfactprig = exp(jfactzrig*kzrig*dz/2.)*jfactzrig*kzrig
zfactmrig = exp(-jfactzrig*kzrig*dz/2.)*jfactzrig*kzrig
zfactpbox = exp(jfactzbox*kzbox*dz/2.)*jfactzbox*kzbox
zfactmbox = exp(-jfactzbox*kzbox*dz/2.)*jfactzbox*kzbox
zfactpmat = exp(jfactzmat*kzmat*dz/2.)*jfactzmat*kzmat
zfactmmat = exp(-jfactzmat*kzmat*dz/2.)*jfactzmat*kzmat

# matrices constructed to include PML

ufact = exp(-alphaPMLu/rho*dtRK)
pfact = exp(-alphaPMLp*dtRK)

alphaptemppxl= ones((Nz,1))*pfact[arange(pfact.shape[0]-1,-1,-1)] #x left?
alphaptemppxr = ones((Nz,1))*pfact[0:]                            #x right
alphaptemppzl = pfact[arange(pfact.shape[0]-1,-1,-1)].reshape(-1,1)*ones((1,Nx)) #z left
alphaptemppzr = pfact[0:].reshape(-1,1)*ones((1,Nx))

alphaptempul = ones((Nz,1))*ufact[arange(ufact.shape[0]-2,-1,-1)]
alphaptempur = ones((Nz,1))*ufact[1:]

alphaptempwl = ufact[arange(ufact.shape[0]-2,-1,-1)].reshape(-1,1)*ones((1,Nx))
alphaptempwr = ufact[1:].reshape(-1,1)*ones((1,Nx)) 

prec = zeros((TRK,rx.shape[0]))
##
###__________________________________________________________________________
##
### loop over time steps

for ii in range(1,int(TRK+1)):
    print(ii)
    for ss in range(1,7):
        # update after eacht time step

        if ss == 1:
            px0leftold = px0left
            pz0leftold = pz0left
            u0leftold = u0left
            w0leftold = w0left

            px0rightold = px0right
            pz0rightold = pz0right
            u0rightold = u0right
            w0rightold = w0right

            px0lowerold = px0lower
            pz0lowerold = pz0lower
            u0lowerold = u0lower
            w0lowerold = w0lower

            px0upperold = px0upper
            pz0upperold = pz0upper
            u0upperold = u0upper
            w0upperold = w0upper
            
            px0old = px0
            pz0old = pz0
            u0old = u0
            w0old = w0
            # initialization and update of matrices after each stage within a time step

        Lpxleft = zeros((Nzleft,Nxleft))
        Lpzleft = zeros((Nzleft,Nxleft))
        Luxleft = zeros((Nzleft,Nxleft))
        Lwzleft = zeros((Nzleft,Nxleft))

        Lpxright = zeros((Nzright,Nxright))
        Lpzright = zeros((Nzright,Nxright))
        Luxright = zeros((Nzright,Nxright))
        Lwzright = zeros((Nzright,Nxright))

        Lpxlower = zeros((Nzlower,Nxlower))
        Lpzlower = zeros((Nzlower,Nxlower))
        Luxlower = zeros((Nzlower,Nxlower))
        Lwzlower = zeros((Nzlower,Nxlower))

        Lpxupper = zeros((Nzupper,Nxupper))
        Lpzupper = zeros((Nzupper,Nxupper))
        Luxupper = zeros((Nzupper,Nxupper))
        Lwzupper = zeros((Nzupper,Nxupper))
        
        Lpx = zeros((Nz,Nx))
        Lpz = zeros((Nz,Nx))
        Lux = zeros((Nz,Nx))
        Lwz = zeros((Nz,Nx))

        #--------------------------------------------------------------
        # horizontal pressure derivatives
        #--------------------------------------------------------------
        # derivatives in box and left and right boundaries
        Lp=spatderp3(concatenate((p0left, p0, p0right), axis=1),\
                  xfactpbox,xfactpmat,arange(1,Nz+1),pow(2,ceil(log2(Nx+2*PMLcells))),\
                  Rmatrixhor,PMLcells,PMLcells,PMLcells+Nx,PMLcells,1,PMLcells)
        Lpxleft[0:Nz,0:PMLcells] = Lp[0:Nz,0:PMLcells]
        Lpx[0:Nz,0:Nx] = Lp[0:Nz,PMLcells:PMLcells+Nx]
        Lpxright[0:Nz,0:PMLcells] = Lp[0:Nz,PMLcells+Nx:2*PMLcells+Nx]

        #--------------------------------------------------------------
        # vertical pressure derivative
        #--------------------------------------------------------------

        # derivatives in box and lower and upper boundaries
        Lp=spatderp3(concatenate((p0lower.transpose(), p0.transpose(), p0upper.transpose()), axis=1),\
                  zfactpbox,zfactpmat,arange(1,Nx+1),pow(2,ceil(log2(Nz+2*PMLcells))),\
                  Rmatrixvert,PMLcells,PMLcells,PMLcells+Nz,PMLcells,1,PMLcells)
        Lpzlower[0:PMLcells,0:Nx] = Lp[0:Nx,0:PMLcells].transpose()
        Lpz[0:Nz,0:Nx] = Lp[0:Nx,PMLcells:PMLcells+Nz].transpose()
        Lpzupper[0:PMLcells,0:Nx] = Lp[0:Nx,PMLcells+Nz:2*PMLcells+Nz].transpose()

        #--------------------------------------------------------------
        # horizontal velocity derivatives
        #--------------------------------------------------------------       

        # derivatives in box and left and right boundaries
        Lu=spatderp3(concatenate((u0left, u0, u0right), axis=1),\
                  xfactmbox,xfactmmat,arange(1,Nz+1),pow(2,ceil(log2(Nx+2*PMLcells))),\
                  Rmatrixhorvel,PMLcells,PMLcells,PMLcells+Nx,PMLcells,2,PMLcells)

        Luxleft[0:Nz,0:PMLcells] = Lu[0:Nz,0:PMLcells]
        Lux[0:Nz,0:Nx] = Lu[0:Nz,PMLcells:PMLcells+Nx]
        Luxright[0:Nz,0:PMLcells] = Lu[0:Nz,PMLcells+Nx:2*PMLcells+Nx]
        
        #--------------------------------------------------------------
        # vertical velocity derivative
        #--------------------------------------------------------------
        
        # derivatives in box and lower and upper boundaries
        Lw=spatderp3(concatenate((w0lower.transpose(), w0.transpose(), w0upper.transpose()), axis=1),\
                  zfactmbox,zfactmmat,arange(1,Nx+1),pow(2,ceil(log2(Nz+2*PMLcells))),\
                  Rmatrixvertvel,PMLcells,PMLcells,PMLcells+Nz,PMLcells,2,PMLcells)

        Lwzlower[0:PMLcells,0:Nx] = Lw[0:Nx,0:PMLcells].transpose()
        Lwz[0:Nz,0:Nx] = Lw[0:Nx,PMLcells:PMLcells+Nz].transpose()
        Lwzupper[0:PMLcells,0:Nx] = Lw[0:Nx,PMLcells+Nz:2*PMLcells+Nz].transpose()      
##
##        # derivatives in left and right boundaries
##        [Lw] = spatderp1r([zeros(1,2*PMLcells);w0lefttemp w0righttemp],zfactmrig,1:2*PMLcells,2*Nzleft,-1,1,2,1);
##        Lwzleft[0:Nzleft,0:PMLcells] = Lw[0:Nzleft,0:PMLcells]
##        Lwzright[0:Nzright,0:PMLcells] = Lw[0:Nzleft,PMLcells:2*PMLcells]

        #--------------------------------------------------------------
        # update of variables
        #--------------------------------------------------------------

        # values in boundary material, only if abs. coeff > 0.005
        # if conditions: if Z > 1000, rigid termination is assumed

        if Zleft <= 1000:
            u0left = u0leftold+(-dtRK*alfa[ss-1]*(1/rholeft*Lpxleft)).real
            w0left = w0leftold+(-dtRK*alfa[ss-1]*(1/rholeft*Lpzleft)).real
            px0left = px0leftold+(-dtRK*alfa[ss-1]*(rholeft*pow(c1,2.)*Luxleft)).real
            pz0left = pz0leftold+(-dtRK*alfa[ss-1]*(rholeft*pow(c1,2.)*Lwzleft)).real

        if Zright <= 1000:
            u0right = u0rightold+(-dtRK*alfa[ss-1]*(1/rhoright*Lpxright)).real
            w0right = w0rightold+(-dtRK*alfa[ss-1]*(1/rhoright*Lpzright)).real
            px0right = px0rightold+(-dtRK*alfa[ss-1]*(rhoright*pow(c1,2.)*Luxright)).real
            pz0right = pz0rightold+(-dtRK*alfa[ss-1]*(rhoright*pow(c1,2.)*Lwzright)).real

        if Zlower <= 1000:
            u0lower = u0lowerold+(-dtRK*alfa[ss-1]*(1/rholower*Lpxlower)).real
            w0lower = w0lowerold+(-dtRK*alfa[ss-1]*(1/rholower*Lpzlower)).real
            px0lower = px0lowerold+(-dtRK*alfa[ss-1]*(rholower*pow(c1,2.)*Luxlower)).real
            pz0lower = pz0lowerold+(-dtRK*alfa[ss-1]*(rholower*pow(c1,2.)*Lwzlower)).real

        if Zupper <= 1000:
            u0upper = u0upperold+(-dtRK*alfa[ss-1]*(1/rhoupper*Lpxupper)).real
            w0upper = w0upperold+(-dtRK*alfa[ss-1]*(1/rhoupper*Lpzupper)).real
            px0upper = px0upperold+(-dtRK*alfa[ss-1]*(rhoupper*pow(c1,2.)*Luxupper)).real
            pz0upper = pz0upperold+(-dtRK*alfa[ss-1]*(rhoupper*pow(c1,2.)*Lwzupper)).real

        u0 = u0old+(-dtRK*alfa[ss-1]*(1/rho*Lpx)).real
        w0 = w0old+(-dtRK*alfa[ss-1]*(1/rho*Lpz)).real
        px0 = px0old+(-dtRK*alfa[ss-1]*(rho*pow(c1,2.)*Lux)).real
        pz0 = pz0old+(-dtRK*alfa[ss-1]*(rho*pow(c1,2.)*Lwz)).real

        
        p0left = px0left+pz0left
        p0right = px0right+pz0right
        p0lower = px0lower+pz0lower
        p0upper = px0upper+pz0upper

        p0 = px0+pz0
    #__________________________________________________________________________
    # Update of values in perfectly matched layer (PML)

    px0left[0:Nzleft,0:PMLcells] = px0left[0:Nzleft,0:PMLcells]*alphaptemppxl[0:Nzleft,0:PMLcells]
    px0right[0:Nzright,0:PMLcells] = px0right[0:Nzright,0:PMLcells]*alphaptemppxr[0:Nzright,:]
    u0left[0:Nzleft,0:PMLcells] = u0left[0:Nzleft,0:PMLcells]*alphaptempul[0:Nz,0:PMLcells]
    u0right[0:Nzright,0:PMLcells] = u0right[0:Nzright,0:PMLcells]*alphaptempur[0:Nz,:]

    pz0lower[0:PMLcells,0:Nxlower] =  pz0lower[0:PMLcells,0:Nxlower]*alphaptemppzl
    w0lower[0:PMLcells,0:Nxlower] =  w0lower[0:PMLcells,0:Nxlower]*alphaptempwl
    pz0upper[0:PMLcells,0:Nxupper] =  pz0upper[0:PMLcells,0:Nxupper]*alphaptemppzr
    w0upper[0:PMLcells,0:Nxupper] =  w0upper[0:PMLcells,0:Nxupper]*alphaptempwr

    p0left = px0left+pz0left
    p0right = px0right+pz0right
    p0lower = px0lower+pz0lower
    p0upper = px0upper+pz0upper
    p0 = px0+pz0

    #--------------------------------------------------------------------------
    # OUTPUT SECTION
    #--------------------------------------------------------------------------
    # Interpolation to receiver positions

    for rr in range(1,rx.shape[0]+1):

        xfactpboxstagg = exp(jfactxbox*kxbox*dxstagg[rr-1]*dx) 
        xfactpmatstagg = exp(jfactxmat*kxmat*dxstagg[rr-1]*dx)
        xfactprigstagg = exp(jfactxrig*kxrig*dxstagg[rr-1]*dx)
        
        prectempbox = spatderp3(concatenate((p0left, p0, p0right), axis=1),\
                                  xfactpboxstagg,xfactpmatstagg,arange(1,Nz+1),pow(2,ceil(log2(Nx+2*PMLcells))),\
                                  Rmatrixhor,PMLcells,PMLcells,PMLcells+Nx,PMLcells,1,PMLcells)
        prectemp = concatenate((p0lower.transpose(), prectempbox[0:Nz,PMLcells:Nx+PMLcells].transpose(), p0upper.transpose()), axis=1)

        zfactpboxstagg = exp(jfactzbox*kzbox*dzstagg[rr-1]*dx)
        zfactpmatstagg = exp(jfactzmat*kzmat*dzstagg[rr-1]*dx)

        prectemp = spatderp3(prectemp,zfactpboxstagg,zfactpmatstagg,arange(1,Nx+1),pow(2,ceil(log2(Nz+2*PMLcells))),\
                               Rmatrixvert,PMLcells,PMLcells,PMLcells+Nz,PMLcells,1,PMLcells)
        prectemp = prectemp.transpose()
        
        
        prec[ii-1,rr-1] = prectemp[round(floor(rz[rr-1])-1),round(floor(rx[rr-1])-1)-PMLcells]


savetxt('ptest.txt',prec)

#Graph for prec number 1
t = np.arange(0, TRK, 1)

fig, ax = plt.subplots()

ax.plot(t,prec[:,0])


#OPEN FILE OF PREC_1 FROM MATLAB (THIS IS ONLY FOR TEST 1)

# Open the file from matlab (Test 1 Receiver 1) in read mode
with open("prec_MAT.txt", 'r') as file:
    # Read each line from the file
    lines = file.readlines()

# Process the lines and extract the values
data = []
for line in lines:
    # Split the line by whitespace and convert values to floats
    values = [float(val) for val in line.strip().split()]
    data.append(values)

# Print the extracted data
for values in data:
    print(values)

rec_1 =[ ]
for i in data:
    rec_1.append(i[0])
ax.plot(t,rec_1)