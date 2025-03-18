# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np
import math as m

def lagrange_red(c11, c22, c12):
    (cc11, cc22, cc12) = (c11, c22, c12)
    if c12 < 0:
        cc12 = -c12
        c12 = cc12
    if c22 < c11:
        # print("ciao")
        a = c11
        c11 = c22
        c22 = a
    if 2*c12 > c11:
        # print("ciaone")
        d11 = c11
        d12 = c12 - c11
        d22 = c22 + c11 - 2 * c12
       # print('d11={:}, d22={:},d12={:}'.format(d11,d22,d12))
        (cc11, cc22, cc12) = lagrange_red(d11, d22, d12)
        #print('cc11={:}, cc22={:},cc12={:}'.format(cc11,cc22,cc12))
        return cc11, cc22, cc12
    #print('cc11={:}, cc22={:},cc12={:}'.format(c11,c22,c12))
    return cc11, cc22, cc12


def fromctodisk(c11, c22, c12):
    yy = 1/c11
    xx = yy*c12
    boh2 = 0
    y = (2*xx/(m.pow(xx, 2)+m.pow(1+yy, 2)))
    x = ((-1+m.pow(xx, 2)+m.pow(yy, 2))/(m.pow(xx, 2)+m.pow(1+yy, 2)))
    ray = m.sqrt(m.pow(x, 2)+m.pow(y, 2))
    if x > 0:
        boh2 = m.atan(y/x)
    elif x < 0 and y >= 0:
        boh2 = m.atan(y/x)+1*np.pi
    elif x < 0 and y < 0:
        boh2 = m.atan(y/x)-1*np.pi
    elif x == 0 and y > 0:
        boh2 = 0.5*np.pi
    elif x == 0 and y < 0:
        boh2 = -0.5*np.pi
    angle = boh2
    return ray, angle


def invert(m11, m12, m21, m22):
    mm11 = m22
    mm22 = m11
    mm12 = -m12
    mm21 = -m21
    m11 = mm11
    m22 = mm22
    m12 = mm12
    m21 = mm21
    return m11, m12, m21, m22


def pathgen(alpha, theta):
    theta = theta*np.pi/180
    c11 = m.pow(alpha, 2)*m.pow(m.sin(theta), 4) + \
        m.pow((1-alpha*m.cos(theta)*m.sin(theta)), 2)
    c22 = m.pow(alpha, 2)*m.pow(m.cos(theta), 4) + \
        m.pow((1+alpha*m.cos(theta)*m.sin(theta)), 2)
    c12 = alpha*m.pow(m.cos(theta), 2)*(1-alpha*m.cos(theta)*m.sin(theta)) - \
        alpha*m.pow(m.sin(theta), 2)*(1+alpha*m.cos(theta)*m.sin(theta))
    yy = 1/c22
    xx = yy*c12
    boh2 = 0
    y = (2*xx/(m.pow(xx, 2)+m.pow(1+yy, 2)))
    x = ((-1+m.pow(xx, 2)+m.pow(yy, 2))/(m.pow(xx, 2)+m.pow(1+yy, 2)))
    ray = m.sqrt(m.pow(x, 2)+m.pow(y, 2))
    if x > 0:
        boh2 = m.atan(y/x)
    elif x < 0 and y >= 0:
        boh2 = m.atan(y/x)+1*np.pi
    elif x < 0 and y < 0:
        boh2 = m.atan(y/x)-1*np.pi
    elif x == 0 and y > 0:
        boh2 = 0.5*np.pi
    elif x == 0 and y < 0:
        boh2 = -0.5*np.pi
    angle = boh2
    return ray, angle

def pathgenrect(alpha,detc):
    alpham=0.5*alpha
    c11=m.pow(m.cosh(alpham)-m.sinh(alpham),2)
    c22=m.pow(m.cosh(alpham)+m.sinh(alpham),2)
    c12=0.
    print("c11: ",c11,"  c22: ",c22,"  c12: ",c12)
    #print("det= ",c11*c22-c12**2)
    yy=m.sqrt(detc)/c22
    xx=m.sqrt(detc)*c12/c22
    boh2=0
    y=(2*xx/(m.pow(xx,2)+m.pow(1+yy,2)))
    x=((-1+m.pow(xx,2)+m.pow(yy,2))/(m.pow(xx,2)+m.pow(1+yy,2)))  
    ray=m.sqrt(m.pow(x,2)+m.pow(y,2))
    if x>0:
        boh2=m.atan(y/x)  
    elif x<0 and y>=0:
        boh2=m.atan(y/x)+1*np.pi
    elif x<0 and y<0:
        boh2=m.atan(y/x)-1*np.pi
    elif x==0 and y>0:
        boh2=0.5*np.pi
    elif x==0 and y<0:
        boh2=-0.5*np.pi      
    angle=boh2
    return ray,angle  



c11 = 1
c22 = 2
c12 = 1
#print('c11={:}, c22={:},c12={:}'.format(c11,c22,c12))
(c11, c22, c12) = lagrange_red(c11, c22, c12)
#print('c11r={:}, c22r={:},c12r={:}'.format(c11,c22,c12))

##
azimuths = np.radians(np.linspace(0, 360, 360))
zeniths = np.linspace(0, .99, 100)
values = np.zeros([azimuths.size, zeniths.size])
r, theta = np.meshgrid(zeniths, azimuths)
##print('azimuths_size={:}, zeniths_size={:}, vals_size={:}'.format(azimuths.size,zeniths.size,values.size))

azimuths2 = np.radians(np.linspace(0, 360, 360))
zeniths2 = np.linspace(0, .99, 100)
values2 = np.zeros((500, 2))
k = 0
boh2 = 0
r2, theta2 = np.meshgrid(zeniths2, azimuths2)



# values2 = np.zeros((100, 2))
# values3 = np.zeros((100, 2))
valuespoints = np.zeros((512, 2))
valuespoints2 = np.zeros((512, 2))
valuespointshex = np.zeros((512, 2))
valuespointshex2 = np.zeros((512, 2))



# for i in range(0, 100):
#     if i != 0:
#         # print(fromctodisk(1/i,i,0))
#         values2[i][1] = fromctodisk(1/i, i, 0)[1]
#         values2[i][0] = fromctodisk(1/i, i, 0)[0]
#         values3[i][1] = fromctodisk(i, 1/i, 0)[1]
#         values3[i][0] = fromctodisk(i, 1/i, 0)[0]
k = 0
for it11 in range(-4, 4):
    for it12 in range(-4, 4):
        for it21 in range(-4, 4):
            if it11 == 0:
                m11 = 0
                m22 = -1
                m12 = 1
                m21 = 1
            elif it11 != 0:
                m11 = int(it11)
                m12 = int(it12)
                m21 = int(it21)
##               print(isinstance(m12, int))
                m22t = int((1+int(it12)*int(it21))/int(it11))
                if (m22t*m11-m12*m21) == 1:
                    m22 = m22t
                    m11 = int(it11)
                    m12 = int(it12)
                    m21 = int(it21)
                    #print("Determinant is {:}".format(m22*m11-m12*m21))
                else:
                    m11 = 1
                    m22 = 1
                    m12 = 0
                    m21 = 0

            c11 = m22*m22+m21*m21
            c22 = m12*m12+m11*m11
            c12 = -m22*m12-m21*m11
            gammasq = m.sqrt(4/3)
            cc11 = gammasq*(m22*m22+m21*m21-m21*m22)
            cc22 = gammasq*(m12*m12+m11*m11-m12*m11)
            cc12 = gammasq*(-m12*m22+0.5*m11*m22+0.5*m21*m12-m21*m11)
##            print('c11={:}, c22={:},c12={:}'.format(c11,c22,c12))
            if c11 != 0:
                # valuespoints[k][1] = fromctodisk(c11, c22, c12)[1]
                # valuespoints[k][0] = fromctodisk(c11, c22, c12)[0]
                valuespoints2[k][1] = fromctodisk(c22, c11, c12)[1]
                valuespoints2[k][0] = fromctodisk(c22, c11, c12)[0]
                # valuespointshex[k][1] = fromctodisk(cc11, cc22, cc12)[1]
                # valuespointshex[k][0] = fromctodisk(cc11, cc22, cc12)[0]
                valuespointshex2[k][1] = fromctodisk(cc22, cc11, cc12)[1]
                valuespointshex2[k][0] = fromctodisk(cc22, cc11, cc12)[0]

                k = k+1
sizep = k
a = 10
aa = 2*a*2*a*2*a
dim2=100
values4 = np.zeros((aa, dim2, 2))
values5 = np.zeros((aa, dim2, 2))
valuesrh = np.zeros((aa, dim2, 2))
valuesrh2 = np.zeros((aa, dim2, 2))
k = 0
for it11 in range(-a, a):
    for it12 in range(-a, a):
        for it21 in range(-a, a):
            if it11 == 0:
                m11 = 0
                m22 = -1
                m12 = 1
                m21 = 1
            elif it11 != 0:
                m11 = int(it11)
                m12 = int(it12)
                m21 = int(it21)
##               print(isinstance(m12, int))
                m22t = int((1+int(it12)*int(it21))/int(it11))
                if (m22t*m11-m12*m21) == 1:
                    m22 = m22t
                    m11 = int(it11)
                    m12 = int(it12)
                    m21 = int(it21)
                #print("Determinant is {:}".format(m22*m11-m12*m21))
                else:
                    m11 = 1
                    m22 = 1
                    m12 = 0
                    m21 = 0


##            print('c11={:}, c22={:},c12={:}'.format(c11,c22,c12))

            for jj in range(0, dim2):
                i = jj+1
                #print('jj is {:} and i is {:}'.format(jj,i))
                c11 = m22*m22*i+m21*m21*(1/i)
                c22 = m12*m12*i+m11*m11*(1/i)
                c12 = -m22*m12*i-m21*m11*(1/i)
                cci11 = m22*m22*(1/i)+m21*m21*i
                cci22 = m12*m12*(1/i)+m11*m11*i
                cci12 = -m22*m12*(1/i)-m21*m11*i

                crh11 = m22*m22*i+m21*m21*((1/i)+(i/4))-2*m21*m22*(i/2)
                crh22 = m12*m12*i+m11*m11*((1/i)+(i/4))-2*m12*m11*(i/2)
                crh12 = -m12*m22*i+(m11*m22+m21*m12) * \
                    (i/2)-m21*m11*((1/i)+(i/4))
                crhb11 = m22*m22*((1/i)+(i/4))+m21*m21*i-2*m21*m22*(i/2)
                crhb22 = m12*m12*((1/i)+(i/4))+m11*m11*i-2*m12*m11*(i/2)
                crhb12 = -m12*m22*((1/i)+(i/4)) + \
                    (m11*m22+m21*m12)*(i/2)-m21*m11*i

                values4[k][jj][1] = fromctodisk(c11, c22, c12)[1]
                values4[k][jj][0] = fromctodisk(c11, c22, c12)[0]

                values5[k][jj][1] = fromctodisk(cci11, cci22, cci12)[1]
                values5[k][jj][0] = fromctodisk(cci11, cci22, cci12)[0]

                valuesrh[k][jj][1] = fromctodisk(crh11, crh22, crh12)[1]
                valuesrh[k][jj][0] = fromctodisk(crh11, crh22, crh12)[0]
                
                valuesrh2[k][jj][1] = fromctodisk(crhb11, crhb22, crhb12)[1]
                valuesrh2[k][jj][0] = fromctodisk(crhb11, crhb22, crhb12)[0]
            k = k+1
            # print(valuesrh[2][99][1])
            
#Shear paths
# path0 = np.zeros((100, 2))
# for i in range(0, path0.shape[0]):
#     path0[i][1] = pathgen(-0.1*path0.shape[0]+0.2*i, 0)[1]
#     path0[i][0] = pathgen(-0.1*path0.shape[0]+0.2*i, 0)[0]
# path90 = np.zeros((100, 2))
# for i in range(0, path90.shape[0]):
#     path90[i][1] = pathgen(-0.1*path90.shape[0]+0.2*i, 90)[1]
#     path90[i][0] = pathgen(-0.1*path90.shape[0]+0.2*i, 90)[0]
# pathrec=np.zeros((50,2))
# for i in range(0,pathrec.shape[0]):
#     detcsq=1.
#     pathrec[i][1]=pathgenrect(-0.1*pathrec.shape[0]+0.2*i,detcsq)[1]
#     pathrec[i][0]=pathgenrect(-0.1*pathrec.shape[0]+0.2*i,detcsq)[0]
            
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
size4 = k
#Plot rectangular and rhombic lines
for j in range(0, size4):
    #ax.plot(values4[j, :, 1], values4[j, :, 0], "--", lw=.7, c='Gray')
    #ax.plot(values5[j, :, 1], values5[j, :, 0], "--", lw=.7, c='Gray')
    ax.plot(valuesrh[j, :, 1], valuesrh[j, :, 0], lw=0.7, c='Gray')
    # ax.plot(valuesrh2[j,:,1], valuesrh2[j,:,0],lw=1.2,c='Gray')
##                print('it11={:}, it12={:},it21={:}'.format(it11,it12,it21))
##                print("a straight line in k={:}??".format(k-1))
##                print('m11={:}, m12={:},m21={:},m22={:}'.format(m11,m12,m21,m22))
# "



####################################################################################################
#Plot Square and Triangular Wells
ax.plot(valuespoints2[:,1], valuespoints2[:,0],'s',linestyle='None',markersize=5,mfc='Black')
ax.plot(valuespointshex2[:,1], valuespointshex2[:,0],'^',linestyle='None',markersize=5,mfc='Red')


#####################################################################################################

#ax.plot(azimuthss, zenithss,lw=1.5,c="Black")
# ax.plot( path0[:,1],path0[:,0],lw=2.5,c="gray")
# ax.plot( path90[:,1],path90[:,0],lw=2.5,c="gray")
# ax.plot( pathrec[:,1],pathrec[:,0],lw=2.5,c="gray")

ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
ax.set_rmax(1)

##ax.plot(values2[:,1], values2[:,0],"g--",lw=2)
##ax.plot(values3[:,1], values3[:,0],"g--",lw=2)
plt.savefig('RobertaCode/poincdisk_fd.png', dpi=500)

#plt.show()