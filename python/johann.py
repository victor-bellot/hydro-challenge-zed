import numpy as np
from numpy import arctan2,sign,arctan,cos,sin,pi
from numpy.linalg import norm,det
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse



def clear(ax):
    plt.pause(0.001)
    plt.cla()
    ax.set_xlim(ax.xmin,ax.xmax)
    ax.set_ylim(ax.ymin,ax.ymax)

def init_figure(xmin,xmax,ymin,ymax): 
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')   
    ax.xmin=xmin
    ax.xmax=xmax
    ax.ymin=ymin
    ax.ymax=ymax
    clear(ax)
    return ax

def angle(x):
    x=x.flatten()
    return np.arctan2(x[1],x[0])

def add1(M):
    M=np.array(M)
    return np.vstack((M,np.ones(M.shape[1])))

def tran2H(x,y):
    return np.array([[1,0,x],[0,1,y],[0,0,1]])

def rot2H(a):
    return np.array([[cos(a),-sin(a),0],[sin(a),cos(a),0],[0,0,1]])

def draw_arrow(x,y,θ,L,col='darkblue',w=1):
    plot2D(tran2H(x,y)@rot2H(θ)@arrow2H(L),col,w)

def plot2D(M,col='black',w=1):
    plt.plot(M[0, :], M[1, :], col, linewidth = w) 


def arrow2H(L):
    e=0.2
    return add1(L*np.array([[0,1,1-e,1,1-e],[0,0,-e,0,e]]))

def draw_disk(ax,c,r,col,alph=0.7,w=1):
    #draw_disk(ax,array([[1],[2]]),0.5,"blue")
    e = Ellipse(xy=c, width=2*r, height=2*r, angle=0,linewidth = w)   
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(alph)  # transparency
    e.set_facecolor(col)


    

def draw_saildrone(x,u,ψ,awind,ψ_ap,a_ap,fr,fs,ff):
    mx,my,θ,v,w,δs=list(x[0:6,0])
    u1,u2=list(u[0:2,0])
    hull=add1(np.array([[-1,5,7,7,5,-1,-1,-1],[-2,-2,-1,1,2,2,-2,-2]]))
    sail=np.array([[-5,2.5],[0,0],[1,1]])
    rudder=np.array([[-1,1],[0,0],[1,1]])
    R=tran2H(mx,my)@rot2H(θ)
    Rs=tran2H(3,0)@rot2H(δs)
    Rf=tran2H(-6,0)@rot2H(-u2)
    Rr=tran2H(-1,0)@rot2H(u1)
    draw_arrow(17,17,ψ,5*awind,'red')
    draw_arrow(17,17,ψ_ap+θ,5*a_ap,'green')
    plot2D(R@Rs@Rf@rot2H(-pi/2)@arrow2H(ff),'blue')
    plot2D(R@Rs@rot2H(-pi/2)@arrow2H(0.001*fs),'blue')
    plot2D(R@Rr@rot2H(pi/2)@arrow2H(0.001*fr),'blue')
    draw_disk(ax,R@np.array([[3],[0],[1]]),0.5,"red")
    plot2D(R@hull,'black');
    plot2D(R@Rs@sail,'red',2);
    plot2D(R@Rr@rudder,'red',2);
    plot2D(R@Rs@Rf@rudder,'green',2);

    
def f(x,u):
    x,u=x.flatten(),u.flatten()
    θ=x[2]; v=x[3]; w=x[4]; δr=u[0]; u2=u[1]; δs=x[5];
    w_ap = np.array([[awind*cos(ψ-θ) - v*cos(ψ-θ)],[awind*sin(ψ-θ)]])
    ψ_ap = angle(w_ap)
    a_ap=norm(w_ap)
    fr = p4*v*sin(δr)
    fs = p3*(a_ap)* sin(δs-ψ_ap)
    dx=v*cos(θ) + p0*awind*cos(ψ)
    dy=v*sin(θ) + p0*awind*sin(ψ)
    dv=(fs*sin(δs)-fr*sin(δr)-p1*v**2)/p8
    dw=(fs*(p5-p6*cos(δs)) - p7*fr*cos(δr) - p2*w*v)/p9
    ff=(a_ap)* sin(δs-u2-ψ_ap)
    dδs = p10*ff*cos(u2)+   fs/p3 #=> voile
    xdot=np.array([[dx],[dy],[w],[dv],[dw],[dδs]])
    return xdot,ψ_ap,a_ap,fr,fs,ff


def regulateur (x,q,a,b):
    psi=2
    zeta=pi/4
    theta=x[2]
    m=np.array([x[0],x[1]]).reshape(2,1)
    f=np.hstack(((b-a)/norm(b-a),m-a))
    e=det(f)   
    phi=arctan2(b[1,0]-a[1,0],b[0,0]-a[0,0])
    if abs(e)>r/2 :
        q=sign(e)
    #theta_etoile=phi - (2*0.35/3.14)*arctan(e/r)
    theta_etoile=phi - arctan(e/r)
    if (((cos(psi-theta_etoile)+ cos(zeta)) < 0) or (abs(e)-r < 0 and (cos(psi-phi) + cos(zeta)) < 0)):
        thetabar=pi+psi-zeta*q
    else :
        thetabar=theta_etoile    
    if cos (thetabar-theta)>0:
        sigma_r=sigma_r_max*sin(theta-thetabar)
    else :
        sigma_r=sigma_r_max*sign(sin(theta-thetabar))
    #sigma_max=1
    sigma_s_max = pi/2*((cos(psi-thetabar)+1)/2)**q
    
    u =np.array([sigma_r,[sigma_s_max],[q]]).reshape(3,1)
   # print(u)
    return u,m

"""δr=0.3*sawtooth(θ-θ_bar)
δsbar=-sign(sin(ψ-θ))*(pi/4)*(cos(ψ-θ)+1)
dδs=sawtooth(δs-δsbar)"""

def calcule_angle_incidence(m):
    d_ap = ψ_ap - m
   
    d_true = ψ- m
    produit_1=d_ap *d_true
    n1=norm(d_ap)
    n2=norm(d_ap)
    print(n1)
    gamma = arccos((d_ap @ d_true) / (norm(d_ap) * norm(d_true)))

    return gamma


ax=init_figure(-150,150,-80,80)
p0=0.1 #dérive
p1=50 #frottement du bateau
p2=6000 #frottement à la rotation
p3=1000 # coef voile
p4=2000 # coef gouvernail
p5=0.01 #distance centre de poussée de la voile au mat
p6=1 #distance quille mât
p7=2 # distance gouvernail-quille
p8=300 #masse
p9=10000 # moment d'inertie
p10 = 1 #coef flag
#x = array([[0,1,-pi/2,0,0,1]]).T   #x=(x,y,θ,v,w,δs)
x = np.array([100,-70,-3,1,0,1]).reshape(6,1)   #x=(x,y,θ,v,w,δs)
ψ_ap=5
dt = 0.1

q=1
r=10 # taille couloir 
sigma_r_max =5 # angle max du gouvernail
awind,ψ = 3*pi/2,2 # vitesse du vent, angle du vent apparent
#---------------------------

a = np.array([[75],[-75]])   
b = np.array([[-75],[-75]])
c=np.array([-75,20]).reshape(2,1)
d=np.array([-150,20]).reshape(2,1)
coord=[a,b,c,d]


i=0
def control (i,x):
    while True:
        clear(ax)
        plt.plot([coord[i][0][0],coord[i+1][0][0]],[coord[i][1][0],coord[i+1][1][0]],'red')
        
        u,m=regulateur(x,q,coord[i],coord[i+1])
        vector_director = coord[i+1] - coord[i]
        vector_point_B= coord[i+1]
        if vector_director[0] == 0:
            # Segment vertical
            vector_orthogonal = np.array([1, 0])
        elif vector_director[1] == 0:
            # Segment horizontal
            vector_orthogonal = np.array([0, 1])
        else:
            # Segment diagonal
            coord_y_norm = -(vector_director[0] ** 2) / vector_director[1]
            vector_orthogonal = np.array([vector_director[0], coord_y_norm])
        plt.quiver(vector_point_B[0], vector_point_B[1], vector_orthogonal[0], vector_orthogonal[1], color='red')
        # droite

        x2 = coord[i+1][0][0] + vector_orthogonal[0]
        y2 = coord[i+1][1][0] + vector_orthogonal[1]
        c=np.array([x2,y2]).reshape(2,1)

        # Tracer la droite
       
        #plot([coord[i+1][0][0],x2],[coord[i+1][0][0],y2],'blue')
     
        vecteur_ab = coord[i+1] - c

        # Vecteur entre le point a et votre position
        vecteur_a_vous = m - c

        # Calcul du produit vectoriel entre le vecteur ab et le vecteur a_vous
        produit_vectoriel = np.cross(vecteur_ab.T, vecteur_a_vous.T)
        
        if produit_vectoriel <30:

           print ("stop")
           i=i+1
         
           # a,b,i = simu_follow_line(W,X,lx = 0.2,ly= 0.3)
           # print("i",i)
        
      
        xdot,ψ_ap,w_ap,fr,fs,ff=f(x,u)
        x = x + dt*xdot
        #if x[0,0]**2+x[1,0]**2>150: x[0,0]=0; x[1,0]=0
        draw_saildrone(x,u,ψ,awind,ψ_ap,w_ap,fr,fs,ff)
        if i==3:
            break
control(i,x)


