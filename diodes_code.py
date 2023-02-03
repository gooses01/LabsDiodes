#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd
import glob
from decimal import Decimal

kB = 1.3806452e-23
e = 1.6e-19
h =  6.625e-34
c = 3e8

def normalize_y(x, y):
    area = np.trapz(y, x)
    y_normalized = y / area
    return y_normalized

plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors= colors+['tab:purple','tab:darkorange']
print(colors)

e = 1.6e-19
T_room = 273.15+20.1
kB = 1.38e-23
def eq1(V,I0):
    return I0*(np.exp(-e*V/(kB*T_room))-1)

def eq4(V,I0,nid):
    return I0*np.exp(e*V/(nid*kB*T_room))

def gauss(x,a,mu,sig):
    return a*np.exp(-0.5*((x-mu)/sig)**2)

def gauss2(x,a,b,mu,mu2,sig,sig2):
    return a*np.exp(-0.5*((x-mu)/sig)**2) + b*np.exp(-0.5*((x-mu2)/sig2)**2)

def gauss3(x,a,b,c,mu,mu2,mu3,sig,sig2,sig3):
    return a*np.exp(-0.5*((x-mu)/sig)**2) + b*np.exp(-0.5*((x-mu2)/sig2)**2) + c*np.exp(-0.5*((x-mu3)/sig3)**2)



#%% Forward Bias Diodes
data_zener2V7=pd.read_csv("all_data//tk3_zener2V7_DIODE.csv",skiprows=7)
data_zener9V1=pd.read_csv("all_data//fwd_ZENER_9V1_tk2.csv",skiprows=7)
data_silicon=pd.read_csv("all_data//tk3_silicon_DIODE.csv",skiprows=7)
data_ger=pd.read_csv("all_data//tk4_Ger_DIODE.csv",skiprows=7)
data_gaIn = pd.read_csv("all_data//fwd_GAIN_293K4.csv",skiprows=7)

V = []
lnI = []
V.append(data_zener2V7["Value"].to_numpy())
lnI.append(np.log(data_zener2V7["Reading"].to_numpy()))

V.append(data_zener9V1["Value"].to_numpy())
lnI.append(np.log(data_zener9V1["Reading"].to_numpy()))

V.append(data_silicon["Value"].to_numpy())
lnI.append(np.log(data_silicon["Reading"].to_numpy()))

V.append(data_ger["Value"].to_numpy())
lnI.append(np.log(data_ger["Reading"].to_numpy()))

V.append(data_gaIn["Value"].to_numpy())
lnI.append(np.log(data_gaIn["Reading"].to_numpy()))

labels = ["Zener 2V7","Zener 9V1","Silicon","Germanium","GaIn"]

init = [0.3,1]
fit = []
cov = []
#for i in range(len(V)):
#    fit1,cov1 = spo.curve_fit(eq4,V[i],I[i],p0=init)
#    fit.append(fit1)
#    cov.append(cov1)

#print(fit[2])

#N = len(V)
#plt.plot(V[:N],I[:N],'+')

for i in range(len(V)):
    plt.plot(V[i],lnI[i],"o",label=labels[i])
#Vs =np.linspace(0,1,100)
#plt.plot(V[2],np.log(I[2]),"o",color="g",label=labels[2])
#plt.plot(Vs,eq4(Vs,8.42869829e-16,1.2))
plt.title("Forward Voltage Bias")
plt.xlabel("Voltage (V)")
plt.ylabel("lnCurrent (I)")
#plt.xlim(0,1)
#plt.ylim(-0.1,0.3)
plt.grid()
plt.legend()
plt.show()

fit1 = np.polyfit(V[0][:75],lnI[0][:75],1)
fit2 = np.polyfit(V[0][100:130],lnI[0][100:130],1)
Vs1 =np.linspace(0,0.6,100)
Vs2 = np.linspace(0.65,0.9,100)
plt.plot(V[0],lnI[0],"o")
plt.plot(Vs1,fit1[0]*Vs1+fit1[1])
plt.plot(Vs2,fit2[0]*Vs2+fit2[1])
plt.title("Forward Voltage Bias: %s" %labels[0])
plt.xlabel("Voltage (V)")
plt.ylabel("lnCurrent (lnI)")
nid1 = e/(fit1[0]*kB*T_room)
nid2 = e/(fit2[0]*kB*T_room)
print("%s" %labels[0],nid1,nid2)
#plt.plot(V[0][100:130],lnI[0][100:130],"o")
plt.plot()
plt.show()


fit1,cov1 = np.polyfit(V[1][5:20],lnI[1][5:20],1,cov=1)
Vs1 =np.linspace(0.3,0.9,100)
plt.plot(V[1],lnI[1],"o")
#plt.plot(V[1][5:20],lnI[1][5:20],"o")
plt.plot(Vs1,fit1[0]*Vs1+fit1[1])
plt.title("Forward Voltage Bias: %s" %labels[1])
plt.xlabel("Voltage (V)")
plt.ylabel("lnCurrent (lnI)")
print(fit1[0],np.sqrt(cov1[0][0]))
nid1 = e/(fit1[0]*kB*T_room)
#nid2 = e/(fit2[0]*kB*T_room)
print("%s" %labels[1],nid1)
#plt.plot(V[0][100:130],lnI[0][100:130],"o")
plt.plot()
plt.show()

fit1 = np.polyfit(V[2][:20],lnI[2][:20],1)
Vs1 =np.linspace(0.5,0.8,100)
plt.plot(V[2],lnI[2],"o")
plt.plot(Vs1,fit1[0]*Vs1+fit1[1])
nid1 = e/(fit1[0]*kB*T_room)
print("%s" %labels[2],nid1)
plt.title("Forward Voltage Bias: %s" %labels[2])
plt.xlabel("Voltage (V)")
plt.ylabel("lnCurrent (lnI)")
plt.show()

fit1 = np.polyfit(V[3][:15],lnI[3][:15],1)
Vs1 =np.linspace(0.,0.2,100)
fit2 = np.polyfit(V[3][50:],lnI[3][50:],1)
Vs2 =np.linspace(0.5,3,100)
plt.plot(V[3],lnI[3],"o")
plt.plot(Vs1,fit1[0]*Vs1+fit1[1])
plt.plot(Vs2,fit2[0]*Vs2+fit2[1])
nid1 = e/(fit1[0]*kB*T_room)
nid2 = e/(fit2[0]*kB*T_room)
print("%s" %labels[3],nid1)
print("%s" %labels[3],nid2)
plt.title("Forward Voltage Bias: %s" %labels[3])
plt.xlabel("Voltage (V)")
plt.ylabel("lnCurrent (lnI)")
plt.show()

plt.plot(V[4],lnI[4],"o")
fit1 = np.polyfit(V[4][1:15],lnI[4][1:15],1)
Vs1 =np.linspace(0.,0.8,100)
plt.plot(Vs1,fit1[0]*Vs1+fit1[1])
plt.title("Forward Voltage Bias: %s" %labels[4])
plt.xlabel("Voltage (V)")
plt.ylabel("lnCurrent (lnI)")
nid1 = e/(fit1[0]*kB*T_room)
print("%s" %labels[4],nid1)
plt.show()

#%% Forward LEDs
data_Gaas=pd.read_csv("all_data//gaAs_LED.csv",skiprows=7).sort_values("Value")
data_green=pd.read_csv("all_data//green_LED.csv",skiprows=7).sort_values("Value")
data_red=pd.read_csv("all_data//red_LED.csv",skiprows=7).sort_values("Value")
data_orange=pd.read_csv("all_data//orangeLED_20C.csv",skiprows=7).sort_values("Value")
data_blue=pd.read_csv("all_data//blue_LED.csv",skiprows=7).sort_values("Value")

V=[]
I=[]
lnI=[]
V.append(data_Gaas["Value"].to_numpy())
I.append(data_Gaas["Reading"].to_numpy())
lnI.append(np.log(data_Gaas["Reading"].to_numpy()))

V.append(data_green["Value"].to_numpy())
I.append(data_green["Reading"].to_numpy())
lnI.append(np.log(data_green["Reading"].to_numpy()))

V.append(data_red["Value"].to_numpy())
I.append(data_red["Reading"].to_numpy())
lnI.append(np.log(data_red["Reading"].to_numpy()))


V.append(data_orange["Value"].to_numpy())
I.append(data_orange["Reading"].to_numpy())
lnI.append(np.log(data_orange["Reading"].to_numpy()))


V.append(data_blue["Value"].to_numpy())
I.append(data_blue["Reading"].to_numpy())
lnI.append(np.log(data_blue["Reading"].to_numpy()))

V_p = []
I_p = []
lnI_p = []
for i in range(len(V)):
    V_p_=[]
    I_p_=[]
    lnI_p_=[]
    for j in range(len(V[i])):
        V_p_.append(V[i][j])
        I_p_.append(I[i][j])
        lnI_p_.append(np.log(I[i][j]))
        #if V[i][j]>0.36:
        #    V_p_.append(V[i][j])
        #    I_p_.append(I[i][j])
        #    lnI_p_.append(np.log(I[i][j]))
    V_p.append(V_p_)
    I_p.append(I_p_)
    lnI_p.append(lnI_p_)


colors = ['k','g','r','tab:orange','b']
labels = ['GaAs','Green','Red','Orange','Blue']

for i in range(len(V)):
    plt.plot(V_p[i],I_p[i],"o",color=colors[i],label=labels[i])
plt.title("LED I-V curves")
plt.axvline(0.36)
plt.xlabel("Voltage (V)")
plt.ylabel("Current (I)")
#plt.xlim(0,1)
#plt.ylim(-0.1,0.3)
plt.grid()
plt.legend()
plt.show()

lims = [(35,100),(95,130),(9,15),(28,44),(18,24)]
#plt.plot(V[5],lnI[5])
#plt.plot(V[5][18:24],lnI[5][18:24],"o")
#plt.show()

fits0 = []
fits1 = []
for i in range(len(V)):
    fit,cov = np.polyfit(V[i][lims[i][0]:lims[i][1]],lnI[i][lims[i][0]:lims[i][1]],1,cov=1)
    fits0.append(fit[0])
    fits1.append(fit[1])
    Vs = np.linspace(V[i][lims[i][0]],V[i][lims[i][1]],100)
    plt.plot(V[i],lnI[i],'o')
    plt.plot(Vs,fit[0]*Vs + fit[1])
    plt.title("LED I-V curves: %s" %labels[i])
    plt.xlabel("Voltage (V)")
    plt.ylabel("lnCurrent (lnI)")
    plt.show()

nid = e/(np.array(fits0)*kB*T_room)
V0 = -(17+np.array(fits1))/np.array(fits0)
print(labels)
print(nid)
print(V0)

#%% Reverse bias diodes 300K
zener9V1_78K = pd.read_csv("all_data//zener_9V1_78K1.csv",skiprows=7).sort_values("Value")
zener2V7_78K = pd.read_csv("all_data//zener2V7_78K3.csv",skiprows=7).sort_values("Value")
data_zener2V7=pd.read_csv("all_data//tk3_zener2V7_DIODE.csv",skiprows=7)
data_zener9V1=pd.read_csv("all_data//9V1_20mA_0V5.csv",skiprows=7)

#zener2V7_78K_2 = pd.read_csv("all_data//fwd_zener_2V7_78K1.csv",skiprows=7).sort_values("Value")
#zener9V1_78K_2 = pd.read_csv("all_data//fwd_zener_9V1_78K.csv",skiprows=7).sort_values("Value")

V=[]
I=[]
lnI=[]
V.append(zener9V1_78K["Value"].to_numpy())
I.append(zener9V1_78K["Reading"].to_numpy())
lnI.append(np.log(zener9V1_78K["Reading"].to_numpy()))

V.append(zener2V7_78K["Value"].to_numpy())
I.append(zener2V7_78K["Reading"].to_numpy())
lnI.append(np.log(zener2V7_78K["Reading"].to_numpy()))

V.append(data_zener2V7["Value"].to_numpy())
I.append(data_zener2V7["Reading"].to_numpy())
lnI.append(np.log(data_zener2V7["Reading"].to_numpy()))

V.append(data_zener9V1["Value"].to_numpy())
I.append(data_zener9V1["Reading"].to_numpy())
lnI.append(np.log(data_zener9V1["Reading"].to_numpy()))

#V.append(data_zener9V1_2["Value"].to_numpy())
#I.append(data_zener9V1_2["Reading"].to_numpy())
#lnI.append(np.log(data_zener9V1_2["Reading"].to_numpy()))

#V.append(zener9V1_78K_2["Value"].to_numpy())
#I.append(zener9V1_78K_2["Reading"].to_numpy())
#lnI.append(np.log(zener9V1_78K_2["Reading"].to_numpy()))

#V.append(zener2V7_78K_2["Value"].to_numpy())
#I.append(zener2V7_78K_2["Reading"].to_numpy())
#lnI.append(np.log(zener2V7_78K_2["Reading"].to_numpy()))

colors = ['r','g','b','k','tab:orange']
labels = ['9V1 78.1K','2V7 78.3K','2V7 300K','9V1 300K','9V1 300K 2']

for i in range(len(V)):
    plt.plot(V[i],I[i],"o",color=colors[i],label=labels[i])
#i = 2
#plt.plot(V[i],I[i],"o",color=colors[i],label=labels[i])
plt.title("Zener sweeps liquid nitrogen")
plt.xlabel("Voltage (V)")
plt.ylabel("Current (I)")
plt.grid()
plt.legend()
plt.show()

plt.plot(V[3],I[3],'o',color='k',label=labels[i])
plt.show()

#%% FULL ZENER room temp
zener9V1_78K = pd.read_csv("all_data//full_9V1.csv",skiprows=7).sort_values("Value")
zener2V7_78K = pd.read_csv("all_data//full_2V7.csv",skiprows=7).sort_values("Value")
GaIn=pd.read_csv("all_data//full_GaIn.csv",skiprows=7)
sil=pd.read_csv("all_data//full_SIL.csv",skiprows=7)

#zener2V7_78K_2 = pd.read_csv("all_data//fwd_zener_2V7_78K1.csv",skiprows=7).sort_values("Value")
#zener9V1_78K_2 = pd.read_csv("all_data//fwd_zener_9V1_78K.csv",skiprows=7).sort_values("Value")

V=[]
I=[]
lnI=[]
V.append(zener9V1_78K["Value"].to_numpy())
I.append(zener9V1_78K["Reading"].to_numpy())
lnI.append(np.log(zener9V1_78K["Reading"].to_numpy()))

V.append(zener2V7_78K["Value"].to_numpy())
I.append(zener2V7_78K["Reading"].to_numpy())
lnI.append(np.log(zener2V7_78K["Reading"].to_numpy()))

V.append(GaIn["Value"].to_numpy())
I.append(GaIn["Reading"].to_numpy())
lnI.append(np.log(GaIn["Reading"].to_numpy()))

V.append(sil["Value"].to_numpy())
I.append(sil["Reading"].to_numpy())
lnI.append(np.log(sil["Reading"].to_numpy()))

#V.append(data_zener9V1_2["Value"].to_numpy())
#I.append(data_zener9V1_2["Reading"].to_numpy())
#lnI.append(np.log(data_zener9V1_2["Reading"].to_numpy()))

#V.append(zener9V1_78K_2["Value"].to_numpy())
#I.append(zener9V1_78K_2["Reading"].to_numpy())
#lnI.append(np.log(zener9V1_78K_2["Reading"].to_numpy()))

#V.append(zener2V7_78K_2["Value"].to_numpy())
#I.append(zener2V7_78K_2["Reading"].to_numpy())
#lnI.append(np.log(zener2V7_78K_2["Reading"].to_numpy()))

colors = ['r','g','b','k','tab:orange']
labels = ['Zener 9V1','Zener 2V7','GaIn','Silicon']

for i in range(len(V)):
    plt.plot(-V[i],lnI[i],"o",color=colors[i],label=labels[i])
#i = 2
#plt.plot(V[i],I[i],"o",color=colors[i],label=labels[i])
plt.title("Diode Sweeps full")
plt.xlabel("Voltage (V)")
plt.ylabel("Current (I)")
plt.grid()
plt.legend()
plt.xlim(-2,3)
plt.show()

#plt.plot(V[3],I[3],'o',color='k',label=labels[i])
#plt.show()


#%% LEDs at temps 0-70C

temp = np.array([0,10,20,30,40,50,60,70])
blue = ['all_data//bfwd0CX.csv','all_data//bfwd10C.csv','all_data//bfwd20C.csv','all_data//bfwd30C.csv','all_data//bfwd40C.csv','all_data//bfwd50C.csv','all_data//bfwd60C.csv','all_data//bfwd70C.csv']
green = ['all_data//gfwd0CX.csv','all_data//gfwd10C.csv','all_data//gfwd20C.csv','all_data//gfwd30C.csv','all_data//gfwd40C.csv','all_data//gfwd50C.csv','all_data//gfwd60C.csv','all_data//gfwd70C.csv']
red = ['all_data//rfwd0CX.csv','all_data//rfwd10C.csv','all_data//rfwd20C.csv','all_data//rfwd30C.csv','all_data//rfwd40C.csv','all_data//rfwd50C.csv','all_data//rfwd60C.csv','all_data//rfwd70C.csv']

V_blue = []
V_green = []
V_red = []
I_blue = []
I_green = []
I_red = []
lnI_blue = []
lnI_green = []
lnI_red = []
for i in range(len(temp)):
    blue_data = pd.read_csv(blue[i],skiprows=7).sort_values('Value')
    V_blue.append(blue_data["Value"].to_numpy())
    I_blue.append(blue_data["Reading"].to_numpy())
    lnI_blue.append(np.log(blue_data["Reading"].to_numpy()))
    green_data = pd.read_csv(green[i],skiprows=7).sort_values('Value')
    V_green.append(green_data["Value"].to_numpy())
    I_green.append(green_data["Reading"].to_numpy())
    lnI_green.append(np.log(green_data["Reading"].to_numpy()))
    red_data = pd.read_csv(red[i],skiprows=7).sort_values('Value')
    V_red.append(red_data["Value"].to_numpy())
    I_red.append(red_data["Reading"].to_numpy())
    lnI_red.append(np.log(red_data["Reading"].to_numpy()))

for i in range(len(temp)):
    plt.plot(V_blue[i],lnI_blue[i],'o',color = 'b',label='Blue')
    plt.plot(V_green[i],lnI_green[i],'o',color = 'g',label='Green')
    plt.plot(V_red[i],lnI_red[i],'o',color = 'r',label='Red')
    plt.xlabel('Voltage (V)')
    plt.ylabel('lnCurrent (lnI)')
    plt.title('LEDs at %s C' %temp[i])
    plt.xlim(1,4)
    plt.grid()
    plt.show()


#%% LEDs at temps 0-70C 2

temp = np.array([0,10,20,30])
files = []
files.append(['all_data//blue_LED//blue_LED_%sC.csv' %t for t in temp])
files.append(['all_data//gaas_LED//GaAs_LED_%sC.csv' %t for t in temp])
files.append(['all_data//green_LED//green_LED_%sC.csv' %t for t in temp])
files.append(['all_data//orange_LED//orange_LED_%sC.csv' %t for t in temp])
files.append(['all_data//red_LED//red_LED_%sC.csv' %t for t in temp])
#blue = ['all_data//bfwd0CX.csv','all_data//bfwd10C.csv','all_data//bfwd20C.csv','all_data//bfwd30C.csv','all_data//bfwd40C.csv','all_data//bfwd50C.csv','all_data//bfwd60C.csv','all_data//bfwd70C.csv']
#green = ['all_data//gfwd0CX.csv','all_data//gfwd10C.csv','all_data//gfwd20C.csv','all_data//gfwd30C.csv','all_data//gfwd40C.csv','all_data//gfwd50C.csv','all_data//gfwd60C.csv','all_data//gfwd70C.csv']
#red = ['all_data//rfwd0CX.csv','all_data//rfwd10C.csv','all_data//rfwd20C.csv','all_data//rfwd30C.csv','all_data//rfwd40C.csv','all_data//rfwd50C.csv','all_data//rfwd60C.csv','all_data//rfwd70C.csv']



V_blue = []
V_green = []
V_red = []
I_blue = []
I_green = []
I_red = []
lnI_blue = []
lnI_green = []
lnI_red = []

V = []
I = []
for file in files:
    V_col = []
    I_col = []
    for i in range(len(temp)):
        data = pd.read_csv(file[i],skiprows=7).sort_values('Value')
        V_col.append(data['Value'].to_numpy())
        I_col.append(data['Reading'].to_numpy())
    V.append(V_col)
    I.append(I_col)

labels = ['Blue','GaAs','Green','Orange','Red']
colors = ['b','k','g','tab:orange','r']
for i in range(len(temp)):
    for j in range(len(files)):
        plt.plot(V[j][i],I[j][i],'o',color=colors[j],label=labels[j])
    plt.xlabel('Voltage (V)')
    plt.ylabel('lnCurrent (lnI)')
    plt.title('LEDs at %s C' %temp[i])
    plt.xlim(1,4)
    plt.grid()
    plt.show()


"""
for i in range(len(temp)):
    plt.plot(V_blue[i],lnI_blue[i],'o',color = 'b',label='Blue')
    plt.plot(V_green[i],lnI_green[i],'o',color = 'g',label='Green')
    plt.plot(V_red[i],lnI_red[i],'o',color = 'r',label='Red')
    plt.xlabel('Voltage (V)')
    plt.ylabel('lnCurrent (lnI)')
    plt.title('LEDs at %s C' %temp[i])
    plt.xlim(1,4)
    plt.grid()
    plt.show()




def eq4(V,I0):
    return I0*np.exp(e*V/(2*kB*T))


def eq(V,I0,a):
    return I0*(np.exp(a*V))

init = [0.3,3]
#fit,cov = spo.curve_fit(eq1,V[:N],I[:N],p0=init)
#fit,cov = spo.curve_fit(eq4,V[:N],I[:N],p0=init)
fit,cov = spo.curve_fit(eq,V[:N],I[:N],p0=init)

print(fit)
#print(kB*T/e)

plt.plot(V[:N],eq(V[:N],fit[0],fit[1]))

plt.show()

print('nid = ',1/(fit[1]*kB*T/e))
"""

#%% Spectrometer measurements room temp
labels = ['GaAs LED','Green LED','Orange LED','Red LED']
colors = ['k','g','tab:orange','r']
files = ['Lab_Y3//spectrometer//GaAs_1V5_187mA_294K7.csv','Lab_Y3\spectrometer\green\green_3V_55mA_295K2.csv','Lab_Y3\spectrometer\orange_LED\orange_2V5_190mA_295K.csv','Lab_Y3//spectrometer//redLED.csv']
delimiters = [';',';',';',',',',']

I = []
wvl = []
for i in range(len(files)):
    data = np.loadtxt(files[i],skiprows=48,unpack=1,delimiter=delimiters[i])
    I.append(data[0])
    wvl.append(data[1])

mu = []
sig = []
a = []
initial_guess = [1,650,20]
xs = np.linspace(200,1000,1000)
for i in range(len(labels)):
    fit,cov = spo.curve_fit(gauss,I[i],wvl[i],p0=initial_guess)
    #plt.plot(I[i],wvl[i],'o',color=colors[i],label=labels[i])
    plt.plot(xs,gauss(xs,fit[0],fit[1],fit[2]),color=colors[i])
    a.append(fit[0])
    mu.append(fit[1])
    sig.append(fit[2])
    plt.plot()
    plt.xlabel(r'$\lambda$ (nm)')
    plt.ylabel('Intensity')
plt.legend()
plt.show()

print(a)
print(mu)
print(sig)

#%% Spectrometer measurements LN temp GREEN

V_vals = [3.45,3.5,3.6,3.7,3.8,3.9,3.0,4.0,4.5,5.0]
del V_vals[6]
file_names = glob.glob("Lab_Y3//spectrometer//green//*")[2:]
del file_names[6]
files = [x for _, x in sorted(zip(V_vals, file_names))]
labels = ['%.2f V' %x for x in V_vals]
labels_sorted = [x for _, x in sorted(zip(V_vals,labels))]

print(files)

i_sorted = []
wvl_sorted = []
for i in range(len(V_vals)):
    data = np.loadtxt(files[i],skiprows=48,unpack=1,delimiter=';')
    i_sorted.append(data[0])
    wvl_sorted.append(data[1])

#order = [3,2,3,3,2,2,2,2,2,2]
#del order[6]
i00 = (3,[9.52172924e-02, 9.71545396e-02, 5.89524743e-02, 5.49547839e+02, 5.59100863e+02, 5.72228489e+02, 3.77443201e+00, 4.79587590e+00, 2.08369806e+01])
i02 = (2,[4.24693727e-01, 2.12462198e-01, 5.63760325e+02, 5.76618283e+02, 9.20507913e+00, 1.76002744e+01])
i03 = (3,[1.85062867e-01, 3.33954818e-01, 1.40194034e-01, 5.52547309e+02, 5.61004533e+02, 5.73269699e+02, 3.15357866e+00, 7.22404871e+00, 2.00695190e+01])
i01 = (3,[0.165, 3.14720331e-01, 1.37015744e-01, 5.55229448e+02, 5.63335903e+02, 5.73407451e+02, 3, 7, 1.94792939e+01])
i04 = (2,[4.27993621e-01,2.52469824e-01, 5.67026880e+02, 5.78389810e+02, 1.00197533e+01, 1.86622754e+01])
i05 = (2,[6.39633203e-01, 3.22481644e-01, 5.68214918e+02, 5.78983611e+02, 1.04706309e+01, 2.02585137e+01])
#i06 = (2,[6.72777387e-01, 3.03361353e-01, 5.75523216e+02, 5.86190990e+02, 1.16117752e+01, 2.18131039e+01])
i07 = (2,[3.51719225e-01, 1.82850004e-01, 5.70644012e+02, 5.81077563e+02, 1.07561786e+01, 2.03793965e+01])
i08 = (1,[ 2.11566416e-01, 5.76555820e+02, 1.59093823e+01])
i09 = (2,[4.83914646e-01, 2.56744918e-01, 5.76459808e+02, 5.85304470e+02,1.17140550e+01, 2.30051066e+01])

i0 = [i00[1],i01[1],i02[1],i03[1],i04[1],i05[1],i07[1],i08[1],i09[1]]
order = [i00[0],i01[0],i02[0],i03[0],i04[0],i05[0],i07[0],i08[0],i09[0]]

order_sorted = [x for _, x in sorted(zip(V_vals,order))]
i0_sorted = [x for _, x in sorted(zip(V_vals,i0))]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors= colors+['k','c']
print(colors)

fig_all2,ax_all2 = plt.subplots()
xs = np.linspace(500,700,1000)
ys = []
for i in range(len(V_vals)):
    #fig,ax = plt.subplots()
    if order_sorted[i] == 1:
        y = gauss(xs,i0_sorted[i][0],i0_sorted[i][1],i0_sorted[i][2])
    elif order_sorted[i] == 2:
        y = gauss2(xs,i0_sorted[i][0],i0_sorted[i][1],i0_sorted[i][2],i0_sorted[i][3],i0_sorted[i][4],i0_sorted[i][5])
    elif order[i] == 3:
        y = gauss3(xs,i0_sorted[i][0],i0_sorted[i][1],i0_sorted[i][2],i0_sorted[i][3],i0_sorted[i][4],i0_sorted[i][5],i0_sorted[i][6],i0_sorted[i][7],i0_sorted[i][8])
    ys.append(y)
    #ax.plot(wvl_sorted[i],i_sorted[i],'o',color=colors[i],label=labels_sorted[i])
    #ax.plot(xs,y)
    #ax.set_xlabel(r'$\lambda$ (nm)')
    #ax.set_ylabel('Intensity')
    #ax.legend()
    #ax.set_xlim(530,600)
    #fig.show()


ys_sorted = [ys[0],ys[3],ys[1],ys[2],ys[4],ys[5],ys[6],ys[7],ys[8]]
for i in range(len(V_vals)):
    ax_all2.plot(xs,normalize_y(xs,ys_sorted[i]),color=colors[i],label=labels_sorted[i])

ax_all2.set_xlabel(r'$\lambda$ (nm)')
ax_all2.set_ylabel('Intensity')
ax_all2.legend()
ax_all2.set_xlim(530,610)
plt.savefig('Plots//green_LN_Vspectra.png')

#%% HERE
files = glob.glob("Lab_Y3//spectrometer//green//*")[2:]
V_vals = [3.45,3.5,3.6,3.7,3.8,3.9,3.0,4.0,4.5,5.0]
labels = ['%.2f V' %x for x in V_vals]

print(files)
print(files[6])

Int = []
wvl = []
for i in range(len(labels)):
    data = np.loadtxt(files[i],skiprows=48,unpack=1,delimiter=';')
    Int.append(data[0])
    wvl.append(data[1])

print(len(Int))
print(len(wvl))

fits = []
initial_guess = [1,1,650,600,20,20]
xs = np.linspace(500,600,1000)

#[5.74e-2,1.63e-1,1,]
l = [553.5,562.8]

fixed = [0.15,0.125,551.5,560,5,2.5]

def gauss2(x,a,b,mu,mu2,sig,sig2):
    return a*np.exp(-0.5*((x-mu)/sig)**2) + b*np.exp(-0.5*((x-mu2)/sig2)**2)

def gauss3(x,a,b,c,mu,mu1,mu2,sig,sig2,sig3):
    return a*np.exp(-0.5*((x-mu)/sig)**2) + b*np.exp(-0.5*((x-mu1)/sig2)**2) + c*np.exp(-0.5*((x-mu2)/sig3)**2)



a = 8
f=0.65
#fit1,cov1 = spo.curve_fit(gauss2,I[a],wvl[a],p0=[0.5,0.5,20,20]) #[0.5,0.5,550,560,10,10]
plt.plot(Int[a],wvl[a],'o',color=colors[a],label=labels[a])
plt.axvline(5.96424988e+02)
plt.axvline(l[0])
plt.axvline(l[1])
#plt.plot(xs,gauss2(xs,fit1[0],fit1[1],fit1[2],fit1[3]),color=colors[a])
#plt.plot(xs,gauss2(xs,0.15,0.125,551.5,560,5,2.5),color=colors[a+1])
#plt.plot(xs,gauss3(xs,f*0.17,f*0.125,f*0.09,551.5,560,571,5,2.5,22),color=colors[a+2])
i01 = [1,570,20]
i02 = [1,1,570,575,20,30]
fit1,cov1 = spo.curve_fit(gauss,I[a],wvl[a],p0=i01)
fit2,cov2 = spo.curve_fit(gauss2,I[a],wvl[a],p0=i02)
plt.plot(xs,gauss(xs,fit1[0],fit1[1],fit1[2]),color=colors[0])
plt.plot(xs,gauss2(xs,fit2[0],fit2[1],fit2[2],fit2[3],fit2[4],fit2[5]),color=colors[1])
#plt.plot(xs,gauss3(xs,i03[0],i03[1],i03[2],i03[3],i03[4],i03[5],i03[6],i03[7],i03[8]),color=colors[0])
plt.xlabel(r'$\lambda$ (nm)')
plt.ylabel('Intensity')
plt.legend()
plt.xlim(530,600)
#print(a,order,order[a])
#print(files[a])
plt.show()
print(fit1)
print(fit2)

#%% ln(I/T^3) vs 1/T single measurements
V,T,I = np.loadtxt('all_data\Smalldt_sweeps\single_data.csv',skiprows=1,usecols=(1,2,3),unpack=1,delimiter=',')
labels = ['GaAs','Green','Orange']
cols = ['k','g','tab:orange']
nid = np.array([1.95838899, 1.88588823, 1.62742584])

T = np.unique(T) + 273.15

V1 = []
I1 = []
V2 = []
I2 = []
V3 = []
I3 = []
for i in range(len(V)):
    if V[i] == 1.5:
        V1.append(V[i])
        I1.append(I[i])
    elif V[i] == 2.5:
        V2.append(V[i])
        I2.append(I[i])
    elif V[i] == 2:
        V3.append(V[i])
        I3.append(I[i])
Vs = np.array([V1[0],V2[0],V3[0]])
Is = [I1,I2,I3]

fig,ax = plt.subplots()
for i in range(len(Is)):
    ax.plot(T,Is[i],'o',color=cols[i],label=labels[i])
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Current (mA)')
    ax.legend()
plt.show()

# ys: ln(I/T^3)
# xs: 1/T
ys = [np.log(np.array(Is[0])/(T**3)), np.log(np.array(Is[1])/(T**3)), np.log(np.array(Is[2])/(T**3))]
xs = 1/(kB*np.array(T))
fig,axes = plt.subplots(1,3)
fits = []
Es = []
for i in range(len(labels)):
    ax=axes[i]
    fit,cov = np.polyfit(xs,ys[i],1,cov=1)
    E = Vs[i]/nid[i] - fit[0]/e
    Es.append(E)
    ax.plot(xs,ys[i],'o',color=cols[i],label = labels[i] + '\n' + r'$E_g$ = %.3f eV' %E + '\n' + r'$\lambda$ = %.1f nm' %(1e9*h*c/(E*e)))
    ax.plot(xs,fit[0]*xs+fit[1])
    fits.append(fit)
    ax.set_xlabel(r'$\frac{1}{k_BT}$')
    ax.set_ylabel(r'ln($\frac{I}{T^3}$)')
    ax.legend()
plt.show()





#%% ln(I/T^3) vs 1/T sweeps
T = np.array([15,16,17,18,19,20,21,22,23,24,25])
T_meas = T[3:]
V0 = [1.5,2.5,2]
cols = ['k','g','tab:orange']
gaas_files = ['all_data//Smalldt_sweeps//%s//gaas_%sC.csv' %(t,t) for t in T]
green_files = ['all_data//Smalldt_sweeps//%s//green_%sC.csv' %(t,t) for t in T]
orange_files = ['all_data//Smalldt_sweeps//%s//orange_%sC.csv' %(t,t) for t in T]
nid = np.array([1.95838899, 1.88588823, 1.62742584])


TvsV_gaas = []
TvsI_gaas = []
TvsV_green = []
TvsI_green = []
TvsV_orange = []
TvsI_orange = []
nid_gaas = []
nid_green = []
nid_orange = []
cut = [70,170,200]
for i in range(len(T)):
    data1 = pd.read_csv(gaas_files[i],skiprows=7)
    a,b = data1["Value"].to_numpy(),data1["Reading"].to_numpy()
    TvsV_gaas.append(a)
    TvsI_gaas.append(b)
    nid_gaas.append(e/(np.polyfit(a[cut[0]:],b[cut[0]:],1,cov=0)[0]*kB*T[i]))

    data2 = pd.read_csv(green_files[i],skiprows=7).sort_values("Value")
    a,b = data1["Value"].to_numpy(),data1["Reading"].to_numpy()
    TvsV_green.append(data2["Value"].to_numpy())
    TvsI_green.append(data2["Reading"].to_numpy())
    #nid_green.append(e/(np.polyfit(data2["Value"].to_numpy()[cut[1]:],data2["Reading"].to_numpy()[cut[1]:],1,cov=0)[0]*kB*T[i]))

    data3 = pd.read_csv(orange_files[i],skiprows=7)
    TvsV_orange.append(data3["Value"].to_numpy())
    TvsI_orange.append(data3["Reading"].to_numpy())
    #nid_orange.append(e/(np.polyfit(data3["Value"].to_numpy()[cut[2]:],data3["Reading"].to_numpy()[cut[2]:],1,cov=0)[0]*kB*T[i]))

#nid = [nid_gaas,nid_green,nid_orange]
#print(nid[0])

"""fig,ax = plt.subplots()
cut = 200 #70,170,200
for i in [3,4,5,6,7,8,9,10]:
    ax.plot(TvsV_orange[i][cut:],TvsI_orange[i][cut:],'o',label='%s' %i)
ax.legend()
plt.show()"""

print('gaas',len(TvsV_gaas))
print('green',len(TvsV_green))
print('orange',len(TvsV_orange))

TvsV_gaas = TvsV_gaas[3:]
TvsI_gaas = TvsI_gaas[3:]
cut_off_gaas = 113
for i in range(len(TvsV_gaas)):
    TvsV_gaas[i] = TvsV_gaas[i][:cut_off_gaas]
    TvsI_gaas[i] = TvsI_gaas[i][:cut_off_gaas]

TvsV_green = TvsV_green[3:]
TvsI_green = TvsI_green[3:]
cut_off_green = 213
for i in range(len(TvsV_green)):
    TvsV_green[i] = TvsV_green[i][:cut_off_green]
    TvsI_green[i] = TvsI_green[i][:cut_off_green]

TvsV_orange = TvsV_orange[3:]
TvsI_orange = TvsI_orange[3:]
cut_off_orange = 212
for i in range(len(TvsV_orange)):
    TvsV_orange[i] = TvsV_orange[i][:cut_off_orange]
    TvsI_orange[i] = TvsI_orange[i][:cut_off_orange]

print('gaas',len(TvsV_gaas))
print('green',len(TvsV_green))
print('orange',len(TvsV_orange))


VvsT_gaas = np.transpose(TvsV_gaas)
IvsT_gaas = np.transpose(TvsI_gaas)
VvsT_green = np.transpose(TvsV_green)
IvsT_green = np.transpose(TvsI_green)
VvsT_orange = np.transpose(TvsV_orange)
IvsT_orange = np.transpose(TvsI_orange)


TvsV = [TvsV_gaas, TvsV_green, TvsV_orange]
VvsT = [VvsT_gaas, VvsT_green, VvsT_orange]
IvsT = [IvsT_gaas, IvsT_green, IvsT_orange]

#print(TvsV[1])

ys = []
xs = []
Es = []
for i in range(len(IvsT)): #select LED color
    y_col = []
    x_col = []
    E_col = []
    for j in range(len(IvsT[i])): #select fixed voltage
        y = np.log(IvsT[i][j]/(T_meas**3))
        x = 1/(kB*T_meas)
        y_col.append(y)
        x_col.append(x)
        EvsT = []
        #for t in range(len(T_meas)-1):
        #    s = (y[t+1] - y[t])/(x[t+1] - x[t])
        #    EvsT.append(VvsT[i][j][0]/nid[i] - s/e)
        fit,cov = np.polyfit(x,y,1,cov=1)
        ## NOTES: CHANGE THIS SO THAT IT TAKES INTO ACCOUNT N_ID CHANGE WITH TEMP (CALC NID FROM EACH SWEEP)
        E_col.append(VvsT[i][j][0]/nid[i] - fit[0]/e)
        #E_col.append(EvsT)
    fig,ax = plt.subplots()
    ax.plot(TvsV[i][0],E_col,'o',color=cols[i]) #1e9*h*c/(np.array(E_col)*e)
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel(r'$E_G$ (eV)')
    #ax.set_ylabel(r'$\lambda$ (nm)')
    ys.append(y_col)
    xs.append(x_col)
    Es.append(E_col)
print('y',len(ys[0]),len(ys[1]),len(ys[1]))
print('x',len(xs[0]),len(xs[1]),len(xs[1]))





# %%
