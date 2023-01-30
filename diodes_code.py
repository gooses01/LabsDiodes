#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import pandas as pd

e = 1.6e-19
T_room = 273.15+20.1
kB = 1.38e-23
def eq1(V,I0):
    return I0*(np.exp(-e*V/(kB*T_room))-1)

def eq4(V,I0,nid):
    return I0*np.exp(e*V/(nid*kB*T_room))



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
        if V[i][j]>0.36:
            V_p_.append(V[i][j])
            I_p_.append(I[i][j])
            lnI_p_.append(np.log(I[i][j]))
    V_p.append(V_p_)
    I_p.append(I_p_)
    lnI_p.append(lnI_p_)


colors = ['k','g','r','tab:orange','b']
labels = ['GaAs','Green','Red','Orange','Blue']

for i in range(len(V)):
    plt.plot(V_p[i],lnI_p[i],"o",color=colors[i],label=labels[i])
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
for i in range(len(V)):
    fit,cov = np.polyfit(V[i][lims[i][0]:lims[i][1]],lnI[i][lims[i][0]:lims[i][1]],1,cov=1)
    fits0.append(fit[0])
    Vs = np.linspace(V[i][lims[i][0]],V[i][lims[i][1]],100)
    plt.plot(V[i],lnI[i],'o')
    plt.plot(Vs,fit[0]*Vs + fit[1])
    plt.title("LED I-V curves: %s" %labels[i])
    plt.xlabel("Voltage (V)")
    plt.ylabel("lnCurrent (lnI)")
    plt.show()

nid = e/(np.array(fits0)*kB*T_room)
print(nid)



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
    plt.plot(-V[i],I[i],"o",color=colors[i],label=labels[i])
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





"""


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

# %%
