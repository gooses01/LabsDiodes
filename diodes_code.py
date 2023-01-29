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

fits = []
for i in range(len(V)):
    fit,cov = np.polyfit(V[i][lims[i][0]:lims[i][1]],lnI[i][lims[i][0]:lims[i][1]],1,cov=1)
    fits.append(fit)
    Vs = np.linspace(V[i][lims[i][0]],V[i][lims[i][1]],100)
    plt.plot(V[i],lnI[i],'o')
    plt.plot(Vs,fit[0]*Vs + fit[1])
    plt.title("LED I-V curves: %s" %labels[i])
    plt.xlabel("Voltage (V)")
    plt.ylabel("lnCurrent (lnI)")
    plt.show()



#%% Reverse bias diodes 300K



#%% Reverse bias diodes 77K



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
