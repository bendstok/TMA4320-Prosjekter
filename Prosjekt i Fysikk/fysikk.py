import numpy as np
import matplotlib.pyplot as plt
# Gjør figurer manuelt større
plt.rcParams['figure.figsize'] = [12, 8]
"""
oppgave a
"""

R = 8.31446261815324

""""
T_c = 8 * a / (27 * R * b) 
p_c = a / 27 * b**2
V_c = 3 * b
"""


# Eksperiementelle verdier for vann
#kelvin
T_c = 647.096

#pascal
p_c = 22064000

#liter
eV_c = 0.055948



#fra t_c og p_c, får vi:
a = 27 * R**2 * T_c**2 / (64 * p_c)
b = R * T_c / (8 * p_c)

print("Volum  gitt: " +  str(3 * b))

# Fra https://www.engineeringtoolbox.com/water-properties-temperature-equilibrium-pressure-d_2099.html#heat_capacity
# T -> 640K P -> 20 300 000 Bar
eT_eksp = 640
ep_eksp = 20300000
a_eksp = 27 * R**2 * T_c**2 / (64 * p_c)
b_eksp = R * T_c / (8 * p_c)

print("Volum litteratur: " + str(3 *b_eksp))

"""
oppgave b
"""


plottstart = 75
plottslutt = 300
plottmengde = (plottslutt - plottstart) + 1

Vplot = np.linspace(plottstart, plottslutt, plottmengde)

def vanderwaalP(R, T, V, a, b):
    vanP = R*T/(V-b) - a/(V**2)
    return vanP

vanP = np.zeros(plottmengde)
for i in range(plottmengde):
    vanP[i] = vanderwaalP(R, T_c, Vplot[i], 27 * R**2 * T_c**2 / (64 *  22.064), R * T_c / (8 * 22.064))
plt.plot(Vplot, vanP)
plt.grid()
plt.xlabel("Volum [mL]")
plt.ylabel("Trykk [MPa]")
plt.title(r"Trykk kurven for vann gitt ved volum ved, $T = 647.096 K$")
plt.show()

"""
Denne isotermen har et sadelpunkt, og går ikke oppover, men nedover.
Den andre isotermenn bøyer seg langt ned, så opp igjen, for så å gå ned igjen
lavere temperatur vil gjøre at den bøyer seg kraftigere.
det kritiske punktet er der hvor den slutter å gå oppover, og begynner da bare å gå nedover. idk.
"""

"""
oppgave c
"""

"""
1 = sinh^2(2c/T_c)

c = 1
"""
def eksfuncTC(t_c):
    return np.sinh(2*c/t_c) - 1

def deriverte(f, x, h):
    deriv = (f(x + h) - f(x)) / h
    return deriv

def newtmet(f, x, h, tol, k):
    xliste = np.zeros(k)
    xliste[0] = x
    for i in range(k):
        x -= (f(x) / deriverte(f, x, h))
        xliste[i+1] = x
        if f(x) < tol:
            break
    return x, xliste

c = 1
h = 0.01
tol = 0.0001
k = 10000

start = 0.5
t_c = start



numericalTC, xlisteNewton = newtmet(eksfuncTC, t_c, h, tol, k)
print(numericalTC)

analyticTC = 2/np.log(1 + 2**0.5)
print(analyticTC)

"""
very close. viser at det er omtrent det ja pog
startverdi? (ser på det senere lol)
"""

"""
oppgave d
"""

#  Utregner e_i
e_i = abs(xlisteNewton - analyticTC)

# Fjærner unødvendige verdier for plotting, funksjonen blir konstant
plotting_e_i =  np.zeros(100)
for val in range(100):
    plotting_e_i[val] = e_i[val]

# Plotting
plt.plot(plotting_e_i)
plt.plot()
plt.grid()
plt.xlabel("Verdi for i")
plt.ylabel(r"Verdi  for $e_i$")
plt.title(r"Kurven til $e_i$ ved økende i")
plt.show()

# Utregning av p_i
p_i = np.zeros(9)

# Forsiktig med rangen her, blir fort uendelig store tall!
for xx in range(2,9):
    p_i[xx] = np.log(e_i[xx] / (e_i[xx - 1])) / np.log(e_i[xx - 1]/e_i[xx - 2])


# Fjærning av de to nullene foran 
index = [0,1]
# Bare for å være ordentlig forsiktig at det ikke blir tull
p_i = np.copy(np.delete(p_i, index))

print("Utregnet q: " + str(np.mean(p_i)))

# Kan vell si at det er omtrentlig lik 2, som er verdien for Newtons Metode?