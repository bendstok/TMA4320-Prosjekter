import numpy as np
import matplotlib.pyplot as plt

"""
oppgave a
"""

R = 8.31446261815324

"""
T_c = 8 * a / (27 * R * b) 
p_c = a / 27 * b**2
V_c = 3 * b
"""

#kelvin
eT_c = 647.096

#megapascal
ep_c = 22.064

#milliliter
eV_c = 55.948


#IKKE SI-ENHETER ^^^^ DE ER MEGAPASCAL OG MILLILITER

"""
fra t_c og p_c, får vi:
"""
a = 27 * R**2 * T_c**2 / (64 * p_c)
b = R * T_c / (8 * p_c)

ea = 27 * R**2 * ekspT_c**2 / (64 * ekspp_c)
eb = R * ekspT_c / (8 * ekspp_c)

print(3 * ekspB)
print(ekspV_c)
"""
sjekk om det er samme enhet? men ser at ~91 > ~56
"""


"""
oppgave b
"""


plottstart = 75
plottslutt = 300
plottmengde = (plottslutt - plottstart) + 1
Vplot = np.linspace(plottstart, plottslutt, plottnengde)
def vanderwaalP(R, T, V, a, b):
    vanP = R*T/(V-b) - a/(V**2)
    return vanP

vanP = np.zeros(plottmengde)
for i in range(plottmengde):
    vanP[i] = vanderwaalP(R, eT_c, Vplot[i], ea, eb)
plt.plot(Vplot, vanP)
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



numericalTC = newtmet(eksfuncTC, t_c, h, tol, k)[0]
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

"""
e_i, calc that. se på notatene lol
"""
