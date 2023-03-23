"""
Damptrykkurven
Skrevet av: Lasse Matias Dragsjø, Bendik Kvamme Stokland, og Thomas Olaussen

Damptrykkurven er en termofysisk begrep som beskriver overgangen fra vann til damp ved ulik trykk, temperatur, og volum.
Den illustreres vanligvis i en plott med varierende trykk og temperatur, eller varierende trykk og volum.
Denne krven, og flere slike kurver, er nyttige i termodynamikken, siden de kan anvendes til mange ulike ingeniørformål
Kurvene kan bli hentet på flere måter:
Enten kan den løses analytisk, eller man kan bruke van der Waals ligning til å få et foreklet modell av den.
Verdiene til deres variabler kan deretter bli funnet ved å gjøre eksperimentelle utøvelser.

I dette rapporten går vi gjennom hvordan vi gjør de numeriske metodene:
Vi går gjennom van der waals tilstandsligning, og hvorfor den ikke er tilstrekkelig til å beskrive kurven
Så går vi gjennom ?, og finner fram hvordan vi skal finne disse kurvene.

Som en oppvarming, starter vi med å se på på noen eksperimentelle verdier for van der waals tilstandsligning.
"""

# Importerer biblioteker
import numpy as np
import matplotlib.pyplot as plt

# Gjør figurer manuelt større
plt.rcParams['figure.figsize'] = [12, 8]

"""
oppgave a

Vi ser på de kritiske verdiene til temperatur, trykk, og volum for van der waals tilstandsligning.
Den kritiske kunptet er ver punktet på kurvediagrammene der forskjellene mellom væske og gass opphører.
Van der waals tilstandsligning er gitt ved:

p = RT /(V - b)  - a/V^2,

der a og b er konstanter som man finner eksperimentellt.
De kritiske verdiene for p, T, og V er:

T_c = 8 * a / (27 * R * b) 
p_c = a / 27 * b**2
V_c = 3 * b

Vi undersøker hva a og b blir med eksperimentelle verdier for disse kritiske verdiene,
og drøfter videre hva dette forteller oss.
"""

# Universelle gasskonstant
R = 8.31446261815324


# Eksperiementelle verdier for vann
T_c = 647.096 #Kelvin
p_c = 22064000 #Pascal
V_c = 0.000055948 #kubikkmeter

# BØR VI BRUKE MPA OG ML ISTEDENFOR? DET GJØR PLOTTEN FRA 1B MYE MER MENING

#fra t_c og p_c, får vi:
a = 27 * R**2 * T_c**2 / (64 * p_c)
b = R * T_c / (8 * p_c)

print("Volum fra b: " +  str(3 * b))
print("Volum eksperimentellt: " +  str(V_c))


# Fra https://www.engineeringtoolbox.com/water-properties-temperature-equilibrium-pressure-d_2099.html#heat_capacity
# T -> 640K P -> 20 300 000 Bar

eT_eksp = 640
ep_eksp = 20300000
a_eksp = 27 * R**2 * eT_eksp**2 / (64 * ep_eksp)
b_eksp = R * eT_eksp / (8 * ep_eksp)

print("Volum fra b (engi tool): " +  str(3 * b_eksp))
print("Volum eksperimentellt (engi tool): " +  str(V_c))

# FINN DEN EKSPERIMENTELLE VOLUM HER? ^^



"""
Her ser vi at den eksperimentelle verdien for 5.6*10^-5, mens den gir 9.1*10^-5 når verdeien fra a og b puttes inn i volumet
Dette viser oss at selv om vi har fått a ob b fra T_c og p_c, så er det ingen kombinasjoner av a bo g som også gir samme volum.
"""



"""
oppgave b

Nå plotter vi van der waals kurve med en satt T_c, fra 75ml til 300ml, og sammenligner det med kurven fra figur 2 på vår guide.
"""

# VISE ET BILDE FRA GUIDEN OM DET?

# Plotteverdier
plottstart = 75
plottslutt = 300
plottmengde = (plottslutt - plottstart) + 1

# Lager plottepunktene for volum
Vplot = np.linspace(plottstart, plottslutt, plottmengde)

# Definerer van der Waals ligning
def vanderwaalP(R, T, V, a, b):
    
    """
    Van der Waals ligning
    
    ...
    
    Input:
    R: universelle gasskonstant
    T: temperatur
    V, volum
    a, konstant a
    b, konstant b
    
    Output:
    vanP, van der Waal trykk
    """
    
    vanP = R*T/(V-b) - a/(V**2)
    return vanP

# itererer gjennom van der Waals ligning med volumene
vanP = np.zeros(plottmengde)
for i in range(plottmengde):
    vanP[i] = vanderwaalP(R, T_c, Vplot[i], 27 * R**2 * T_c**2 / (64 *  22.064), R * T_c / (8 * 22.064))

# Plotter    
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
det kritiske punktet er der hvor den slutter å gå oppover, og begynner da bare å gå nedover.
"""



"""
oppgave c

Nå som vi har fått litt innsikt i wan der Waals ligning, gåt vi over et verktøy vi skal bruke til å få damptrykkurven: Newtons metode
Vi ser på et eksempel: faseovergangen i et 2d ising-modell.

Fra ordnet til ordnet tilstand, har kritisk temperatur denne ligningen:
Sinh^2(2c/T_c) = 1.

Den analytiske løsningen til den er T_c = 2c / ln(1 + 2^0.5).
Vi bruker c = 1, og finner løsningen numerisk ved newtons metode
"""



# Definerer ising-modell funksjonen
def eksfuncTC(t_c):
    
    """
    ising-modell funksjonen, slik at riktig svar gir 0
    
    ...
    
    Input:
    t_c: kritisk temperatur
    
    Output:
    ising-modell funksjonen, slik at riktig svar gir 0
    """
    
    return np.sinh(2*c/t_c) - 1

# Definerer numerisk derivajson
def deriverte(f, x, h):
    
    """
    Numerisk derivasjon
    
    ...
    
    Input:
    f: funksjon
    x: x-punkt
    h: steglengde
    
    Output:
    deriv: f'(x), dem deriverte i det punktet.
    """
    
    deriv = (f(x + h) - f(x)) / h
    return deriv

# Definerer newtons metode for én variabel.
def newtmet(f, x, h, tol, k):
    
    """
    Numerisk derivasjon
    
    ...
    
    Input:
    f: funksjon
    x: x-punkt
    h: steglengde
    tol = feiltoleranse
    k = antall maksimum iterasjoner
    
    Output:
    x: x-punktet som er innenfor toleransebetingelsen til riktig svar
    xliste: en liste av alle x-punktene funksjonen har gått gjennom
    """
        
    xliste = np.zeros(k)
    xliste[0] = x
    for i in range(k):
        x -= (f(x) / deriverte(f, x, h))
        xliste[i+1] = x
        if f(x) < tol:
            break
    return x, xliste



# Henter verdier
c = 1
h = 0.01
tol = 0.0001
k = 10000

start = 0.5
t_c = start


# Finner numerisk og analytiske svar, og printer dem
numericalTC, xlisteNewton = newtmet(eksfuncTC, t_c, h, tol, k)
print(numericalTC)

analyticTC = 2/np.log(1 + 2**0.5)
print(analyticTC)


# Finner numeriske svar med ulike startverdier
numericalTCliste = np.zeros(100)
for i in range(100):
    start = i
    numericalTCliste[i] = newtmet(eksfuncTC, t_c, h, tol, k)[0]
print(numericalTCliste)

"""
Verdiene til den analytisle og den numeriske er veldig nerme hverandre.
Vi kan dermed si at vi har implementert newtons meotde riktig, og at den analytiske løsningen er riktig

For ulike startverdier, ser vi at alle startverdier fra 0 til 100 konvergerer til det samme tallet.
Dermed har det ikke noe å si hvor man starter ved dette inntervallet
"""



"""
oppgave d

Newtons metode konvergerer kvadratisk nor den har et fungerende startpunkt.
Dette er analytisk vist.
Vi finner ut av dette numerisk, ved å bruke en formel som viser omtrentlig konvergensraten q
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

"""
Vi ser at verdien for q er omtrent 2, som er det den analytiske løsninger foreslår til oss.
"""
