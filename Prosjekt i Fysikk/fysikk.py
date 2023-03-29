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
import scipy
from scipy.optimize import curve_fit
from scipy import interpolate

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
def deriverte(func, x, h = 10e-6):
    
    """
    Numerisk derivasjon
    
    ...
    
    Input:
    func: funksjon
    x: x-punkt
    h: steglengde
    
    Output:
    deriv: f'(x), den deriverte i det punktet.
    """
    
    deriv = (func(x + h) - func(x)) / h
    return deriv

# Definerer newtons metode for én variabel.
def newtmet(func, x, tol, k):
    
    """
    Rask versjon av newtons metode for én variabel
    
    ...
    
    Input:
    func: funksjon
    x: x-punkt
    tol = feiltoleranse
    k = antall maksimum iterasjoner
    
    Output:
    x: x-punktet som er innenfor toleransebetingelsen til riktig svar
    xliste: en liste av alle x-punktene funksjonen har gått gjennom
    """
        
    xliste = np.zeros(k)
    xliste[0] = x
    for i in range(k):
        x -= (func(x) / deriverte(func, x))
        xliste[i+1] = x
        if func(x) < tol:
            break
    return x, xliste



# Henter verdier
c = 1
tol = 0.0001
k = 10000

start = 0.5
t_c = start


# Finner numerisk og analytiske svar, og printer dem
numericalTC, xlisteNewton = newtmet(eksfuncTC, t_c, tol, k)
print(numericalTC)

analyticTC = 2/np.log(1 + 2**0.5)
print(analyticTC)


# Finner numeriske svar med ulike startverdier
numericalTCliste = np.zeros(100)
for i in range(100):
    start = i
    numericalTCliste[i] = newtmet(eksfuncTC, t_c, tol, k)[0]
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
plotting_e_i = np.zeros(100)
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
Den viser oss et verdi som er litt lavere enn det, men det er trolig grunnet numeriske feil.
"""


"""
Oppgave 1e
"""


def NewtonTwoVariable(F, J, X, max_i, tol = 10e-4):
    
    """
    Løser et ikke lineært system med newtons to variabel metode.
    
    Input:
    F: Funksjonene slik at F = 0\n
    J: Jacobi matrisen til F\n
    X: Start verdier\n
    max_k: max antall iterasjoner\n
    tol: Toleranse
    
    Output:
    X: verdiene til nullpunkt
    """
    
    F_val = F(X)
    F_norm = np.linalg.norm(F_val)
    iter = 0
    while(abs(F_norm) < tol and iter <= max_i):
        delta = np.linalg.solve(J(X), - F_val)
        X = X + delta
        F_val = F(X)
        F_norm = np.linalg.norm(F_val)
        iter += 1
    return X

def V_11(T,V):

    
    return 0 - (R*T)/(V-b) + a/(V^2)


#Ferdig kode
def func(V,T,a,b):
    """
    Likningssystemet som skal løses
    
    Input:
    V: Liste med variabler (Volum), V[0] = V_v, V[1] =V_g
    T: Temperatur
    a: Konstant
    b: Konstant
    
    Output:
    V_list: Utregnet svar
    """
    #Definerer Gasskonstanten
    R = 8.31446261815324
    
    #Lager en liste for systemet av likninger
    V_list=np.zeros(2)
    
    #Regner ut
    V_list[0]=(R*T)/(V[1]-b)-a/(V[1]**2)-(R*T)/(V[0]-b)+a/(V[0]**2)
    V_list[1]=R*T/(V[1]-V[0])*np.log((V[1]-b)/(V[0]-b))-a/(V[1]*V[0])-R*T/(V[1]-b)+a/(V[1]**2)
    return V_list

def Jacobi(V,T,a,b):
    """
    Jacobi matrisen til likningssystemet func
    
    Input:
    V: Liste med variabler (Volum), V[0] = V_v, V[1] =V_g
    T: Temperatur
    a: Konstant
    b: Konstant
    
    Output:
    matrix: Jacobi matrisen til likningsystemet func
    """
    #Definerer Gasskonstanten
    R = 8.31446261815324
    #Oppretter Jacobi matrise   
    matrix = np.zeros((2,2))
    #Regner ut Jacobi matrisen
    matrix[0,0] = R*T/((V[1]-b)**2)+2*a/(V[0]**3)
    matrix[0,1] = -R*T/((V[1]-b)**2)+2*a/(V[1]**3)
    matrix[1,0] = -R*T/((V[1]-V[0])*(V[0]-b))+V[1]*a/((V[1]*V[0])**2)+R*T*np.log((V[1]-b)/(V[0]-b))/((V[1]-V[0])**2)
    matrix[1,1] = R*T/((V[1]-V[0])*(V[1]-b))+R*T/((V[1]-b)**2)+V[0]*a/((V[1]*V[0])**2)+V[0]*a/((V[0]*V[1])**2)-2*a/(V[1]**3)-R*T*np.log((V[1]-b)/(V[0]-b))/((V[1]-V[0])**2)
    return matrix

def newtonMultiple(func, Jacobi, x, T, a, b, h=0.0001, tol=0.0001, k=1000):
    """
    Funksjonen bruker Newtons metode for å løse likningssytemet
    
    Input:
    func: Likningssystemet som skal løses
    Jacobi: Jacobi matrisen til likningssystemt
    x: Startsverdi for variablene som det skal løses for
    T: Temperatur
    a: Konstant
    b: Konstant
    h: Steglengde
    tol: Toleranse
    k: Maks antall iterasjoner
    
    Output:
    x_list[-1]: Nullpunkt for likningsystemet
    """
    #Oppretter liste av lister for å lagre nullpunkter og setter inn startsverdi
    x_list=np.zeros((k+1,len(x)))
    x_list[0]=x
    
    #Newtons metode
    for i in range(k):
        x_list[i+1] = x_list[i]-np.linalg.inv(Jacobi(x_list[i],T,a,b))@func(x_list[i],T,a,b)
        if abs(func(x_list[i+1],T,a,b)).max()<tol:
            x_liste = x_liste[0:i+2]
            break
    #Returnerer det siste steget
    return x_list[-1]

#Regner ut V_v og V_g for forskjellige T
#Lager liste med T verdier mellom T_lower og T_upper
T_lower=274
T_upper=647
T_list=np.linspace(T_lower,T_upper,T_upper-T_lower+1)

#Lager list for å lagre V_v og V_g
V_v = np.zeros(len(T_list))
V_g = np.zeros(len(T_list))

#Setter konstanter (KAN FJERNES HVIS DET ALLEREDE ER DEFINERT)
a=0.5534843454745333
b=3.0480961647686777e-05

#Setter startspunkt for newtons metode (V_0[0]: V_v, V_0[1]: V_g)
V_0 = np.array([12658e-6,35.6e-6])

#Regner ut de først V_v og V_g med utgangspunkt i V_0 og T_lower
V_v[0] = newtonMultiple(func,Jacobi,V_0,T_lower,a,b)[0]
V_g[0] = newtonMultiple(func,Jacobi,V_0,T_lower,a,b)[1]

#Regner ut resten av V_v og V_g med forskjellig T
V_list = np.array([V_v[0],V_g[0]])
for i in range(1,len(T_list)):
    V_list = newtonMultiple(func,Jacobi,V_list,T_list[i],a,b,tol=0.1)
    V_v[i] = V_list[0]
    V_g[i] = V_list[1]
    
#Plotter
plt.plot(T_list,V_v,label="V_v")
plt.legend()
plt.show()
plt.plot(T_list,V_g,label="V_g")
plt.legend()
plt.show()

"""
Oppgave 1f
Vi henter eksperimentelle verdier fra nettsiden https://www.engineeringtoolbox.com/water-properties-d_1573.html:
"""

# 1 bar = 100'000 Pa
barToPa = 10e5

TrykkP = barToPa * np.array([6.1E-03, 0.0099, 0.0354, 0.105, 0.272, 0.622, 1.29, 2.46, 4.37, 7.34, 11.7, 17.9, 26.4, 44.6, 71.1, 108, 159, 180, 203, 221])

# Verdier som passer for Vv, Vg, og L:
TrykkVerdier = barToPa * np.array([6.1E-03, 0.0354, 0.622, 26.4, 203, 221])

"""
Krever 1e for videre koding av 1f
"""



"""1g"""

def p_8(T,V):
    return (R*T)/(V-b) - a/(V**2)

T_g = 460

# Tilfeldige verdier
vv = 10
p_v = 10
p_g = 30


volume_space = np.linspace(5,12,380) # Plot
plt.plot(volume_space, p_8(T_g,volume_space), label="P(V)")
plt.plot(vv, p_v, 'bo', label=r"Punkt $(V_v,p(V_v))$")
plt.plot(vv, p_g, 'ro', label=r"Punkt $(V_v,p(V_v))$")
plt.axline((vv,vv),(p_v,p_g), ls="--", c="r",label=r"Linje Gjennom $(V_v,p(V_v)$ og $(V_v,p(V_g)$))")
plt.xlabel("Volum [L]")
plt.ylabel("Trykk [P]")
plt.title("")
plt.legend()
plt.show()
            

            

# Oppgave 2


"""2a"""
# https://www.engineeringtoolbox.com/water-properties-d_1573.html
# https://www.engineeringtoolbox.com/water-properties-temperature-equilibrium-pressure-d_2099.html


# Vi har 1 mol
Mm = 18E-3

#               0.01C 26.9C 86.9C 227C 367C 373.946C 
T2 =  np.array([273.16, 300, 360, 500, 640, 647.1])
Vg2 = np.array([Mm/0.00485, Mm/0.02558, Mm/0.3786, Mm/13.20, Mm/177.1, Mm/322.0]) 
Vv2 = np.array([Mm/999.8, Mm/996.5, Mm/967.4, Mm/831.3, Mm/481.5, Mm/322.0])
#                      25C     90C   220C    360C 373.95C
L2  = np.array([45054, 43988, 41120, 33462, 12967, 0]) # Joule / mol 


"""2b"""
def Vv2t_func(X,a,b,c,d,e):
    return  a*np.exp(X)+b -c*X**2 + d*X + e*X**3

def Vg2t_func(X,a,b,c):
    return  a*(b**X) + c

fig_t, (ax_t_vg, ax_t_vv, ax_t_l) = plt.subplots(1,3)

popt_vv2, covt_vv2 = curve_fit(Vv2t_func,T2,Vv2)
popt_vg2, covt_vg2 = curve_fit(Vg2t_func,T2,Vg2)

ax_t_vg.plot(T2,Vg2,"r+", label=r"Data for $V_g(T)$")
ax_t_vg.plot(T2, Vg2t_func(T2,*popt_vg2), c="r", label=r"Curvefit for $V_g(T)$")
ax_t_vg.set_xlabel("Temperatur [K]")
ax_t_vg.set_ylabel(r"Volum [$m^3$]") 
ax_t_vg.legend()

ax_t_vv.plot(T2,Vv2, "b+", label=r"Data for $V_v(T)$")
ax_t_vv.plot(T2, Vv2t_func(T2,*popt_vv2), c="b", label=r"Curvefit for $V_v(T)$")
ax_t_vv.set_xlabel("Temperatur [K]")
ax_t_vv.set_ylabel(r"Volum [$m^3$]") 
ax_t_vv.legend()
fig_t.tight_layout()

def l2t_func(X,a,b,c,d):
    return d*X**3 + a*X**2 + b*X + c

popt_L2, covt_L2 = curve_fit(l2t_func,T2,L2)

ax_t_l.plot(T2,L2,"r+", label=r"Data for $L(T)$")
ax_t_l.plot(T2, l2t_func(T2,*popt_L2), c="r", label=r"Curvefit for $L(T)$")
ax_t_l.set_xlabel("Temperatur [K]")
ax_t_l.set_ylabel(r"Latent Varme [$Joule/Mol$]") 
ax_t_l.legend()
fig_t.suptitle("Kurve interpolering av innhentet data")
plt.show()

"""2c"""
n = abs(274-647-1)
T_intervall = np.linspace(274,647,n)

a = T_intervall[0]
b = T_intervall[-1]

h = (b-a)/n

def p_int(T):
    return l2t_func(T, *popt_L2)/(T*(Vg2t_func(T,*popt_vg2) - Vv2t_func(T,*popt_vv2)))

I_simp = p_int(a)

I_simp_val = np.zeros(n)
I_simp_val[0] = p_int(a)

for i in range(1,n):
    if (i % 2 == 0):
        I_simp += 2*p_int(T_intervall[i])
        I_simp_val[i] = I_simp
    else:
        I_simp += 4*p_int(T_intervall[i])
        I_simp_val[i] = I_simp
    

I_simp = I_simp*h/3

print(I_simp)

plt.title("Simpsons integrering av likning (13), Kurve interpolering")
plt.xlabel("Temperatur [K]")
plt.ylabel("Trykk [P]")
plt.plot(T_intervall,I_simp_val)
plt.show()

"""2d"""

interpol_vv = interpolate.CubicSpline(T2,Vv2)
interpol_vg = interpolate.CubicSpline(T2,Vg2)
interpol_l  = interpolate.CubicSpline(T2,L2)

fig_2d, (ax_t_vg2d, ax_t_vv2d, ax_t_l2d) = plt.subplots(1,3)

popt_vv2, covt_vv2 = curve_fit(Vv2t_func,T2,Vv2)
popt_vg2, covt_vg2 = curve_fit(Vg2t_func,T2,Vg2)

ax_t_vg2d.plot(T2,Vg2,"r+", label=r"Data for $V_g(T)$")
ax_t_vg2d.plot(T2, Vg2t_func(T2,*popt_vg2), c="r", label=r"Curvefit for $V_g(T)$")
ax_t_vg2d.plot(T2,interpol_vg(T2),c='g', label="Kubisk Spline")
ax_t_vg2d.set_xlabel("Temperatur [K]")
ax_t_vg2d.set_ylabel(r"Volum [$m^3$]") 
ax_t_vg2d.legend()

ax_t_vv2d.plot(T2,Vv2, "b+", label=r"Data for $V_v(T)$")
ax_t_vv2d.plot(T2, Vv2t_func(T2,*popt_vv2), c="b", label=r"Curvefit for $V_v(T)$")
ax_t_vv2d.plot(T2,interpol_vv(T2),c='g', label="Kubisk Spline")
ax_t_vv2d.set_xlabel("Temperatur [K]")
ax_t_vv2d.set_ylabel(r"Volum [$m^3$]") 
ax_t_vv2d.legend()
fig_2d.tight_layout()

ax_t_l2d.plot(T2,L2,"r+", label=r"Data for $L(T)$")
ax_t_l2d.plot(T2, l2t_func(T2,*popt_L2), c="r", label=r"Curvefit for $L(T)$")
ax_t_l2d.plot(T2,interpol_l(T2),c='g', label="Kubisk Spline")
ax_t_l2d.set_xlabel("Temperatur [K]")
ax_t_l2d.set_ylabel(r"Latent Varme [$Joule/Mol$]") 
ax_t_l2d.legend()
fig_2d.suptitle("Figur av Kubisk spline mot Kurve interpolering")
plt.show()

"""2e"""

def p_interpol(T):
    return interpol_l(T)/(T*interpol_vg(T) - interpol_vv(T))

n = abs(274-647-1)
T_intervall = np.linspace(274,647,n)

a = T_intervall[0]
b = T_intervall[-1]

h = (b-a)/n

I_simp_inter = p_interpol(a)

I_simp_val_inter = np.zeros(n)
I_simp_val_inter[0] = p_interpol(a)

for i in range(1,n):
    if (i % 2 == 0):
        I_simp_inter += 2*p_interpol(T_intervall[i])
        I_simp_val_inter[i] = I_simp_inter
    elif(i == n):
        I_simp_inter += p_interpol(T_intervall[i])
        I_simp_val_inter[i] = I_simp_inter
    else:
        I_simp_inter += 4*p_interpol(T_intervall[i])
        I_simp_val_inter[i] = I_simp_inter
    

I_simp_inter = I_simp_inter*h/3

print(I_simp_inter)

plt.title("Simpsons integrering av likning (13), Kubisk spline")
plt.xlabel("Temperatur [K]")
plt.ylabel("Trykk [P]")
plt.plot(T_intervall,I_simp_val_inter)
plt.show()

"""2f"""

# Finne et punkt (p0,t0), bruker krital punktet
p0_an = p_c
L_an = 1000 # Vi må finne egen verdi som passer best
T0_an = T_c

analytic_solution = p0_an*np.exp((L_an/R) * (1/T0_an - 1/T_intervall))

plt.plot(T_intervall, analytic_solution, label="Analytisk løsning, L = {}".format(L_an))
plt.plot(T_intervall,I_simp_val,label="Kurve interpolering")
plt.plot(T_intervall,I_simp_val_inter,label="Kubisk spline")
plt.xlabel("Temperatur [K]")
plt.ylabel("Trykk [P]")
plt.legend()
plt.show()
