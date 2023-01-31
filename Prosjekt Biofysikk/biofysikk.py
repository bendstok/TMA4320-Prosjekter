'''Finne tumor med numeriske metoder

Dette prosjektet angår å bruke numeriske metoder for å finne mulige tumor i kroppen.
Rapporten går gjennom bit for bit hvordan en slik kode skal konstrueres, og går i detalj hvorfor koden blir konstruert slik.
Prosjektet ender med en fungerende metode å finne tumor i pasienter.

Vi må først forstå metoden før vi går løs på å kode en løsning.
Metoden baserer seg på å måle vannmolekylenes bevegelse i kroppen.
Siden mennesker har 70% vann, er dette en god tilnermelse.
De måles i den virkelige verden ved å bruke deres magnetiske egenskaper, og deres måter å bevege seg i kroppen.
Denne bevegenlsen er kalt for dispersjon.
Dispersjon forteller hvordan vannmolekyler sprer seg over tid, ved at vannet sprer seg tregere i områder med høyere materialtetthet.
Dette er nyttig, siden tumorer er karakterisert ved ukontrollert celledeling, som gir høyere materialtetthet.
Til sist kan vi måle vannets dispersjon ved å se på hvordan vannets mangetiske egenskaper retter seg opp enten ved samme sted, eller andre steder.
Dette betyr at vi kan bruke magnetiske målinger til å finne tumorer.

Først går vi nermere inn på dispersjonslikningen:
![bilde.png](attachment:bilde.png)
![bilde-9.png](attachment:bilde-9.png)

konstanten D er dispersjonskonstanten.
Jo lavere den er, jo tregere sprer molekyler seg.
Matematikere har vist at dispersjon følger en gaussisk sannsynlighetsfordeling, og at forventningsverdien til posisjonen av et vannmolekyls posisjon, når det går ut i det uendelige, er startpunktet selv.
Først skal vi vise at hvis σ^2 = at, så løser dette dispersjonslikningen ved riktig valg av a:

![bilde-4.png](attachment:bilde-4.png) -->
![bilde-5.png](attachment:bilde-5.png) -->
![bilde-2.png](attachment:bilde-2.png)
![bilde-6.png](attachment:bilde-6.png)
![bilde-7.png](attachment:bilde-7.png)
![bilde-8.png](attachment:bilde-8.png)

Med a = 2D, løser likningen seg.

Bilder er tatt fra wolfram alpha, og (Prosjektoppgavearket)
'''


'''
# https://www.wolframalpha.com/input?i2d=true&i=D%5BDivide%5BPower%5Be%2C-%5C%2840%29Divide%5BPower%5Bx%2C2%5D%2C%5C%2840%292*a*t%5C%2841%29%5D%5C%2841%29+%5D%2Csqrt%5C%2840%292*pi*a*t%5C%2841%29%5D+%2C%7Bx%2C2%7D%5D
# https://www.wolframalpha.com/input?i2d=true&i=D%5BDivide%5BPower%5Be%2C-%5C%2840%29Divide%5BPower%5Bx%2C2%5D%2C%5C%2840%292*a*t%5C%2841%29%5D%5C%2841%29+%5D%2Csqrt%5C%2840%292*pi*a*t%5C%2841%29%5D%2Ct%5D
For å skaffe bilder hvis nødvendig ^^
'''


'''
Nå går vi løs på det numeriske.
Vi starter med å konstruere en 1-dimensjonal virrevandrer, som beveger seg ett skritt enten til høyre eller til venstre, med lik sannsynlighet. Akkurat nå lager vi en enkel kode. vi forbedrer den senere.
'''


høyreSannsynlighet = 0.5

'''(Oppgave 1b)'''

# Importerer biblioteker (libraries)
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools as it
from scipy.optimize import curve_fit

# Virrevandring funksjon. hS = høyreSannsynlighet
def virrevandring(M, hS, randomNums, dx, dt):
    
    """
    Simulererer en virrevandrer i en dimensjon
    
    ...
    
    Input: \n
    M  --> Virrevandreren vil bevege seg n-1 ganger \n
    hS  --> Tilfeldig tall må være større enn denne for å gå til høyre (+dx) \n
    randomNums --> En 1d array med lengde (n-1) med tilfeldige tall i intervallet [0,1] \n
    dx --> Hvor langt den vil vandre pr tidsintervall \n
    dt --> Tidsintervall \n 
    
    Output: \n
    To vektorer, 'posisjon' og 'tidsIntervaller':  \n
    posisjon --> En 1d array med lengde M, som viser posisjonen til virrevandreren \n
    tidsIntervaller --> En 1d array med lengde M som viser tidspunktet til en posisjon,
    altså at virrevandreren er i posisjon[n] ved tidspunkt tidsIntervaller[n].  
    """
    
    # Setter tidsintervaller, og en posisjonsvektor
    tidsIntervaller = np.linspace(0, dt*(M-1), (M))
    posisjon = np.zeros(M)
    
    # Itererer gjennom å bevege seg til høyre eller til venstre
    for i in range(M-1):
        if randomNums[i] < hS:
            posisjon[i+1] = posisjon[i] + dx 
        else:
            posisjon[i+1] = posisjon[i] - dx
            
    # Returnerer
    return(posisjon, tidsIntervaller)



# Setter konstanter og tilfeldige tall
dx = 1
dt = 1
M = 10
hS = 0.5
randomNums = np.random.uniform(0,1,M-1)

# Priner vektoren
print(virrevandring(M, hS, randomNums, dx, dt))

'''
Her har vi en kode som definerer en virrevandrende funkjson.
For hver tidssteg beveger den seg ett hakk enten til høyre eller til venstre.
For nå har vi dx = 1 = dt, M = 10, og hS = 0.5, der hS betegner sannsynligheten til å gå til høyre.
'''



'''
Med denne enkle modellen, tester vi den med ulike sannsynligheter å gå til høyre eller venstre, for å sjekke om de er representative med den virkelige verden.
Vi tar hS = 0.45, 0.5, og 0.55.
'''

'''(Oppgave 1c)'''

# Setter konstanter og tilfeldige tall
M = 10000
randomNums = np.random.uniform(0,1,M-1)

# Plotter
for i in range(3):
    hS = i*0.05 + 0.45
    plotterVirrevandring = virrevandring(M, hS, randomNums, dx, dt)
    #plt.plot(plotterVirrevandring[1], plotterVirrevandring[0])

'''
Her plottes hS = 0.45, 0.50, og 0.55.
Vi Bruker 10000 steg for å få det mer representativt.

Dette gjør at vi forventer hS = 0.45 å gi 4500 høyre og 5500 venstre, og netto verdi 4500 - 5500 = -1000
I likhet med hS = 0.55, forventes 5500 til høyre og 4500 til venstre, så 5500 - 4500 = 1000.
Dette er akkurat det vi ser på plotten; dermed er den representativt for hS
'''



'''
Nå som vi har en virrevandrer, lager vi flere av dem samtidig. Vi lager den rask, og viktigst av alt, forståelig
'''
    
'''Oppgave 1d'''

# N_antall_virrevandrere funksjon
def n_antall_virrevandrere(M, N, hS, randomNums, dx, dt):
    
    """
    Simulererer n virrevandrer i en dimensjon
    
    ...
    
    Input: \n
    M  --> Virrevandreren vil bevege seg M-1 ganger \n
    N  --> Antall virrevandrere \n
    hS  --> Tilfeldig tall må være større enn denne for å gå til høyre (+dx) \n
    randomNums --> En N*(M-1) matrise med tilfeldige tall i intervallet [0,1] \n
    dx --> Hvor langt den vil vandre pr tidsintervall \n
    dt --> Tidsintervall \n 
    
    Output: \n
    En matrise 'posisjon' og en vektor 'tidsIntervaller': \n
    posisjon --> En N*M matrise med N virrevandrere som viser posisjonen til virrevandreren ved tidssteg M \n
    tidsIntervaller --> En 1d array med lengde M som viser tidspunktet til en posisjon,
    altså at virrevandreren er i posisjon[n][i] ved tidspunkt tidsIntervaller[n].  
    """
    
    # Setter tidsintervaller, og en posisjonsmatrise
    tidsIntervaller = np.linspace(0, dt*(M-1), (M))
    posisjon = np.zeros((N, M))
    
    #Stacker N virrevandrere til en matrise, og returnerer
    for i in range(N):
        posisjon[i] = virrevandring(M, hS, randomNums[i], dx, dt)[0]
    return posisjon, tidsIntervaller



# Setter konstanter og tilfeldige tall
M = 10
N = 10
hS = 0.5
randomNums = np.random.uniform(0,1,(N,M-1))

# Printer resultat
print(n_antall_virrevandrere(N, M, hS, randomNums, dx, dt))



# Setter konstanter og tilfeldige tall, til å kjøre en kjøretids-test
M = 1000
N = 1000
randomNums = np.random.uniform(0,1,(N,M-1))

# Kjøretids-test (treg)
time_slow_b4 = time.time()
n_antall_virrevandrere(N, M, hS, randomNums, dx, dt)
time_slow_after = time.time()
time_slow_diff = time_slow_after - time_slow_b4

'''
Dette viser en N_antall_virrevandrende funkjson.
Den lager N virrevandrere med M tidsposisjoner, satt sammen til en N*M matrise.
dx = 1 = dt, M = 10, N = 10, hS = 0.5
Koden kjører så en kjøretidstest med M = N = 1000, slik at den kan sammenlignes med en bedre kode senere.
'''



'''
Nå forbedrer vi kodene slik at de kan kjøre raskere. En forklaring på hvordan de er raskere er uder denne koden
kumulativVirrevandring = kVv. kumulativPosisjon = kP
'''

'''(Oppgave 1e)'''

# Kumulativ virrevandring funksjon. k = kumulativ, P = Posisjon
def kVv(M, N, hS, randomNums, dx, dt):
    
    """
    Simulererer n virrevandrere i en dimensjon (rask versjon)
    
    ...
    
    Input: \n
    M  --> Virrevandreren vil bevege seg M-1 ganger \n
    N  --> Antall virrevandrere \n
    hS --> Tilfeldig tall må være større enn denne for å gå til høyre (+dx) \n
    randomNums --> En N*M matrise med tilfeldige tall i intervallet [0,1] \n
    dx --> Hvor langt den vil vandre pr tidsintervall \n
    dt --> Tidsintervall \n 
    
    Output: \n
    En matrise 'posisjon' og en vektor 'tidsIntervaller': \n
    posisjon --> En N*M matrise med N virrevandrere som viser posisjonen til virrevandreren ved tidssteg M \n
    tidsIntervaller --> En 1d array med lengde M som viser tidspunktet til en posisjon,
    altså at virrevandreren er i posisjon[n][i] ved tidspunkt tidsIntervaller[n].  
    """
    
    # Kopierer fra tilfeldiget tall, og deler matriseinnholdet i to mellom hS
    kP = np.copy(randomNums)
    kP[kP > hS] = dx
    kP[kP < hS] = -dx
    
    # kP gjøres om til en matrise, og setter startposisjonen på x = 0
    kP = kP.reshape(N,M)
    kP[:,0] = 0
    
    # Akkumulerer gjennom kumulasjonsradene
    for i in range(N):
        kP[i] = list(it.accumulate(kP[i]))
        
    # Setter tidsintervaller, og returnerer
    tidsIntervaller = np.linspace(0, dt*(M-1), (M))
    return(kP, tidsIntervaller)



# Setter tilfeldige tall med N = M = 1000, til å kjøre en kjøretids-test
randomNums = np.random.uniform(0,1,(M,N))

# Kjøretids-test (rask)
time_fast_b4 = time.time()
kVv(M, N, 0.5, randomNums, 1, 1)
time_fast_after = time.time()
time_fast_diff = time_fast_after - time_fast_b4

# Sammenligner kjøretidene
print("Tid treig: " + str(time_slow_diff))
print("Tid rask: " + str(time_fast_diff))
print("Tid spart: " + str(time_slow_diff - time_fast_diff))

'''
Koden viser en N_antall_virrevandrende funkjson, men bygd på en annen måte.
Istedenfor å bruke mange for-løkker, starter den med å lage hele lengden av tilfeldigheter
Så endres dem til dx eller -dx, avhengig av om den er større eller mindre enn hS
Den gjøres til en matrise, og setter startpunktene med 0.
Deretter brukes itertools til å regne den kumulative summen til hver virrevandrer.
Denne delen bruker en for-løkke, men den brukes ikke så mye som i den andre koden.
Dermed resulterer vi med en kode som er omtrent 5x ganger raskere enn den forrige.
'''



'''
Med en forbedret kode, finner vi dens empiriske varians, slik at vi kan sammenligne den med den analytiske løsningen på første oppgave.
Vi forklarer observasjonene vi får av koden, og hvordan den kan nermere bli lik det analytiske svaret.
'''

"""(Oppgave 1f)"""

# Empirisk_varians funksjon
def empirisk_varians(Matrise):
    
    """
    Regner ut empirisk varians til hver kollonne til en MxM matrise
    
    ...
    
    Input:
    Matrise --> MxM kvadratisk matrise

    Output:
    empirisk_varians --> 1d array, som inneholder den empiriske variansen til tilhørende kollonnen i Matrise, altså er
    empirisk_varians[n] den empiriske variansen til Matrise[i,n], der i går fra 0->n
    """

    coloumns = len(Matrise) 
    rows = coloumns
    variance  = np.zeros(coloumns)

    for j in range(coloumns):
        # j er kollonnen

        # Vil inneholde alle verdier i kolonne j
        coloumn_j = np.zeros(coloumns)

        for i in range(rows):
            coloumn_j[i] = Matrise[i][j]
        
        #Utregning av mean og variansen til kolonne j
        mean = np.sum(coloumn_j)/coloumns
        variance[j] += sum((coloumn_j  - mean)**2)/coloumns

    return variance



# Setter konstanter og tilfeldige tall
M = 100
N = 100
randomNums = np.random.uniform(0,1,(M,N))

# Kjører empirisk_varians
positions, time_intervall = n_antall_virrevandrere(M, N, hS, randomNums, dx, dt)
variance_pos = empirisk_varians(positions)

# Curve fitting
def linear(x, a, b):
    return a*x + b

# Scipy magi skjer under
# Vi  er kun interresert i popt, som inneholder hva den beste verdien av a og b er
popt, pcov = curve_fit(linear, time_intervall, variance_pos)

# Plotting
#plt.plot(time_intervall, linear(time_intervall, *popt), 'r--', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
#plt.plot(time_intervall, variance_pos, label="Empirisk Varians")
#plt.xlabel('Tid')
#plt.ylabel('Verdi')
#plt.legend()
#plt.show()

'''
Forklar det vi ser her btw
'''

'''
Hvis vi ønsker at den empiriske variansen skal samsvare mer med den analytiske resultatet i 1a, så bør vi ha større M og N.
For M sin del, er det fordi tilfeldighet vil i løpet av uendelig tid jevne ut sine tilfeldigheter, og gi ut den ekte sannsynlighetsfordelingen; som i dette tilfellet er den analytisle empiriske variansen.
For N sin del, er det fordi de vil gi et bedre gjennomsnittsverdi, ved at flere av den samme typen vil gi en feil proporsjonal med 1/Sqrt(n); så med n --> Uendelig, gir dette oss det analytiske svaret.
'''



'''
Med en empirisk varians som matcher virrevandringene vi vil ha, kan vi fortsette uten å bekymre off for at vi har gjort feil.
Vi utvider den til en 2d virrevandrer. Vi tester denne utvidelsen med systemer som enten er isotrope, eller ikke.
'''

"""(Oppgave 1g)"""

# 2d virrevandrende funksjon. R = retning, P = posisjon, V = virrevandrer. oS = oppSannsynlighet
def toD_virrevandrer(M, N, hS, oS, HogOforhold, dx, dy, dt):
    
    """
    Simulererer n virrevandrer i 2 dimensjoner

    ...

    Input: \n
    M  --> Virrevandreren vil bevege seg M-1 ganger \n
    N  --> Antall virrevandrere \n
    hS --> Tilfeldig tall må være større enn denne for å gå til høyre (+dx) \n
    oS --> Tilfeldig tall må være større enn denne for å gå opp (+dy) \n
    HogOforhold --> Hvor sannsynlig virrevandreren vil gå horisontalt. Så (1 - HogOforhold) er vertikal sannsynlighet.
    dx --> Hvor langt den vil vandre horisontalt pr tidsintervall \n
    dy --> Hvor langt den vil vandre vertikalt pr tidsintervall \n
    dt --> Tidsintervall \n 
    
    Output: \n
    To matriser, 'xP' og 'xY' og en vektor 'tidsIntervaller': \n
    xP --> En N*M matrise med N virrevandrere som viser den horisontale posisjonen til virrevandreren ved tidssteg M \n
    yP --> En N*M matrise med N virrevandrere som viser den vertikale posisjonen til virrevandreren ved tidssteg M \n
    tidsIntervaller --> En 1d array med lengde M som viser tidspunktet til en posisjon,
    altså at virrevandreren er i posisjon[n][i] ved tidspunkt tidsIntervaller[n]. 
    """
    
    # Bestemmer om å bevege seg i x eller y-retning
    randomNums = np.random.uniform(0,1,(M*N))
    
    rRY = np.copy(randomNums)
    rRY = rRY < HogOforhold
    
    rRN = np.copy(randomNums)
    rRN = rRN > HogOforhold
    
    # Bestemmer retning i x og y-retning
    xR = np.random.uniform(0,1,(M*N))
    xR[xR < hS] = -dx
    xR[xR > hS] = dx
    
    yR = np.random.uniform(0,1,(M*N))
    yR[yR < oS] = -dy
    yR[yR > oS] = dy
    
    # Lager kumulativ 2d-virrevandring
    xP = np.zeros(M*N)
    xP[rRN] = xR[rRN]
    xP = xP.reshape(N,M)
    xP[:,0] = 0
    for i in range(N):
        xP[i] = list(it.accumulate(xP[i]))
    
    yP = np.zeros(M*N)
    yP[rRY] = yR[rRY]
    yP = yP.reshape(N,M)
    yP[:,0] = 0
    for i in range(N):
        yP[i] = list(it.accumulate(yP[i]))
    
    # Tidsinterval, og return
    tidsIntervaller = np.linspace(0, dt*(M-1), (M))
    return(xP, yP, tidsIntervaller)



# Setter konstanter og tilfeldige tall
N = 4
M = 1000
dy = 1
HogOforhold = 0.5


# Plotter
for j in range(2):
    hS = 0.4 + 0.1*j
    oS = 0.6 - 0.1*j
    
    plt.figure()   
    for i in range(N):
        plotter2dVirrevandring = toD_virrevandrer(M, N, hS, oS, HogOforhold, dx, dy, dt)
        plt.plot(plotter2dVirrevandring[2], plotter2dVirrevandring[0][i])
    plt.show()
    
    plt.figure()
    for i in range(N):
        plotter2dVirrevandring = toD_virrevandrer(M, N, hS, oS, HogOforhold, dx, dy, dt)
        plt.plot(plotter2dVirrevandring[2], plotter2dVirrevandring[1][i])
   


"""(Oppgave 1h)"""

def n_t(N,M):
    """
    Optelling av hvor mange virrevandrere som krysser origo minst en gang (en dimensjon)
    ...
    Input: \n
    N  --> Antall virrevandrere \n
    M  --> Virrevandreren vil bevege seg M-1 ganger \n
    
    Output: \n
    n  --> Hvor mange av de N virrevandrerne som krysset origo minst en gang
    """
    
    n = 0
    randomNums = np.random.uniform(0,1,(N,M-1))
    posisjon, tidintervaller = n_antall_virrevandrere(N,M,0.5,randomNums,1,1)
    for i in range(N):
        for j in range(2,M):
            if posisjon[i,j] == 0:
                n += 1
                break
    return n

def n_t2d(N,M):
    """
    Optelling av hvor mange virrevandrere som krysser origo minst en gang (to dimensjoner)
    ...
    Input: \n
    N  --> Antall virrevandrere \n
    M  --> Virrevandreren vil bevege seg M-1 ganger \n
    
    Output: \n
    n  --> Hvor mange av de N virrevandrerne som krysset origo minst en gang
    """
    randomNums = np.random.uniform(0,1,(N,M-1))
    posisjon, tidsintervaller = virrevandrere_2d(N, M, 0.5, randomNums)
    n = 0
    for i in range(N):
        for j in range(1,M):
            if posisjon[i][j][0] == 0 and posisjon[i][j][1] == 0:
                n += 1
                break
    return n

def annenn_t2d(N,M):
    """
    Optelling av hvor mange virrevandrere som krysser origo minst en gang (to dimensjoner)
    ...
    Input: \n
    N  --> Antall virrevandrere \n
    M  --> Virrevandreren vil bevege seg M-1 ganger \n
    
    Output: \n
    n  --> Hvor mange av de N virrevandrerne som krysset origo minst en gang
    """
    posisjon, tidsintervaller = annen_2d(N,M)
    n = 0
    for i in range(N):
        for j in range(1,M):
            if posisjon[i,j,0] == 0 and posisjon[i,j,1] == 0:
                n += 1
                break
    return n

'''
Enkel kombinatorikk gir at P(x = 0, t = 1) = 0, og P(x = 0, t = 2) = 0,5 for en dimensjon, og P(x = 0, t = 1) = 0, og P(x = 0, t = 2) = 1/4 for to dimensjoner.
'''
