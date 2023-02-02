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
print("LEGG TIL MATTEBILIDE HER!")
(LEGG TIL MATTEBILIDE HER!)
print("LEGG TIL MATTEBILIDE HER!")
Med a = 2D, løser likningen seg.
'''



'''
Nå går vi løs på det numeriske.
Vi starter med å konstruere en 1-dimensjonal virrevandrer, som beveger seg ett skritt enten til høyre eller til venstre, med lik sannsynlighet. Akkurat nå lager vi en enkel kode. vi forbedrer den senere.
'''

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
#print(virrevandring(M, hS, randomNums, dx, dt))

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
    #plt.plot(plotterVirrevandring[1], plotterVirrevandring[0], label = f"hS = {hS}")
    #plt.xlabel('Tid')
    #plt.ylabel('x-pos')
    #plt.legend()
    #plt.show()

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
#print(n_antall_virrevandrere(N, M, hS, randomNums, dx, dt))



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
#print("Tid treig: " + str(time_slow_diff))
#print("Tid rask: " + str(time_fast_diff))
#print("Tid spart: " + str(time_slow_diff - time_fast_diff))

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
        # j er kolonnen

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
randomNums = np.random.uniform(0,1,(M*N))

# Kjører empirisk_varians
positions, time_intervall = kVv(M, N, hS, randomNums, dx, dt)
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
Funksjonen setter opp rader, kolonner, og variansvatrise til behandling
Den henter ut hver kolonne, og så regner den ut deres gjennomsnittsverdier, og med det, variansen til hver kolonne
Alle deres varians returneres etterpå.
Denne funksjonen brukes for å hente inn variansene.
Så lages en lineær funksjon, og  bruker den i scipy curve fit, for å få den beste matchen av en lineær funksjon til variansen.
funksjonen, og variansmålingene plottes til slutt.
Resultatet viser oss at a ~ 1. Dette betyr at variansen til en virrevandrer øker lineært med tiden.
Det er akkurat dette som skjer ved diffusjon, at variansen er lineært og fullstendig proporsjonal med tida.
Ved sammenigning av oppgave 1a, ser vi at a = 2*D også er lineært, ved D = 0.5!
Begge disse to funnene viser at vi har en modell som faktisk modellerer diffusjon.
Hvis vi ønsker at den empiriske variansen skal samsvare mer med den analytiske resultatet i 1a, så bør vi ha større M og N.
For M sin del, er det fordi tilfeldighet vil i løpet av uendelig tid jevne ut sine tilfeldigheter, og gi ut den ekte sannsynlighetsfordelingen; som i dette tilfellet er den analytisle empiriske variansen.
For N sin del, er det fordi de vil gi et bedre gjennomsnittsverdi, ved at flere av den samme typen vil gi en feil proporsjonal med 1/Sqrt(n); så med n --> Uendelig, gir dette oss det analytiske svaret.
'''



'''
Med en empirisk varians som matcher virrevandringene vi vil ha, kan vi fortsette uten å bekymre off for at vi har gjort feil.
Vi utvider den til en 2d virrevandrer. Vi tester denne utvidelsen med systemer som enten er isotrope, eller ikke.
'''

"""(Oppgave 1g)"""

# 2d virrevandrende funksjon. R = retning, P = posisjon, V = virrevandrer. oS = oppSannsynlighet. Y = Ja, N = Nei
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


# Plottingsfunksjon
def plotToD_virrevandrer(M, N, hS, oS, HogOforhold, dx, dy, dt):
    plt.figure()
    for j in range(2):
        hS = 0.4 + 0.1*j
        oS = 0.6 - 0.1*j
        
        
        plt.subplot(2, 2, 1 + 2*j)
        for i in range(N):
            plotter2dVirrevandring = toD_virrevandrer(M, N, hS, oS, HogOforhold, dx, dy, dt)
            plt.plot(plotter2dVirrevandring[2], plotter2dVirrevandring[0][i], label = f"hS = {hS}")
        plt.xlabel('Tid')
        plt.ylabel('x-pos')
        plt.legend()
        
        plt.subplot(2, 2, 2 + 2*j)
        for i in range(N):
            plotter2dVirrevandring = toD_virrevandrer(M, N, hS, oS, HogOforhold, dx, dy, dt)
            plt.plot(plotter2dVirrevandring[2], plotter2dVirrevandring[1][i], label = f"oS = {oS}")
        plt.xlabel('Tid')
        plt.ylabel('y-pos')
        plt.legend()   
               
    plt.show()
    return
    
    
# Plotter
#plotToD_virrevandrer(M, N, hS, oS, HogOforhold, dx, dy, dt)
    
'''
Her lages en effektig kumulativ kode for 2d virrevandring.
Først bestemmes om virrevandrered går horisontalt eller vertikalt
Så bestemmer den om det er + eller - retningn den beveger seg
Deretter lages xP og yP slik at de beveger seg i kun én retning per iterasjon
Til slutt blir deres tilfedligheter gjort til kumulative sum, og iterert summert over, slik som i oppgave 1e.
Tidsinterval lages, og returneres.
'''



'''
Nå har vi en 2d virrevandrer. Vi koder en anvendelse til den
Vi koder om virrevandreren returnerer til startpunktet, og andelen av virrenandrere som gjør det
'''

"""(Oppgave 1h)"""

# Andel kryssende virrevandrere, 1d
def n_t(M, N, hS, randomNums, dx, dt):
    
    """
    Optelling av hvor mange virrevandrere som krysser origo minst en gang (1 dimensjon)
    
    ...
    
    Input: \n
    enD_virrevandrer --> funksjonen kVv(M, N, hS, randomNums, dx, dt)
    
    Output: \n
    andel --> andelen av de N virrevandrerne som krysset startpunktet minst en gang.
    """
    
    # Henter ut virrevandringene
    sjekkStartpunkt = kVv(M, N, hS, randomNums, dx, dt)
    
    #Setter tallet for antall krysninger av startpunktet
    ant = 0
    
    #itererer gjennom hver virrevandrer, uten å ha med starten; både x of y retning. legger til + 1 hvis den krysser startpunktet
    for i in range(N):
        Ja = sjekkStartpunkt[0][:, 1: len(sjekkStartpunkt[0][i])][i] == 0
        if True in Ja:
            ant += 1
            
    # Regnet ut forhold, og returnerer
    andel = ant / N
    return andel



# Printer andel, 2d
# print(n_t(kVv(M, N, hS, randomNums, dx, dt)))



# Andel kryssende virrevandrere, 2d
def n_t2d(M, N, hS, oS, HogOforhold, dx, dy, dt):
    
    """
    Optelling av hvor mange virrevandrere som krysser origo minst en gang (2 dimensjoner)
    
    ...
    
    Input: \n
    toD_virrevandrer --> funksjonen toD_virrevandrer(M, N, hS, oS, HogOforhold, dx, dy, dt)
    
    Output: \n
    andel --> andelen av de N virrevandrerne som krysset startpunktet minst en gang.
    """
    
    # Henter ut virrevandringene
    sjekkStartpunkt = toD_virrevandrer(M, N, hS, oS, HogOforhold, dx, dy, dt)
    
    #Setter tallet for antall krysninger av startpunktet
    ant = 0
    
    #itererer gjennom hver virrevandrer, uten å ha med starten; både x of y retning. legger til + 1 hvis den krysser startpunktet
    for i in range(N):
        xJa = sjekkStartpunkt[0][:, 1: len(sjekkStartpunkt[0][i])][i] == 0
        yJa = sjekkStartpunkt[1][:, 1: len(sjekkStartpunkt[1][i])][i] == 0
        beggeJa = np.logical_and(xJa, yJa)
        if True in beggeJa:
            ant += 1
            
    # Regnet ut forhold, og returnerer
    andel = ant / N
    return andel



# Printer andel, 2d
# print(n_t2d(toD_virrevandrer(M, N, hS, oS, HogOforhold, dx, dy, dt)))


'''
N_t og N_t2d er veldig like, og gjør omtrent det samme.
De henter først in virrevanderen i funksjonene.
så sjekker den om den er ved startpunktet, for bøde x-posisjon og y-posisjon.
den teller opp hvor mange virrevandrere som gjør det, og regner ut andelen.

Enkel kombinatorikk gir at P(x = 0, t = 1) = 0, for begge dimensjoner.
Dette er fordi den aldri ikke vil bevege seg, så den beveger seg vekk fra startpunktet.
For t = 2, gir dette P(x = 0, t = 2) = 0,5 og P(x = 0, t = 2) = 1/4 for henholdsvis en og to dimensjoner, hvis hS = oS = HogOforhold = 0.25
Dette er fordi det er 50% mulighet å velge den motsatte retningen for 1 dimensjon, siden det er 1 av 2 retningsmuligheter,
og 25% mulighet å velge den motsatte retningen for 2 dimensjoner, siden det er 1 av 4 retningsmuligheter,
'''



'''
Nå bruker vi koden fra forrige oppgave til å teste om n_t og n_t2d over uendelig lang tid vil gi et svar son er nerme det analytiske svaret P(x = 0, t → ∞) = 1.
Vi plotter n(t) for å finne ut av dette.
'''

"""(Oppgave 1i)"""

# Andel kryssende virrevandrere, 1d, plottet
def n_tPlot(M, N, hS, randomNums, dx, dt):
    
    """
    Plotter optelling av hvor mange virrevandrere som krysser origo minst en gang, over tid (1 dimensjon)
    
    ...
    
    Input: \n
    enD_virrevandrer --> funksjonen kVv(M, N, hS, randomNums, dx, dt)
    
    Output: \n
    Plot av n_t --> En plot av andelen virrevandrer som har krysset startpunktet minst en gang, over tid.
    """
    
    # Henter ut virrevandringene
    sjekkStartpunkt = kVv(M, N, hS, randomNums, dx, dt)
    
    #Setter tallet for antall krysninger av startpunktet, og sjekker om virrevandrer er i startpunktet
    andel = np.array([0, 0])
    Ja = sjekkStartpunkt[0][:, 1: len(sjekkStartpunkt[0][0])] == 0
    
    #itererer gjennom hver tidssteg. sjekker om en linje har True, og sletter de linjene etterpå. legger til antall True funnet.
    for i in range(M-1):
        count = np.where(Ja[:,i] == True)[0]
        Ja = np.delete(Ja, count, 0)
        if len(count) != 0:
            add = andel[-1] + (len(count))/N
            andel = np.append(andel, add)
        else:
            andel = np.append(andel, andel[-1])
    
    # Fjerner ektra 0 på starten
    andel = np.delete(andel, 0)
    
    # Plotter n_t over tid
    plt.figure()
    plt.plot(sjekkStartpunkt[1], andel, label = "n_t over tid")
    plt.xlabel('Tid')
    plt.ylabel('n_t')
    plt.legend()
    plt.show()
    return



# Andel kryssende virrevandrere, 2d, plottet
def n_t2dPlot(M, N, hS, oS, HogOforhold, dx, dy, dt):
    
    """
    Plotter optelling av hvor mange virrevandrere som krysser origo minst en gang, over tid (2 dimensjoner)
    
    ...
    
    Input: \n
    toD_virrevandrer --> funksjonen toD_virrevandrer(M, N, hS, oS, HogOforhold, dx, dy, dt)
    
    Output: \n
    Plot av n_t2d --> En plot av andelen virrevandrer som har krysset startpunktet minst en gang, over tid.
    """
    
    # Henter ut virrevandringene
    sjekkStartpunkt = toD_virrevandrer(M, N, hS, oS, HogOforhold, dx, dy, dt)
            
    #Setter tallet for antall krysninger av startpunktet, og sjekker om virrevandrer er i startpunktet
    andel = np.array([0, 0])
    xJa = sjekkStartpunkt[0][:, 1: len(sjekkStartpunkt[0][0])] == 0
    yJa = sjekkStartpunkt[1][:, 1: len(sjekkStartpunkt[1][0])] == 0
    Ja = np.logical_and(xJa, yJa)
    
    #itererer gjennom hver tidssteg. sjekker om en linje har True, og sletter de linjene etterpå. legger til antall True funnet.
    for i in range(M-1):
        count = np.where(Ja[:,i] == True)[0]
        Ja = np.delete(Ja, count, 0)
        if len(count) != 0:
            add = andel[-1] + (len(count))/N
            andel = np.append(andel, add)
        else:
            andel = np.append(andel, andel[-1])
    
    # Fjerner ektra 0 på starten (for å fikse en bug)
    andel = np.delete(andel, 0)
    
    # Plotter n_t over tid
    plt.figure()
    plt.plot(sjekkStartpunkt[2], andel)
    plt.xlabel('Tidssteg')
    plt.ylabel('n_t')
    plt.legend()
    plt.show()
    return



# Setter konstanter og tilfeldige tall
M = 100
N = 100
hS = 0.5
oS = 0.5
dx = 1
dy = 1
dt = 1
HogOforhold = 0.5
randomNums = np.random.uniform(0,1,(M*N))

# Printer ut andelene for 1d of 2d virrevandrer
#n_tPlot(M, N, hS, randomNums, dx, dt)
#n_t2dPlot(M, N, hS, oS, HogOforhold, dx, dy, dt)

'''
Koden printer virrevandrere med normale forhold, og med 1000 av dem med 100000 steg, for å få et nermere svar.
Den teller for hver kolonne om virrevandreren er tilbake til startpunktet, og noterer ned de som har gjort det.

Vi har valgt M = 100000, for her er M = t, og vi ønsker å få et høyt t for å få et resultat som lingner den analytiske.
M = 1000 er der for å gi 3 desimaler ved andelen, for å gi et mer presis tall.

Resultatene viser oss at for 1-dimensjon, er den praktisk talt lik 1.
Dermed har vi vist at P(x = 0, t → ∞) for én dimensjon lik 1.
For to dimensjoner derimot, gir den omtrent 0.75.
Dette alene er ikke så veldig overbevisende at den vil gå til 1
Men, ved å se på 2d-plotten i forhold til 1d-plotten, ser vi at de har samme tendens til å gå mot 1, bare mye tregere.
Siden den oppfører seg akkurat som det én dimensjon gjør, men mye tregere, kan vi se at P (x = 0, t → ∞) for to dimensjoner også er lik 1
'''



'''
Nå går vi over til fase 2 av rapporten; vi gjør nå anveldelsene.
Først lager vi slik at vi kan endre på steglengdene, og tidssteglengdene
Vi finner også Diffusjonskonstanten med de nye stegverdiene
'''

"""(Oppgave 2a)"""

# Setter vilkårlig tall
x_steglengde = 0.000004
y_steglengde = 0.000004
t_steglengde = 0.01

# Setter de vilkårlig tallene som steglengdene
dx = x_steglengde
dz = y_steglengde
dt = t_steglengde

'''
Det er bare å endre på dx, dy, og dt her, fordi vi kodet de andre kodene til å ta inn variabelen dx, dy, og dt istedenfor 1.
med dx = 0.000004 meter, og a = dx, ser vi at D = a / 2 = 0.000002
'''



'''
Sample text
'''

"""Oppgave 2b"""
# I tumor K: del_x reduseres med sqrt(t_k)
# Om tumor K og I overlapper: del_x reduseres med sqrt(t_k * t_i)


def absolute_distance(x_1, y_1, x_2, y_2):
    """
    Input: To punkter (x_1,y_1) og (x_2,y_2)
    Output: Returner absolutt distance
    """
    return(np.sqrt( (x_2 - x_1)**2 + (y_2 - y_1)**2))


def tumor_del_x(space_2d, area, Antall_tumors, central_points, tumor_koeff):
    """
    Input: \n
    space_2d: Et 2 dimensjonlt matrise
    Area: Arealet til tumorene (Alle har likt areal)
    Antall_tumors: Antallet tumors
    Central_Points: Tumorene antas sirkulære, denne listen inneholder alle de sentrale punktene,
    dvs at central_points[0] vil returnere [x,y] til sentrumet i den sirkulære tumoren
    tumor_koeff: Tumor koeffisienten, se oppgavetekst for definisjon
    Return:
    del_x: Inneholder alle del_x til alle punkter i det 2d rommet, dvs at
    del_x[x][y] gir del_x til det punktet
    """
    delta_t = 0.01
    # I millimeter
    friskt_vev_del_x = 0.004
    diffus_frisk = (friskt_vev_del_x)**2/(2*delta_t)

    radius = np.round(np.sqrt(area/np.pi))
    
    # Ideen her er som følger:
    # hvert punkt i tumor_koeff_posisjon tilsvarer samme punkt i space_2d,
    # men i tumor_koeff_posisjon[x][y] appender vi hvilke tumorer som er der, dvs tumor-koeffisientene
    # Eksempel, si 2 tumorer (tumor i og tumor j) deler punkt [x,y]
    # Da vil vi:
    # tumor_koeff_posisjon[x][y].append(tumor_koeff[i])
    # og
    # tumor_koeff_posisjon[x][y].append(tumor_koeff[j])
    # Dette er Loop 1

    # Så vil vi regne ut del_x for alle punktene  ut ifra om d er noen tumorer der
    # dette er Loop 2
    z = len(space_2d)
    rows, cols = (z, z)
    tumor_koeff_posisjon = [[[] for i in range(cols)] for j in range(rows)]

    # Loop 1
    for l in range(Antall_tumors):
        # Looper over tumorene
        # Finner sentral punktene
        x1 = central_points[l][0]
        y1 = central_points[l][1]
        
        # Vi looper over 2d rommet
        for x in range(z):
            for y in range(z):
                # Om punktet er innenfor/på radiusen er tumoren der -> Vi appenderer den spesifikke tumor_koeff
                if(absolute_distance(x1,y1,x,y) <= radius):
                    tumor_koeff_posisjon[x][y].append(tumor_koeff[l])
                else:
                    continue
    
    # Denne matrisen vil inneholde alle del_x
    # som vi utregner i Loop 2 og vil returnere
    del_x = [[[] for i in range(cols)] for j in range(rows)]

    # Loop 2
    for x in range(z):
        # Vi setter standarverdiene
        # og looper over rommet
        d_x = 0.004
        temp = 1
        for y in range(z):

            # Om der er tumor-koeffisienter i listen vil vi regne ut den nye del_x
            # Ellers er det friskt vev

            if len(tumor_koeff_posisjon[x][y]) >= 1:
                # Vi looper over alle koeffisientene og lagrer en temp verdi som vi så trekker ifra d_x
                # og setter in denne verdien i [x,y] punktet vi er på
                for koeff in tumor_koeff_posisjon[x][y]:
                    temp *= np.sqrt(koeff)
                d_x *= temp
                del_x[x][y] = d_x 

                d_x = 0.004
                temp = 1
            else:
                del_x[x][y] = friskt_vev_del_x


    return del_x

# Test Verdier
space2d = [[[] for i in range(10)] for j in range(10)]
area = np.pi
Antall_Tumors = 3
Sentral_Punkt = [[0,0], [4,4], [9,9]]

tumor_koeffisients = [0.02,0.03,0.04]

# print(tumor_del_x(space2d, area, Antall_Tumors, Sentral_Punkt, tumor_koeffisients))


"""Oppgave 2c"""

# Oppstartsverdier gitt i oppgaven
N = 2
M = 1000
Antall_Tumors = 15
tumor_koeffisients = [0.1]*Antall_Tumors
Sentral_Punkt = []
for i in range(Antall_Tumors):
    Sentral_Punkt.append([int(np.random.uniform(0,20)), int(np.random.uniform(0,20))])
space2d = [[[] for i in range(20)] for j in range(20)]
area = np.pi
delx = tumor_del_x(space2d, area, Antall_Tumors, Sentral_Punkt, tumor_koeffisients)


# Samme  gamle funksjon, bare at nå er dx avhengig av posisjon
def virrevandrere_2d(N, M, høyreSannsynlighet, tilfeldigeTall, dx, dt):
    """
    Simulererer n virrevandrer i 2 dimensjoner, modifisert for dx avhengig av posisjonen
    ...
    Input: \n
    N  --> Antall virrevandrere \n
    M  --> Virrevandreren vil bevege seg M-1 ganger \n
    høyreSannsynlighet  --> Tilfeldig tall må være større enn denne for å gå til høyre (+dx) \n
    tilfeldigeTall --> En n*n matrise med tilfeldige tall i intervallet [0,1] \n
    dx --> Hvor langt den vil vandre i x retning pr tidsintervall \n
    dt --> Tidsintervall \n 
    dy --> Hvor langtr den vil vandre i y retning  pr tidsintervall \n
    
    Output: \n
    To matriser, 'posisjon' og 'tidsIntervaller': \n
    posisjon --> En n*M matrise som viser posisjonen til virrevandreren \n
    Posisjon[i] = [[0,0], [0,0] ...]
    Der i er hvilken virrevandrer og da vil Posisjon[i][t] gi tilbake
    [x,y], som viser posisjonen til virrevandrer i på tidspunkt t \n
    tidsIntervaller --> En 1d array med lengde m som viser tidspunktet til en posisjon, 
    """

    # Utrolig dårlig, men rask kodet å få dette til
    # Posisjon matrisen er basically lagt opp slik:
    # Posisjon[i] = [[0,0], [0,0] ...]
    # Der i er hvilken virrevandrer og da vil Posisjon[i][t] gi tilbake
    # [x,y], som viser posisjonen til virrevandrer i på tidspunkt t

    rows, cols = (N, M)
    posisjon = [[[0,0] for i in range(cols)] for j in range(rows)]

    tidsIntervaller = np.linspace(0, dt*(N-1), (M))
    for i in range(N):
        # i er raden
        for j in range(M-1):
            # j er kollonnen
            # vi er i rad i og itererer over den med hjelp av j
            z = np.random.uniform(0,1)
            if(z <= 0.5):
                # Vi går i x-retning
                z = 0
                if tilfeldigeTall[i][j] < høyreSannsynlighet:
                    # dx avhengig av posisjon
                    posisjon[i][j+1][z] = posisjon[i][j][z] + dx[int(posisjon[i][j][z])][int(posisjon[i][j][1])]
                    posisjon[i][j+1][1] = posisjon[i][j][1]
                else:
                    posisjon[i][j+1][z] = posisjon[i][j][z] - dx[int(posisjon[i][j][z])][int(posisjon[i][j][1])]
                    posisjon[i][j+1][1] = posisjon[i][j][1]
            else:
                # Vi går i y-retning
                z = 1
                if tilfeldigeTall[i][j] < høyreSannsynlighet:
                    posisjon[i][j+1][z] = posisjon[i][j][z] + dx[int(posisjon[i][j][0])][int(posisjon[i][j][z])]
                    posisjon[i][j+1][0] = posisjon[i][j][0]
                else:
                    posisjon[i][j+1][z] = posisjon[i][j][z] - dx[int(posisjon[i][j][0])][int(posisjon[i][j][z])]
                    posisjon[i][j+1][0] = posisjon[i][j][0]
    return posisjon, tidsIntervaller
randomNums = np.random.uniform(0,1,(N,M-1))

position, timeintervall = virrevandrere_2d(N, M, 0.5, randomNums, delx, dt)

# Plotting av data
'''
fig, (ax0, ax1) = plt.subplots(nrows=2)

im = ax0.pcolormesh(delx, cmap ='Greens', vmin=0, vmax=0.004)
ax0.set_ylabel(r"Y $10^{-6}m$")
ax0.set_xlabel(r"X $10^{-6}m$")
ax0.set_title(r"Posisjon til tumorer, gjennom $\Delta x$")

fig.colorbar(im, ax=ax0)

label = ["Virrevandrer 1", "Virrevandrer 2"]
color = ["g",  "r"]
for i in range(2):
    y_points = np.zeros(len(position[0]))
    x_points = np.zeros(len(position[0]))
    z_points = timeintervall
    for j in range(len(position[0])):
        x_points[j] = position[i][j][0]
        y_points[j] = position[i][j][1]
    
    ax1.plot(x_points, y_points, color[i], label=label[i])
plt.legend()
plt.tight_layout()
plt.show()
'''

print("Output")
