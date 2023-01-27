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

'''Oppgave 1b'''
# Importerer biblioteker (libraries)
import numpy as np
import matplotlib.pyplot as plt
import time

# Setter konstanter og tilfeldige tall:
dx = 1
dt = 1
M = 10
høyreSannsynlighet = 0.5
randomnums = np.random.uniform(0,1,M-1)

#Virrevandring funksjon
def virrevandring(M, høyreSannsynlighet, randomnums, dx, dt):
    """
    Simulererer en virrevandrer i en dimensjon
    
    ...
    
    Input: \n
    n  --> Virrevandreren vil bevege seg n-1 ganger \n
    høyreSannsynlighet  --> Tilfeldig tall må være større enn denne for å gå til høyre (+dx) \n
    tilfeldigeTall --> En 1d array med lengde (n-1) med tilfeldige tall i intervallet [0,1] \n
    dx --> Hvor langt den vil vandre pr tidsintervall \n
    dt --> Tidsintervall \n 
    
    Output: \n
    To vektorer, 'posisjon' og 'tidsIntervaller':  \n
    posisjon --> En 1d array med lengde M, som viser posisjonen til virrevandreren \n
    tidsIntervaller --> En 1d array med lengde M som viser tidspunktet til en posisjon,
    altså at virrevandreren er i posisjon[n] ved tidspunkt tidsIntervaller[n].  
    """
    
    tidsIntervaller = np.linspace(0, dt*(M-1), (M))
    posisjon = np.zeros(M)
    for i in range(M-1):
        if randomnums[i] < høyreSannsynlighet:
            posisjon[i+1] = posisjon[i] + dx 
        else:
            posisjon[i+1] = posisjon[i] - dx
    return(posisjon, tidsIntervaller)

print(virrevandring(M, høyreSannsynlighet, randomnums, dx, dt))

'''
Her har vi en kode som definerer en virrevandrende funkjson.
For hver tidssteg beveger den seg ett hakk enten til høyre eller til venstre.
For nå har vi dx = 1 = dt, M = 10, og høyreSannsynlighet (hS) = 0.5, der hS betegner sannsynligheten til å gå til høyre.
'''



'''
Med denne enkle modellen, tester vi den med ulike sannsynligheter å gå til høyre eller venstre, for å sjekke om de er representative med den virkelige verden.
'''

'''Oppgave 1c'''
# Setter konstanter og tilfeldige tall:
M = 10000
randomnums = np.random.uniform(0,1,M-1)

#Plotter
for i in range(3):
    høyreSannsynlighet = i*0.05 + 0.45
    plotterVirrevandring = virrevandring(M, høyreSannsynlighet, randomnums, dx, dt)
    #plt.plot(plotterVirrevandring[1], plotterVirrevandring[0])

'''
Her plottes høyreSannsynlighet = 0.45, 0.50, og 0.55.
Vi Bruker 10000 steg for å få det mer representativt.

Dette gjør at vi forventer hS = 0.45 å gi 4500 høyre og 5500 venstre, og netto verdi 4500 - 5500 = -1000
I likhet med hS = 0.55, forventes 5500 til høyre og 4500 til venstre, så 5500 - 4500 = 1000.
Dette er akkurat det vi ser på plotten; dermed er den representativt for hS
'''



'''
Nå som vi har en virrevandrer, lager vi flere av dem samtidig. Vi lager den rask, og viktigst av alt, forståelig
'''
    
'''Oppgave 1d'''
# Setter konstanter og tilfeldige tall:
M = 10
N = 10
randomNums = np.random.uniform(0,1,(N,M-1))

#n_antall_virrevandrere funksjon
def n_antall_virrevandrere(N, M, høyreSannsynlighet, randomNums, dx, dt):
    """
    Simulererer n virrevandrer i en dimensjon
    
    ...
    
    Input: \n
    N  --> Antall virrevandrere \n
    M  --> Virrevandreren vil bevege seg M-1 ganger \n
    høyreSannsynlighet  --> Tilfeldig tall må være større enn denne for å gå til høyre (+dx) \n
    tilfeldigeTall --> En N*(M-1) matrise med tilfeldige tall i intervallet [0,1] \n
    dx --> Hvor langt den vil vandre pr tidsintervall \n
    dt --> Tidsintervall \n 
    
    Output: \n
    En Matrise 'posisjon' og en vektor 'tidsIntervaller': \n
    posisjon --> En N*M matrise med N virrevandrere som viser posisjonen til virrevandreren ved tidssteg M \n
    tidsIntervaller --> En 1d array med lengde M som viser tidspunktet til en posisjon,
    altså at virrevandreren er i posisjon[n][i] ved tidspunkt tidsIntervaller[n].  
    """
    
    tidsIntervaller = np.linspace(0, dt*(M-1), (M))
    posisjon = np.zeros((N, M))
    #Stacker N vittevandrere til en matrise
    for i in range(N):
        posisjon[i] = virrevandring(M, høyreSannsynlighet, randomNums[i], dx, dt)[0]
    return posisjon, tidsIntervaller


print(n_antall_virrevandrere(N, M, høyreSannsynlighet, randomNums, dx, dt))


'''
Dette viser en N_antall_virrevandrende funkjson.
Den lager N virrevandrere med M tidsposisjoner, satt sammen til en N*M matrise.
dx = 1 = dt, M = 10, N = 10, høyreSannsynlighet (hS) = 0.5
'''

'''
def n_antall_virrevandrere(N, M, høyreSannsynlighet, tilfeldigeTall, dx, dt):
    """
    Simulererer n virrevandrer i en dimensjon

    ...

    Input: \n
    N  --> Antall virrevandrere \n
    M  --> Virrevandreren vil bevege seg M-1 ganger \n
    høyreSannsynlighet  --> Tilfeldig tall må være større enn denne for å gå til høyre (+dx) \n
    tilfeldigeTall --> En n*n matrise med tilfeldige tall i intervallet [0,1] \n
    dx --> Hvor langt den vil vandre pr tidsintervall \n
    dt --> Tidsintervall \n 
    
    Output: \n
    To matriser, 'posisjon' og 'tidsIntervaller': \n
    posisjon --> En n*n matrise med lengde , som viser posisjonen til virrevandreren \n
    tidsIntervaller --> En 1d array med lengde 10 som viser tidspunktet til en posisjon,
    altså at virrevandreren er i posisjon[n][i] ved tidspunkt tidsIntervaller[n].  

    """
    posisjon = np.zeros((N,N))
    '''''' ^^ eh skal det ikke være nXm eller mXn?''''''
    tidsIntervaller = np.linspace(0, dt*(N-1), (N))
    for i in range(M):
        # i er raden
        for j in range(M-1):
            # j er kollonnen
            # vi er i rad i og itererer over den med hjelp av j
            if tilfeldigeTall[i][j] < høyreSannsynlighet:
                posisjon[i][j+1] = posisjon[i][j] + dx 
            else:
                posisjon[i][j+1] = posisjon[i][j] - dx
    return posisjon, tidsIntervaller
'''
M = 1000
N = 1000
randomNums = np.random.uniform(0,1,(N,M-1))
time_slow_b4 = time.time()
n_antall_virrevandrere(M,N,0.5,randomNums, 1,1)
time_slow_after = time.time()
time_slow_diff = time_slow_after - time_slow_b4

'''
posisjon = np.zeros((N,N))
    ''''''<-- eh skal det ikke være nXm eller mXn?''''''
    tidsIntervaller = np.linspace(0, dt*(N-1), (N))
    for i in range(M):
        # i er raden
        for j in range(M-1):
            # j er kollonnen
            # vi er i rad i og itererer over den med hjelp av j
            if tilfeldigeTall[i][j] < høyreSannsynlighet:
                posisjon[i][j+1] = posisjon[i][j] + dx 
            else:
                posisjon[i][j+1] = posisjon[i][j] - dx
    return posisjon, tidsIntervaller
'''


'''
Nå forbedrer vi kodene slik at de kan kjøre raskere. En forklaring på hvordan de er raskere er uder denne koden
'''

'''kumulativVirrevandring = kVv. kumulativPosisjon = kP'''
'''(Oppgave 1e)'''
M = 1000
N = 1000
randomNums = np.random.uniform(0,1,(M-1)*N)

def kVv(M, N, randomNums, dx, dt):
    kP = np.copy(randomNums)
    høyreSannsynlighet = 0.5
    kP[kP > høyreSannsynlighet] = dx
    kP[kP < høyreSannsynlighet] = -dx
    kP = kP.reshape(N,M-1)
    tidsIntervaller = np.linspace(0, dt*(M-1), (M))
    return(kP, tidsIntervaller)

time_fast_b4 = time.time()
kVv(M, N, randomNums, dx, dt)
time_fast_after = time.time()
time_fast_diff = time_fast_after - time_fast_b4

print("Tid treig: " + str(time_slow_diff))
print("Tid rask: " + str(time_fast_diff))
print("Tid spart: " + str(time_slow_diff - time_fast_diff))

'''
Raskere (muligens) versjon av forrige kode.
Snakk stuff her
'''


'''
Med en forbedret kode, finner vi dens empiriske varians, slik at vi kan sammenligne den med den analytiske løsningen på første oppgave.
Vi forklarer observasjonene vi får av koden, og hvordan den kan nermere bli lik det analytiske svaret.
'''

"""Oppgave 1f"""
from scipy.optimize import curve_fit

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

M = 100
N = 100
np.random.uniform(0,1,(M-1)*N)

randomXnums = np.random.uniform(0,1,(M,N))
positions, time_intervall = n_antall_virrevandrere(N,M,0.5, randomXnums, 1,1)
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

'''Skal ikke scipy brukes her?, og skal ikke variansen forklares?? (scipy.optimize.curve_fit)'''

'''
Hvis vi ønsker at den empiriske variansen skal samsvare mer med den analytiske resultatet i 1a, så bør vi ha større M og N.
For M sin del, er det fordi tilfeldighet vil i løpet av uendelig tid jevne ut sine tilfeldigheter, og gi ut den ekte sannsynlighetsfordelingen; som i dette tilfellet er den analytisle empiriske variansen.
For N sin del, er det fordi de vil gi et bedre gjennomsnittsverdi, ved at flere av den samme typen vil gi en feil proporsjonal med 1/Sqrt(n); så med n --> Uendelig, gir dette oss det analytiske svaret.
'''


"""Oppgave 1g"""
from mpl_toolkits.mplot3d import axes3d

def virrevandrere_2d(N, M, høyreSannsynlighet, tilfeldigeTall, dx, dy, dt):
    """
    Simulererer n virrevandrer i 2 dimensjoner

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
    # Feks: posisjon[0][0][0] vil gi x posisjonen til virrevandrer 0 på tidspunkt tidsIntervaller[0]

    
    # Bruker vanlige python lister, går kanskje treigt 
    # men klarer ikke vri hodet mitt rundt på hvordan d skal gjøres

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
                    posisjon[i][j+1][z] = posisjon[i][j][z] + dx 
                    posisjon[i][j+1][1] = posisjon[i][j][1]
                else:
                    posisjon[i][j+1][z] = posisjon[i][j][z] - dx
                    posisjon[i][j+1][1] = posisjon[i][j][1]
            else:
                # Vi går i y-retning
                z = 1
                if tilfeldigeTall[i][j] < høyreSannsynlighet:
                    posisjon[i][j+1][z] = posisjon[i][j][z] + dy 
                    posisjon[i][j+1][0] = posisjon[i][j][0]
                else:
                    posisjon[i][j+1][z] = posisjon[i][j][z] - dy
                    posisjon[i][j+1][0] = posisjon[i][j][0]
    return posisjon, tidsIntervaller

M = 10
N = 4
dy = 1
høyreSannsynlighet = 0.5
randomNums = np.random.uniform(0,1,(N,M-1))
positions , tidsIntervall = virrevandrere_2d(N, M, høyreSannsynlighet, randomNums, dx, dy, dt)

# Plotting av data
ax1 = plt.axes(projection="3d")
color = ["red", "green", "blue", "gray"]
for i in range(4):
    y_points = np.zeros(len(positions[0]))
    x_points = np.zeros(len(positions[0]))
    z_points = tidsIntervall
    for j in range(len(positions[0])):
        x_points[j] = positions[i][j][0]
        y_points[j] = positions[i][j][1]
        #ax1.plot3D(x_points, y_points, z_points, color[i])

#ax1.set_xlabel('x')
#ax1.set_ylabel('y')
#ax1.set_zlabel('t')
#plt.show()

"""Annen variasjon av 1g"""

N = 4
M = 1000
pr = 0.5
pu = 0.5
dx = 1
dt = 1

def annen_2d(N,M,pr,pu,dx,dt):
    """
    Simulererer N virrevandrere i 2 dimensjoner
    
    ...
    
    Input: \n
    N --> Antall virrevandrere
    M --> Antall tidssteg
    pr --> Sjanse for å ta steg til høyre
    pu --> Sjanse for å ta steg opp
    dx --> størrelse på steg
    dt --> størrelse på tidssteg
    
    Output: \n
    To vektorer, 'posisjon' og 'tidsIntervaller':  \n
    posisjon --> En 3d array med lengde M og høyde N, der hvert element er en 2d vektor (idk, kan ikke skrive...) \n
    tidsIntervaller --> En 1d array med lengde M som viser tidspunktet til en posisjon,
    altså at virrevandreren er i posisjon[n] ved tidspunkt tidsIntervaller[n].  
    """
    tidsIntervaller = np.linspace(0, dt*(M-1), (M))
    posisjon = np.zeros((N,M,2))
    vertOrHori = np.random.randint(0,2,(N,M-1))
    randomDirection = np.zeros((N,M-1,2))
    chance = (pr,pu)
    for i in range(N):
        for j in range(M-1):
            randomDirection[i,j,vertOrHori[i,j]] = np.random.choice([-1,1],p=[1-chance[vertOrHori[i,j]],chance[vertOrHori[i,j]]])
    for i in range(N):
        for j in range(M-1):
            posisjon[i,j+1] = posisjon[i,j] + randomDirection[i,j]
    return posisjon, tidsIntervaller

"Plotter veien de forskjellige virrevandrerne tok sammen med slutt posisjon"

posisjon,t = annen_2d(N,M,pr,pu,dx,dt)
for i in range(N):
    plt.plot(posisjon[i,:,0],posisjon[i,:,1])
for i in range(N):
    plt.plot(posisjon[i,-1,0],posisjon[i,-1,1],"ko")
plt.show()
