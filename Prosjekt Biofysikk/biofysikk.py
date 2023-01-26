'''Finne tumor med numeriske metoder

Dette prosjektet angår å bruke numeriske metoder for å finne mulige tumor i kroppen. Rapporten går gjennom bit for bit hvordan en slik kode skal konstrueres, og går i detalj hvorfor koden blir konstruert slik. Prosjektet ender med en fungerende metode å finne tumor i pasienter.

Vi må først forstå metoden før vi går løs på å kode en løsning. Metoden baserer seg på å måle vannmolekylenes bevegelse i kroppen. Siden mennesker har 70% vann, er dette en god tilnermelse. De måles i den virkelige verden ved å bruke deres magnetiske egenskaper, og deres måter å bevege seg i kroppen. Denne bevegenlsen er kalt for dispersjon. Dispersjon forteller hvordan vannmolekyler sprer seg over tid, ved at vannet sprer seg tregere i områder med høyere materialtetthet. Dette er nyttig, siden tumorer er karakterisert ved ukontrollert celledeling, som gir høyere materialtetthet. Til sist kan vi måle vannets dispersjon ved å se på hvordan vannets mangetiske egenskaper retter seg opp enten ved samme sted, eller andre steder. Dette betyr at vi kan bruke magnetiske målinger til å finne tumorer.

Først går vi nermere inn på dispersjonslikningen:
![bilde.png](attachment:bilde.png)
![bilde-9.png](attachment:bilde-9.png)

konstanten D er dispersjonskonstanten. jo lavere den er, jo tregere sprer molekyler seg. Matematikere har vist at dispersjon følger en gaussisk sannsynlighetsfordeling, og at forventningsverdien til posisjonen av et vannmolekyls posisjon, når det går ut i det uendelige, er startpunktet selv. Først skal vi vise at hvis σ^2 = at, så løser dette dispersjonslikningen ved riktig valg av a:

![bilde-4.png](attachment:bilde-4.png) --> ![bilde-5.png](attachment:bilde-5.png) --> ![bilde-2.png](attachment:bilde-2.png)
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

'''1-d virrevandring (Oppgave 1b)'''
# Importerer libraries
import numpy as np
import matplotlib.pyplot as plt
import time

# Setter konstanter
dx = 1
dt = 1
# Grense for å gå til høyre i virrevandringen
høyreSannsynlighet = 0.5
M = 10
def virrevandring(n, høyreSannsynlighet, tilfeldigeTall, dx, dt):
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
    posisjon = np.zeros(n)
    tidsIntervaller = np.linspace(0, dt*(n-1), (n))
    for i in range(M-1):
        if tilfeldigeTall[i] < høyreSannsynlighet:
            posisjon[i+1] = posisjon[i] + dx 
        else:
            posisjon[i+1] = posisjon[i] - dx
    return(posisjon, tidsIntervaller)

randomnums = np.random.uniform(0,1,M-1)
print(virrevandring(M, høyreSannsynlighet, randomnums, dx, dt))



'''
Med denne enkle modellen, tester vi den med ulike sannsynligheter å gå til høyre eller venstre, for å sjekke om de er representative med den virkelige verden:
'''

'''plotter høyreSannsynlighet = 0.45, 0.50, og 0.55, med 10000 steg (Oppgave 1c)'''
M = 10000
randomnums = np.random.uniform(0,1,M-1)
for i in range(3):
    høyreSannsynlighet = i*0.05 + 0.45
    plotterVirrevandring = virrevandring(M, høyreSannsynlighet,randomnums, dx, dt)
    #plt.plot(plotterVirrevandring[1], plotterVirrevandring[0])

    
'''
Nå som vi har en virrevandrer, lager vi flere av dem samtidig. Vi lager den rask, og viktigst av alt, forståelig
'''
    
'''N virrevandrere med M-1 bevegelser (Oppgave 1d)'''

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
    ''' ^^ eh skal det ikke være nXm eller mXn?'''
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


randomnums = np.random.uniform(0,1,(10,10))

time_slow_b4 = time.time()
print(n_antall_virrevandrere(10,10,0.5,randomnums, 1,1))
time_slow_after = time.time()
time_slow_diff = time_slow_after - time_slow_b4
print(time_slow_diff)

#M = 10
#N = 10
#høyreSannsynlighet = 0.5
#
#N_virrevandrere = np.zeros((N, M))
#randomnums = np.random.uniform(0,1,(N,M-1))
#for i in range(N):
#    N_virrevandrere[i] = virrevandring(M, høyreSannsynlighet, randomnums[i], dx, dt)[0]
#
#print(N_virrevandrere)



'''
Nå forbedrer vi kodene slik at de kan kjøre raskere. En forklaring på hvordan de er raskere er uder denne koden
'''

'''kumulativVirrevandring = kVv. kumulativPosisjon = kP'''
'''Raskere (muligens) versjon av forrige kode (Oppgave 1e)'''
M = 10
N = 5
randomNums = np.random.uniform(0,1,(M-1)*N)

def kVv(M, N, randomNums, dx, dt):
    kP = np.copy(randomNums)
    høyreSannsynlighet = 0.5
    kP[kP > høyreSannsynlighet] = dx
    kP[kP < høyreSannsynlighet] = -dx
    kP = kP.reshape(N,M-1)
    tidsIntervaller = np.linspace(0, dt*(M-1), (M))
    return(kP, tidsIntervaller)

print(kVv(M, N, randomNums, dx, dt))


'''Med en forbedret kode, finner vi dens empiriske varians, slik at vi kan sammenligne den med den analytiske løsningen på første oppgave. Vi forklarer observasjonene vi får av koden, og hvordan den kan nermere bli lik det analytiske svaret.'''

"""Oppgave 1f"""

def empirisk_varians(Matrise):
    """
    Regner ut empirisk varians til hver kollonne til en MxM matrise

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


positions, time_intervall = n_antall_virrevandrere(10,10,0.5,randomnums, 1,1)
variance_pos = empirisk_varians(positions)

#plt.plot(time_intervall, variance_pos)
#plt.show()

'''Skal ikke scipy brukes her?, og skal ikke variansen forklares?? (scipy.optimize.curve_fit)'''

'''Hvis vi ønsker at den empiriske variansen skal samsvare mer med den analytiske resultatet i 1a, så bør vi ha større M og N.'''
'''For M sin del, er det fordi tilfeldighet vil i løpet av uendelig tid jevne ut sine tilfeldigheter, og gi ut den ekte sannsynlighetsfordelingen; som i dette tilfellet er den analytisle empiriske variansen.'''
'''For N sin del, er det fordi de vil gi et bedre gjennomsnittsverdi, ved at flere av den samme typen vil gi en feil proporsjonal med 1/n; så med n --> Uendelig, gir dette oss det analytiske svaret.'''
