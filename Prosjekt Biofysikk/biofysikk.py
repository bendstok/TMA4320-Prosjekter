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


Nå går vi løs på det numeriske.
Vi starter med å konstruere en 1-dimensjonal virrevandrer, som beveger seg ett skritt enten til høyre eller til venstre, med lik sannsynlighet. Akkurat nå lager vi en enkel kode. vi forbedrer den senere.

Med denne enkle modellen, tester vi den, med ulike sannsynligheter å gå til høyre eller venstre, for å sjekke om de er representative med den virkelige verden:

Nå som vi har en virrevnadrer, lager vi flere av dem samtidig. vi lager den rask, og viktigst av alt, forståelig:


# https://www.wolframalpha.com/input?i2d=true&i=D%5BDivide%5BPower%5Be%2C-%5C%2840%29Divide%5BPower%5Bx%2C2%5D%2C%5C%2840%292*a*t%5C%2841%29%5D%5C%2841%29+%5D%2Csqrt%5C%2840%292*pi*a*t%5C%2841%29%5D+%2C%7Bx%2C2%7D%5D
# https://www.wolframalpha.com/input?i2d=true&i=D%5BDivide%5BPower%5Be%2C-%5C%2840%29Divide%5BPower%5Bx%2C2%5D%2C%5C%2840%292*a*t%5C%2841%29%5D%5C%2841%29+%5D%2Csqrt%5C%2840%292*pi*a*t%5C%2841%29%5D%2Ct%5D
# a = 2 ^^

'''1-d virrevandring'''
import numpy as np
import matplotlib.pyplot as plt

dx = 1
dt = 1
høyreSannsynlighet = 0.5
M = 10
def virrevandring(n, høyreSannsynlighet, tilfeldigeTall, dx, dt):
    posisjon = np.zeros(M)
    tidsIntervaller = np.linspace(0, dt*(n-1), (M))
    for i in range(M-1):
        if tilfeldigeTall[i] < høyreSannsynlighet:
            posisjon[i+1] = posisjon[i] + dx 
        else:
            posisjon[i+1] = posisjon[i] - dx
    return(posisjon, tidsIntervaller)

randomnums = np.random.uniform(0,1,M-1)
print(virrevandring(M, høyreSannsynlighet, randomnums, dx, dt))


'''plotter p = 0.45, 0.50, og 0.55'''
M = 10000
randomnums = np.random.uniform(0,1,M-1)
for i in range(3):
    høyreSannsynlighet = i*0.05 + 0.45
    plotterVirrevandring = virrevandring(M, høyreSannsynlighet,randomnums, dx, dt)
    plt.plot(plotterVirrevandring[1], plotterVirrevandring[0])
    
'''N virrevandrere med M-1 bevegelser'''
M = 10
N = 10
høyreSannsynlighet = 0.5
N_virrevandrere = np.zeros((N, M))
randomnums = np.random.uniform(0,1,(N,M-1))
for i in range(N):
    N_virrevandrere[i] = virrevandring(M, høyreSannsynlighet, randomnums[i], dx, dt)[0]
print(N_virrevandrere)
