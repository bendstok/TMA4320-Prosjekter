"""punktum ta det lol."""

"""
T E K S T MARKDOWN!!!!
Dictionary learning

I denne rapporten ser vi på det som kalles for "Dictionary learning", og hvordan en maskin kan gjenkjenne fra hveradnre to ulike tall, 0, og 1
Vi ser først gjennom noen matematiske prosesser for å forstå hvordan vi kan utføre dictionary learning
Så bruker vi data fra MNIST til å utføre dictionary learning, og ser hvordan den projekterer nye bilder til dictilarien den har lært
Til slutt ser vi på hvordan maskinen klassifiserer mange ulike tallbilder i dybde.


Først undersøker vi matematikken. Matrisene A representerer sett med "bilder", der hver kolonne skal representere en bilde.
Disse bildene kan brukes til å trene opp datamaskinen, og til å få dem i maskinens dictionary
Kolonnevektorene b representerer nye bilder.
Disse brukes til å teste hvordan maskinen ser på nye bilder i forhold til dictionarien sin, altså hvordan den klassifiserer de nye bildene.
T E K S T MARKDOWN!!!!
"""

#importerer biblioteker
import numpy as np
import matplotlib.pyplot as plt
import os 

"""
T E K S T MARKDOWN!!!!
Oppgave 1a
Først skriver vi ned vektorene, som vi skal bruke i denne delen av oppgaven
T E K S T MARKDOWN!!!!
"""

#Skriver inn test-datasetter A1 og A2, hver med kolonner som datapunkter
A1 = np.array([[1000, 1], [0, 1], [0, 0]])
A2 = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]])

#Skriver inn test-vektorer, og setter det sammen til en matrise
b1 = np.array([[2], [1], [0]])
b2 = np.array([[0], [0], [1]])
b3 = np.array([[0], [1], [0]])
B = np.concatenate((b1, b2, b3), axis=1)

"""
T E K S T MARKDOWN!!!!
Det første vi gjør, er å åpne opp A1 ved å regne ut dens SVD: A = U @ S @ Vt. U er dens dictionary W, mens S @ Vt er dens vektmatrise H.
Vi undersøker dens egenskaper
T E K S T MARKDOWN!!!!
"""

# Regner ut SVD
U, S, Vt = np.linalg.svd(A1, full_matrices = False)

# Printer ut svd-matrisene
print(f"U =\n{U} \n")
print(f"Formen på U: {U.shape}\n")
print(f"S (egenvektorene) =\n{S} \n")
print(f"Formen på S: {S.shape}\n")
print(f"Vt =\n{Vt} \n")
print(f"Formen på Vt: {Vt.shape}\n")

# Gjør S sine egenvektorer til en matrise. (FS = Full S)
FS = np.diag(S)

# Rekonstruerer, og printer
A = U @ FS @ Vt
print(f"A1 rekonstruert =\n{A}")

"""
T E K S T MARKDOWN!!!!
Her ser vi at U er på formen (3, 2).
Vanligvis skal den være (3, 3). det viser oss at den siste kolonnen var bare fyllt med 0, og er dermed ubrukelig for oss
Dette ser vi er tilfelle, fordi ved å rekonstruere A1, ser vi at vi får samme resultat
Vi finner vi ut hvor viktig de to andre kolonnene er til å rekonstruere A
T E K S T MARKDOWN!!!!
"""

# Regner ut SVD
U, S, Vt = np.linalg.svd(A1, full_matrices = False)

# Fjerner kolonne 2
U[:,1] = 0

# Rekonstruerer, og printer
A = U @ FS @ Vt
print(f"A1 rekonstruert, uten U[:,1] =\n{A}\n")



# Henter svd
U, S, Vt = np.linalg.svd(A1, full_matrices = False)

# Fjerner kolonne 2
U[:,0] = 0

# Rekonstruerer, og printer
A = U @ FS @ Vt
print(f"A1 rekonstruert, uten U[:,0] =\n{A}")
print("Første vekk gir en matrise som ikke på noen måte har de samme verdiene der verdiene ikke er 0.")
print("Det betyr at denne første kolonnen er viktigst for å rekonstruere matrisen")

"""
T E K S T MARKDOWN!!!!
Vi ser at ved å fjerne kolonne 2, får vi omtrent samme svar, bortsett fra indeks A[1][1], som er nå 10^-6, og ikke 1.
Med første kolonne vekk istedenfor, får vi en matrise som ikke på noen måte har de samme verdiene der verdiene ikke er 0.
Det viser oss at den første kolonnen er viktigst for å rekonstruere matrisen
Det er fordi np.linalg.svd sorterer kolonnene fra mest viktig til minst viktig
T E K S T MARKDOWN!!!!
"""

"""
T E K S T MARKDOWN!!!!
Oppgave 1b
nå sjekker vi A2, om hva redusering av dens U-kolonner vil gjøre med rekonstrueringen
T E K S T MARKDOWN!!!!
"""

# Henter svd
U, S, Vt = np.linalg.svd(A2, full_matrices = False)

# Printer ut svd-matrisene
print(f"U =\n{U} \n")
print(f"Formen på U: {U.shape}\n")
print(f"S (egenvektorene) =\n{S} \n")
print(f"Formen på S: {S.shape}\n")
print(f"Vt =\n{Vt} \n")
print(f"Formen på Vt: {Vt.shape}\n")

"""
T E K S T MARKDOWN!!!!
Her ser vi at S har bare 0 ved dens tredje kolonne.
Det vi gi FS @ Vt som har sin tredge rad fylt med bare 0.
Det vil gjøre den tredje U-kolonnen ubrukelig
Dermed kan vi fjerne denne U-kolonnen, og de korresponderende delene ved S, og Vt, og likevel få den samme matrisen
T E K S T MARKDOWN!!!!
"""

# Fjerner kolonne 3
U[:,2] = 0

# Gjør diagonalvektoren til en matrise
FS = np.diag(S)

# Rekonstruerer, og printer
A = U @ FS @ Vt
print(f"A2 rekonstruert, uten U[:,2] =\n{A}")

"""
T E K S T MARKDOWN!!!!
Denne er nøyaktig den samme som den originale A2, som bekrefter våre tanker


Vi ser at slik redusering kan spare oss plass og tid; om vi ønsker full rekonstryering, eller rask og nesten perfect rekonstruering.
Vi skriver en funksjon til å gjøre dette
T E K S T MARKDOWN!!!!
"""


def truncSVD(A, d):
    
    """
    Gjør et SVD med de d viktigste leddene i matrisen, altså et trunktert versjon av vanlig SVD-regning,
    
    Input:
    A: Datasett-martise
    d: Antall U-kolonner/S-singulærvektorer/V-rader som skal brukes
    
    Output:
    W: Dictionaries
    H: Vekt på dictionariesene
    """
    
    # Regner ut full SVD
    U, S, Vt = np.linalg.svd(A, full_matrices = False)

    # Velger de første d relevante vektorer og verdier
    U = U[:, :d]
    S = S[:d]
    Vt = Vt[:d]
    
    # Gjør diagonalvektoren til en matrise
    FS = np.diag(S)
    
    # Setter inn dictionary og vekt, og returnerer
    W = U
    H = FS @ Vt
    return W, H, S, FS, Vt


"""
T E K S T MARKDOWN!!!!
Oppgave 1c
Nå ønsker vi å projektere test-datasettene til de ortogonale dictionaries.
Slik kan maskinen dra inn nye datasett til sine dictionaries.
Vi tester så dens projeksjon med test-matrisen B på den.
T E K S T MARKDOWN!!!!
"""


def orthproj(W, B):
    
    """
    Tar inn et dictionary med ortogonale kolonner W og et sett med kolonner B og prosjekterer B på W.
    
    Input:
    W: Dictionary med ortogonale kolonner
    B: Datasett-matrise, representerer treningsbilder eller testbilder
    
    Output:
    orthproj: En projektert versjon av B på W
    """
    
    #Transponerer W
    Wt = np.transpose(W)
    
    #Projekterer B på W, og returnerer
    orthproj = W @ Wt @ B
    return orthproj



# Henter W1, og printer projeksjonen for B på W1
W1 = truncSVD(A1, 3)[0]
print(f"Projeksjon fra B på W1 =\n{orthproj(W1, B)}\n")

# Henter W2, og printer projeksjonen for B på W1
W2 = truncSVD(A2, 3)[0]
print(f"Projeksjon fra B på W2 =\n{orthproj(W2, B)}")
      
"""
T E K S T MARKDOWN!!!!
Disse verdiene viser oss projeksjonen fra B på W1 og W2

Nå finner vi deres distance, distancen fra datasett-matrisene til våre dictionaries.
Distansen brukes til å se hvor nerme matrisene matcher dictionariesene.
Vi tester så dens distance med test-matrisen B.
T E K S T MARKDOWN!!!!
"""
      
def ortdist(W, B):
    
    """
    Regner ut kolonnevis avstand fra matrise B til dictionary W.
    
    Input:
    W: Dictionary med ortogonale kolonner
    B: Datasett-matrise, representerer treningsbilder eller testbilder
    
    Output:
    dist: Distanse fra A til W.
    """
    
    # Lager en distansevektor, og setter dem 0. Henter så projeksjonen fra B til W
    dist = np.zeros(len(B[0]))
    proj = orthproj(W, B)
    
    #Regner ut distansene til hvert kolonne, og returnerer deres avstand
    for i in range(len(dist)):
        dist[i] = np.linalg.norm(B[:,i] - proj[:,i])
    return dist

# Printer projeksjonen for B på W1 og W2
print(f"Distansen fra B til W1 =\n{ortdist(W1, B)}\n")
print(f"Distansen fra B til W2 =\n{ortdist(W2, B)}")

"""
T E K S T MARKDOWN!!!!
Her ser vi at distansen fra B til W1, kolonnevis, er [0, 1, 0].
Dette viser oss at b1  og b2 er inni W1, mens b2 er utenfor W1.
Dette er riktig svar ifølge oppgaveskjemaet vi følger
For W2, ser vi at alle kolonnene i B er inni W2
T E K S T MARKDOWN!!!!
"""

"""
T E K S T MARKDOWN!!!!
Oppgave 1d
Det kan ta lang tid å trene opp en maskin ned at en SVD kan ta lang tid å renge ut
Derfor gjør vi også et ikke-negativ frengang til projeksjon og distansemåling
Vi lager projeksjonsfunksjonen først
T E K S T MARKDOWN!!!!
"""

def nnproj(W, B, maxiter=50, safeDiv=10e-10):
    """
    Tar inn et ikke-negativ dictionary W og matrise A og returnerer den ikke negative projeksjonen av A på W.
    
    Input:
    W: Ikke-negativ dictionary
    A: Ikke-negativ datasett-matrise, representerer treningsbilder eller testbilder
    maxiter: Antall iterasjoner brukt for å regne ut den ikke-negative vekt-matrisen H
    safeDiv: Konstant ledd i divisor for å unngå null-divisjon
    
    Output:
    proj: Den ikke-negative projeksjonen av B på W.
    """
    
    #Velger tilfeldlige verdier på H (vekt-matrisen) med verdier fra 0 til 1. Brukes til konvergens til å få den ekte H-matrisen = Wt @ B
    H = np.random.uniform(0, 1, [len(W[0,:]), len(B[0])])
    
    #Transponerer W, og henter inn hjelpeverdiene WtB og WtW
    Wt = np.transpose(W)
    WtB = Wt@B
    WtW = Wt@W
    
    # Itererer maxiter ganger for å konvergere H til den ekte H-matrisen.
    for k in range(maxiter):
        H = H*WtB/(WtW@H+safeDiv)
    
    #projekterer B på W, og returnerer projeksonen
    proj = W@H
    return proj, H

# Printer projeksjonene fra B på A1 og A2, og A-enes vekter:
print(f"Projeksjonen fra B til A1, ikke-negativt, =\n{nnproj(A1, B)[0]}\n")
print(f"Vektene til A1, ikke-negativt, =\n{nnproj(A1, B)[1]}\n")
print(f"Projeksjonen fra B til A2, ikke-negativt, =\n{nnproj(A2, B)[0]}\n")
print(f"Vektene til A2, ikke-negativt, =\n{nnproj(A2, B)[1]}")

"""
T E K S T MARKDOWN!!!!
Her ser vi projeksjonene, og vektene. senere vil vi se at disse gir de riktige verdiene, og viser dermed at denne algoritmen funker.
Nå ser vi på distansen fra B til A
T E K S T MARKDOWN!!!!
"""

def nndist(W, B):
    
    """
    Regner ut kolonnevis avstand fra ikke-negative matrise A til ikke-negativ dictionary W.
    
    Input:
    W: Ikke-negativ dictionary
    A: Ikke-negativ datasett-matrise, representerer treningsbilder eller testbilder
    
    Output:
    dist: Distanse fra A til W.
    """
    # Lager en distansevektor, og setter dem 0. Henter så projeksjonen fra B til W
    dist = np.zeros(len(B[0]))
    proj = nnproj(W, B)[0]
    
    #Regner ut distansene til hvert kolonne, og returnerer deres avstand
    for i in range(len(dist)):
        dist[i] = np.linalg.norm(B[:,i] - proj[:,i])
    return dist

"Tester de forskjellige distanse funksjonene"
print(f"Distansen fra B til A1, ikke-negativt, =\n{nndist(A1, B)}\n")
print(f"Distansen fra B til A2, ikke-negativt, =\n{nndist(A2, B)}")

"""
T E K S T MARKDOWN!!!!
Merk at vi her har distansen til A, og ikke til W
Her ser vi at distansen fra B til A1, kolonnevis, er [0, 1, 1/Sq(2)].
Dette viser oss at b1 er inni A1, mens b2 og b3 er utenfor A1.
Vi ser her at b3 er innenfor spennet av SVD A1, men ikke i kjeglen av den ikke-negative A1.
Dette er riktig svar ifølge oppgaveskjemaet vi følger
For A2, ser vi neglisjerbart de samme verdiene, som også er forskjellig fra den vanlige SVD A2
T E K S T MARKDOWN!!!!
"""

"""
T E K S T MARKDOWN!!!!
Oppgave 2a
Nå som vi har matematikken nede og testet, begynner vi med MNIST dataet.
Vi laster den ned, og printer ut de 16 første 0-ene.
T E K S T MARKDOWN!!!!
"""

# Finner 'veien' til der python filen er på din maskin, for å finne
# train og test filene
# Altså, må train og test filene være i samme mappe som denne fila
dir_path = os.path.dirname(os.path.realpath(__file__))

# Henter treningsbildene og testbildene
train = np.load(dir_path + '/train.npy')/255.0
test = np.load(dir_path + '/test.npy')/255.0

# Kvadratisk bildeplottingsfunksjon
def plotimgs(imgs, nplot = 4):
    """
    Plots the nplot*nplot first images in imgs on an nplot x nplot grid. 
    Assumes heigth = width, and that the images are stored columnwise
    input:
        imgs: (height*width,N) array containing images, where N > nplot**2
        nplot: integer, nplot**2 images will be plotted
    """
    
    # Henter antall bilder, og lengden på bildene 
    n = imgs.shape[1]
    m = int(np.sqrt(imgs.shape[0]))

    #sjekker om det er nok bilder til at de kan plottes
    assert(n >= nplot**2), "Need amount of data in matrix N >= nplot**2"

    # Initialiserer subplots
    fig, axes = plt.subplots(nplot,nplot)

    # Setter bakgrunnsfargen
    plt.gcf().set_facecolor("lightgray")

    # Itererer over bildene
    for idx in range(nplot**2):

        # Bryter av hvis vi går utenfor arrayet
        if idx >= n:
            break

        # Indekser
        i = idx//nplot; j = idx%nplot

        # Fjerner akse
        axes[i,j].axis('off')
        axes[i,j].imshow(imgs[:,idx].reshape((m,m)), cmap = "gray")
    
    # Plotter
    fig.tight_layout()
    plt.show()

plotimgs(train[:,1,:], nplot=4)


"""
T E K S T MARKDOWN!!!!
Her ser vi de første 16 0-ene. nå bruker vi 0-ene

Oppgave 2b
Nå ser vi litt på deres SVd, egenskaper.
Vi regner ut deres SVD, og plotter deres 16 første dictionaries
T E K S T MARKDOWN!!!!
"""

n = 1000 # Antall datapunkter
c = 0 # Klasse
d = 16 # 16 viktigste kolonner

A = train[:,c,:n]
W, H, S, FS, Vt = truncSVD(A, d)

plotimgs(W, nplot = 4)

"""
T E K S T MARKDOWN!!!!
Her ser vi de 16 første U-kolonnene.
Merk at disse ikke er de sammen som de første 0-bildene.
Vi ser at disse bildene inneholder viktige egenskaper som tallet 0 har, og at dens egenskaper blir nimdre og mindre representerende for tallet 0


Vi ser på dens singulørvekotrer plottet logaritmisk vis for å mer innsikt i dem
T E K S T MARKDOWN!!!!
"""

plt.semilogy(S)


"""
T E K S T MARKDOWN!!!!
Plotten vise oss at de aller første bildene har store singulæregenvektorer
Den viser at de synker kraftig ned forst, men etterpå går den ganske sakte nedover, med mange singulærvektorer som er ontrent det samme
Hvis vi hadde større d, ville vi ha sett at den begynner p gå kraftig ned igjen, helt til at den krasjer til neglisjerfart 0

Dette forteller oss at singulærvektorene inneholder noen få viktige bilder, mange mindre viktige, men brukbare bilder, og en del bilder som er neglisjerbart. 
T E K S T MARKDOWN!!!!
"""

"""
T E K S T MARKDOWN!!!!
Oppgace 2c
T E K S T MARKDOWN!!!!
"""
