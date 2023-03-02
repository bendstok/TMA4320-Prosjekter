"""punktum ta det lol."""

"""
T E K S T MARKDOWN!!!!
# Dictionary learning

I denne rapporten ser vi på det som kalles for "Dictionary learning", og hvordan en maskin kan gjenkjenne fra hverandre to bilder av ulike tall, 0, og 1. <br>
Vi ser først gjennom noen matematiske prosesser for å forstå hvordan vi kan utføre dictionary learning. <br>
Så bruker vi data fra MNIST til å utføre dictionary learning, og ser hvordan den projekterer nye bilder til dictionary-en (ordboka) den har lært. <br>
Til slutt ser vi på hvordan maskinen klassifiserer mange ulike tallbilder i dybde. <br> <br>

Først undersøker vi matematikken. Matrisene A representerer sett med "bilder", der hver kolonne skal representere et bilde. <br>
Disse bildene kan brukes til å trene opp datamaskinen, og til å få dem i maskinens dictionary. <br>
Kolonnevektorene b representerer nye bilder. <br>
Disse brukes til å teste hvordan maskinen ser på nye bilder i forhold til dictionary-en sin, altså hvordan den klassifiserer de nye bildene.
T E K S T MARKDOWN!!!!
"""

#importerer biblioteker
import numpy as np
import matplotlib.pyplot as plt
import os 

"""
T E K S T MARKDOWN!!!!
### Oppgave 1a
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
Vanligvis skal den være (3, 3). det viser oss at den siste kolonnen var bare fyllt med 0, og er dermed ubrukelig for oss. <br>
Dette ser vi er tilfelle, fordi ved å rekonstruere A1, ser vi at vi får samme resultat. <br>
Vi finner vi ut hvor viktig de to andre kolonnene er til å rekonstruere A.
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
Vi ser at ved å fjerne kolonne 2, får vi omtrent samme svar, bortsett fra indeks A[1][1], som er nå 10^-6, og ikke 1. <br>
Med første kolonne vekk istedenfor, får vi en matrise som ikke på noen måte har de samme verdiene der verdiene ikke er 0. <br>
Det viser oss at den første kolonnen er viktigst for å rekonstruere matrisen. <br>
Det er fordi np.linalg.svd sorterer kolonnene fra mest viktig til minst viktig.
T E K S T MARKDOWN!!!!
"""

"""
T E K S T MARKDOWN!!!!
### Oppgave 1b
Nå sjekker vi A2, om hva redusering av dens U-kolonner vil gjøre med rekonstrueringen
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
Her ser vi at S har bare 0 ved dens tredje kolonne.<br>
Det vi gi FS @ Vt som har sin tredge rad fylt med bare 0. <br>
Det vil gjøre den tredje U-kolonnen ubrukelig. <br>
Dermed kan vi fjerne denne U-kolonnen, og de korresponderende delene ved S, og Vt, og likevel få den samme matrisen.
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
Denne er nøyaktig den samme som den originale A2, som bekrefter våre tanker. <br>

Vi ser at slik redusering kan spare oss plass og tid om vi ønsker full rekonstruering, eller rask og nesten perfect rekonstruering.(Små singulærverdier spiller mindre rolle i rekonstrueringen av den original matrisen) <br>
Vi skriver en funksjon til å gjøre dette.
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
### Oppgave 1c
Nå ønsker vi å projektere test-datasettene til de ortogonale dictionaries. <br>
Slik kan maskinen dra inn nye datasett til sine dictionaries. <br>
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
print(f"Projeksjon av B på W1 =\n{orthproj(W1, B)}\n")

# Henter W2, og printer projeksjonen for B på W1
W2 = truncSVD(A2, 3)[0]
print(f"Projeksjon av B på W2 =\n{orthproj(W2, B)}")
      
"""
T E K S T MARKDOWN!!!!
Disse verdiene viser oss projeksjonen av B på W1 og W2. <br>

Nå finner vi deres distanse; distansen fra datasett-matrisene til våre dictionaries. <br>
Distansen brukes til å se hvor nerme matrisene matcher dictionariesene. <br>
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
Her ser vi at distansen fra B til W1, kolonnevis, er [0, 1, 0]. <br>
Dette viser oss at b1  og b2 er inni W1, mens b2 er utenfor W1. <br>
Dette er riktig svar ifølge oppgaveskjemaet vi følger. <br>
For W2, ser vi at alle kolonnene i B er inni W2.
T E K S T MARKDOWN!!!!
"""

"""
T E K S T MARKDOWN!!!!
### Oppgave 1d
Det kan ta lang tid å trene opp en maskin pga. at en SVD kan ta lang tid å renge ut. <br>
Derfor gjør vi også en ikke-negativ fremgang til projeksjon og distansemåling. <br>
Vi lager projeksjonsfunksjonen først.
T E K S T MARKDOWN!!!!
"""

def nnproj(W, B, maxiter=50, safeDiv=10e-10):
    """
    Tar inn et ikke-negativ dictionary W og matrise A og returnerer den ikke negative projeksjonen av B på W.
    
    Input:
    W: Ikke-negativ dictionary
    B: Ikke-negativ datasett-matrise, representerer treningsbilder eller testbilder
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
print(f"Projeksjonen av B på A1, ikke-negativt, =\n{nnproj(A1, B)[0]}\n")
print(f"Vektene til A1, ikke-negativt, =\n{nnproj(A1, B)[1]}\n")
print(f"Projeksjonen av B på A2, ikke-negativt, =\n{nnproj(A2, B)[0]}\n")
print(f"Vektene til A2, ikke-negativt, =\n{nnproj(A2, B)[1]}")

"""
T E K S T MARKDOWN!!!!
Her ser vi projeksjonene, og vektene. Senere vil vi se at disse gir de riktige verdiene, og viser dermed at denne algoritmen funker. <br>
Nå ser vi på distansen fra B til A.
T E K S T MARKDOWN!!!!
"""

def nndist(W, B):
    
    """
    Regner ut kolonnevis avstand fra ikke-negative matrise B til ikke-negativ dictionary W.
    
    Input:
    W: Ikke-negativ dictionary
    B: Ikke-negativ datasett-matrise, representerer treningsbilder eller testbilder
    
    Output:
    dist: Distanse fra B til W.
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
Merk at vi her har distansen A til B, og ikke W til B. <br>
Her ser vi at distansen fra B til A1, kolonnevis, er [0, 1, 1/Sq(2)]. <br>
Dette viser oss at b1 er inni A1, mens b2 og b3 er utenfor A1. <br>
Vi ser her at b3 er innenfor spennet av SVD A1, men ikke i kjeglen av den ikke-negative A1. <br>
Dette er riktig svar ifølge oppgaveskjemaet vi følger. <br>
For A2, ser vi neglisjerbart de samme verdiene, som også er forskjellig fra den vanlige SVD A2.
T E K S T MARKDOWN!!!!
"""

"""
T E K S T MARKDOWN!!!!
### Oppgave 2a
Nå som vi har matematikken nede og testet, begynner vi med MNIST dataset. <br>
Vi laster det ned, og printer ut de 16 første 0-ene.
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
    Plotter de første nplot*nplot bildene i imgs på et nplot*nplot grid.
    Antar høyde=bredde, og at bildene er lagret kolonnevis.
    
    Input:
    imgs: (høyde=bredde,N) array som inneholder bildene. N > nplot**2
    nplot: Heltall. nplot**2 bilder vil bli plottete
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

plotimgs(train[:,0,:], nplot=4)


"""
T E K S T MARKDOWN!!!!
Her ser vi de første 16 0-ene. nå bruker vi 0-ene
T E K S T MARKDOWN!!!!

T E K S T MARKDOWN!!!!
### Oppgave 2b
Nå ser vi litt på deres SVD, egenskaper. <br>
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
Her ser vi de 16 første U-kolonnene. <br>
Merk at disse ikke er de sammen som de første 0-bildene. <br>
Vi ser at disse bildene inneholder viktige egenskaper som tallet 0 har, og at dens egenskaper blir mimdre og mindre. <br> representerende for tallet 0. <br>

Vi ser på dens singulærvekotrer plottet logaritmisk vis for å mer innsikt i dem.
T E K S T MARKDOWN!!!!
"""

plt.semilogy(S)


"""
T E K S T MARKDOWN!!!!
Plotten vise oss at de aller første bildene har store singulæregenvektorer. <br>
Den viser at de synker kraftig ned først, men etterpå går den ganske sakte nedover, med mange singulærvektorer som er ontrent det samme. <br>
Hvis vi hadde større d, ville vi ha sett at den begynner å gå kraftig ned igjen, helt til at den krasjer til neglisjerbart 0. <br>
    
Dette forteller oss at singulærvektorene inneholder noen få viktige bilder, mange mindre viktige, men brukbare bilder, og en del bilder som er neglisjerbare. 
T E K S T MARKDOWN!!!!
"""

"""
T E K S T MARKDOWN!!!!
Oppgace 2c
T E K S T MARKDOWN!!!!
"""



def manytruncSVD(A, d):
    
    """
    Gjør et svd med en LISTE av de d viktigste leddene i matrisen, altså flere trunkterte versjoner av vanlig SVD-regning.
    
    Input:
    A: Datasett-martise
    d: En liste med antall U-kolonner/S-singulærvektorer/V-rader som skal brukes
    
    Output:
    W: En liste med dictionaries
    """
        
    W = np.array([np.zeros((A.shape[0], A.shape[0]))] * len(d))
    
    maxW = truncSVD(A, max(d))[0]
    
    W[np.argmax(d)][:, :d[np.argmax(d)]] = maxW 
    d[np.argmax(d)] = 0
    
    while max(d) != 0:
        W[np.argmax(d)][:, :d[np.argmax(d)]] = maxW[:, :max(d)]
        d[np.argmax(d)] = 0
    
    return W



def manyorthproj(W, B, antall):
    
    I = ["text"] * antall
    
    images = np.zeros((antall, A.shape[0]))
    
    for i in range(antall):
        images[i] = (np.transpose(orthproj(W[i], b)))
        I[i] = images[i][np.newaxis, :]
    return I

def fiveplotter(W, B, antall):
    """I = image"""
        
    I = manyorthproj(W, B, len(d))
    zeros = np.zeros((1, A.shape[0]))
    b = np.transpose(B)
    
    totimage = np.transpose(np.concatenate((I[0], I[1], zeros, I[2], I[3], zeros, zeros, zeros, b), axis = 0))
    
    plotimgs(totimage, 3)

"""henter verdier"""
A = train[:,c,:n]
d = np.array([16, 32, 64, 128])
W = manytruncSVD(A, d)
antall = 4

"""første bilde"""

b = train[:,0,:1]
fiveplotter(W, b, antall)
"""annen tall"""

b = train[:,1,:1]
fiveplotter(W, b, antall)




"""FMS = Frobenium Norm Squared"""

def FMS(A):
    return sum(sum(A * A))

A = train[:,c,:n]
b = train[:,0,:1]
d = np.arange(1, 784, 20)

W = manytruncSVD(A, d)

I = manyorthproj(W, A[:, :len(d)], len(d))

print(I[0].shape)
print(len(d))

matrisedist = np.zeros(len(d))
print(A[:, :len(d)].shape)
print(A[:, :len(d)][:,0].shape)

for i in range(len(d)):
    matrisedist[i] = FMS(A[:, :len(d)][:,i] - I[i])

plt.semilogy(matrisedist)

annettall = train[:,1,:n]

for i in range(len(d)):
    matrisedist[i] = FMS(annettall[:, :len(d)][:,i] - I[i])

plt.semilogy(matrisedist)
