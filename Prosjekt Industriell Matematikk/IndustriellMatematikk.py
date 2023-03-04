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
try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
except:
    print("Could not find dir_path. If using jupyter notebook, ignore")
    train = np.load('train.npy')/255.0
    test = np.load('test.npy')/255.0
else:
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
    
    Output:
    Plot av de første nplot*nplot bildene i imgs på et nplot*nplot grid.
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
Med vår SVD, tester vi dens trunktering på MNIST-datasettet
Vi ser på fire ulike trunkterte svd, hver med økende grad av d valgte elementer.
med disse dictionaris(ene) med ulike d, ser vi hva vi får når det projekteres på en vektore den er trent på, og en vektor den ikke er trent på
T E K S T MARKDOWN!!!!
"""

# Gjør flere trunkert SVD
def manytruncSVD(A, d):
    """
    Gjør et svd med en LISTE av de d viktigste leddene i matrisen, altså flere trunkterte versjoner av vanlig SVD-regning.

    Input:
    A: Datasett-martise
    d: En liste med antall U-kolonner/S-singulærvektorer/V-rader som skal brukes

    Output:
    W: En liste med dictionaries
    S: Singulærvektorene til W. Brukes i 2d
    """

    W = np.array([np.zeros((A.shape[0], A.shape[0]))] * len(d))

    maxW = truncSVD(A, max(d))[0]

    # Itererer gjennom den første W
    W[np.argmax(d)][:, :d[np.argmax(d)]] = maxW 
    d[np.argmax(d)] = 0

    # Itererer gjennom de resterende W, og returnerer W og dens singulærvektorer
    while max(d) != 0:
        W[np.argmax(d)][:, :d[np.argmax(d)]] = maxW[:, :max(d)]
        d[np.argmax(d)] = 0

    return W, S


# Gjør flere ortogonale projiseringer
def manyorthproj(W, B, antall):
    # Setter opp en flerdimensjonal python liste av de projekterte datasettene, og en endimensjonal numpy liste av dem
    I = ["temp"] * antall

    images = np.zeros((antall, A.shape[0]))

    # Itererer gjennom alle projeksjonene, og returnerer dem
    for i in range(antall):
        images[i] = (np.transpose(orthproj(W[i], b)))
        I[i] = images[i][np.newaxis, :]
    return I


# Printer bildene med orignalbildet
def fiveplotter(W, B, antall):
    """
    Plotter flere bilder sammenlignet med den originale, for inspeksjon
    
    Input:
    W: Dictionaries med ortogonale kolonner
    B: Datasett-matrise, representerer treningsbilder eller testbilder
    antall: hvor mange projeksjoner som skal skje
    
    Output:
    En plott av flere bilder sammenlignet med den originale, for inspeksjon
    """
    # Henter de projekterte bildene, nullmatriser, og originalbildet. I = image

    I = manyorthproj(W, B, len(d))
    zeros = np.zeros((1, A.shape[0]))
    b = np.transpose(B)

    # Setter opp bildene, og plotter dem
    totimage = np.transpose(np.concatenate((I[0], I[1], zeros, I[2], I[3], zeros, zeros, zeros, b), axis = 0))

    plotimgs(totimage, 3)


# Henter verdier
A = train[:,c,:n]
d = np.array([16, 32, 64, 128])
W = manytruncSVD(A, d)[0]
antall = 4


# Printer med den første 0-bildet, der 0 er hva dictionarien er trent opp til
b = train[:,0,:1]
fiveplotter(W, b, antall)

# Printer med den første 1-bildet, der 1 er IKKE hva dictionarien er trent opp til
b = train[:,1,:1]
fiveplotter(W, b, antall)

"""
T E K S T MARKDOWN!!!!
Med en vektor dictionarien er trent opp med, ser vi at med høyere d, klarer den å projektere den originale 0 inntil sin dictionary bedre og bedre.
Det samme skjer visuellt med bildet den ikke er trent på, men her er den alltid mye mer blurry enn den andre vektoren.
Det er fordi dette 1-bildet projiserer den som om bildet var en 0, men det er den ikke.
Da får vi blur mellom 0 og 1, der 1 viser sterkere jo høyere d vi får, men får masse artifater rundt eneren
T E K S T MARKDOWN!!!!
"""

"""
T E K S T MARKDOWN!!!!
Oppgave 2d
For å sjekke bildenes sanne forskell med dictionaries(ene) finner vi deres distance til dem
Vi bruker frobenius-norm for å finne distansene til disse bildematrisen og dictionaries(ene)
T E K S T MARKDOWN!!!!
"""

print(2)
# Frobenium Norm Squared = FMS
def FMS(A):
    
    """
    Regner ut Frobeniumn-normen til en matrise, kvadrert
    
    Input:
    A: en matrise
    
    Output:
    frobenium normen kvadrert
    """
    
    return sum(sum(A * A))

# Henter verdier
b = train[:,0,:1]
d = np.arange(1, 784, 20)
W, S = manytruncSVD(A, d)
I = manyorthproj(W, b, len(d))

# Setter opp en liste av distansene fra bildene og dictionarie(sene).
matrisedist = np.zeros(len(d))

# Itererer gjennom FMS, og printer logaritmisk dictionaries(enes) distanser fra første 0-bilde, med hver 20-ende d
for i in range(len(d)):
    matrisedist[i] = FMS(b[:,0] - I[i])
plt.semilogy(matrisedist)

# Itererer gjennom FMS, og printer logaritmisj dictionaries(enes) distanser fra første 1-bilde, med hver 20-ende d
annettall = train[:,1,:1]
for i in range(len(d)):
    matrisedist[i] = FMS(annettall[:,0] - I[i])
plt.semilogy(matrisedist)
plt.show()

"""
T E K S T MARKDOWN!!!!
Her ser vi at distansen fra 0-bildet synker mer og mer jo høyere d vi har.
Dette gjør mening, siden maskinen har da flere vektorer den kan bruke til å bedre projektere bildet
Vi ser også at den ligner veldig mye på bildets singulærverdier.
Dette er fordi grafen av singulærverdiene er også basert på hvor høye d-verdiene er.
Man kan si at verdiene til singulærvektorene forteller viktigheten til et spesifikt dictionary, altså hvor mye "kraft den skal ha",
som betyr hvor mye den skal påvirke projiseringen, og dermed hvor forskjellig den nå er til det reelle bilde, altså avstanden fra b til projektert b.

Det andre bildet, gir samme distanse uansett d, fordi dette bildet er ikke blitt trent opp i dictionarien,
så unasett hva maskienn gjør, har den ingen dictionaries osm matcher det nye bildet.
T E K S T MARKDOWN!!!!
"""

"""
T E K S T MARKDOWN!!!!
Oppgave 2e
Nå gjør vi det samme som vi hittil har gjort, men med den ikke-negative måten.
Vi plotter deres projeksjoner først, og ser hva vi får av dem
T E K S T MARKDOWN!!!!
"""

# Henter verdier
d = 32
A = train[:,c,:n]
nxn = 4

#Henter test-bilder, og dictionaries
Ann = A[:,np.random.choice(A.shape[1],nxn**2,replace=False)]
Wpluss = A[:,np.random.choice(A.shape[1],d,replace=False)]

# Projekterer bildene, og plotter dem
proj = nnproj(Wpluss, Ann)[0]
plotimgs(proj, nxn)


"""
T E K S T MARKDOWN!!!!
Her ser vi tilfeldig valgte opptrente bilder, og tilfeldig valgte test-bilder.
Vi ser at alle sammen er litt blurry i forhold til bildet med d = 32 på oppgave 2c,
siden dictionarisene er ikke sortert fra best til verst, men er heller plukket frem tilfeldig.
T E K S T MARKDOWN!!!!
"""

"""
T E K S T MARKDOWN!!!!
Oppgave 2f
Nå sjekker vi distansene vi får med høyere d, og ulike bilder, med den ikke-negative metoden.
T E K S T MARKDOWN!!!!
"""


# Gjør flere ikke-negative projiseringer 
def manynnproj(Wpluss, B, d):
    """
    Tar inn et LISTE med dictionaries med ikke-negative kolonner W , og et sett med kolonner B og prosjekterer B på alle W.
    
    Input:
    W: Dictionaries med ikke-negative kolonner
    B: Datasett-matrise, representerer treningsbilder eller testbilder
    antall: hvor mange projeksjoner som skal skje
    
    Output:
    manyproj: En projektert versjon av B på W
    """

    # Setter opp en liste av de projekterte datasettene.
    proj = np.array([np.zeros((A.shape[0], 1))] * len(d))

    # Itererer gjennom den første projeksjonen
    proj[np.argmax(d)]= nnproj(Wpluss, b)[0]
    d[np.argmax(d)] = 0

    # Itererer gjennom de resterende krympende W, og projeksjonene, og returnerer dem
    while max(d) != 0:
        Wpluss = Wpluss[:,np.random.choice(Wpluss.shape[1],max(d),replace=False)]
        proj[np.argmax(d)] = nnproj(Wpluss, b)[0]
        d[np.argmax(d)] = 0

    return proj

#Henter verdier
d = np.logspace(1,3,10, dtype = np.int64)

A = train[:,c,:n]
Wpluss = A[:,np.random.choice(A.shape[1],max(d),replace=False)]

b = train[:,0,:1]


manyproj = manynnproj(Wpluss, b, d)

# Setter opp en liste av distansene fra bildene og dictionarie(sene).
print(manyproj.shape)
"""fiks det messet ^^. gosh >.<"""


"""bildene, som er projisert"""

matrisedist = np.zeros(len(d))

# Itererer gjennom FMS, og printer logaritmisk dictionaries(enes) distanser fra første 0-bilde, med hver 20-ende d
for i in range(len(d)):
    matrisedist[i] = FMS(b - manyproj[i])

"""bruk print(a.shape) for å finne ut om det er i en matrise. da funker FMS"""


# Itererer gjennom FMS, og printer logaritmisk dictionaries(enes) distanser fra første 1-bilde, med hver 20-ende d
plt.semilogy(matrisedist)


annettall = train[:,1,:1]

for i in range(len(d)):
    matrisedist[i] = FMS(annettall - manyproj[i])

plt.semilogy(matrisedist)
plt.show()
"""tilfelidgheter kommer her ja. masse ved slutten"""

"""tilfelidgheter kommer her ja. masse ved slutten"""

"""
T E K S T MARKDOWN!!!!
I forhold til plotten fra 2d, ser vi at distansen fra 0-bildet synker først mer og mer jo høyere d vi har.
Men så ser vi at den blir ganske tilfelidg ved de største d, og at det ikke er alltid at distansen er lav.
Det er fordi H-matrisen som konvergerer ved nnproj er nå så stor at maxiter = 50 er for lite til at den kan skikkelig konvergere.
Dette skaper tilfeldighetene, ettersom H vil nå være preget av tilfeldigheter fra randint.

Det andre bildet, gir samme distanse uansett d, fordi dette bildet er ikke blitt trent opp i dictionarien; selv med den ikke-negative metoden.
T E K S T MARKDOWN!!!!
"""

"""
T E K S T MARKDOWN!!!!
Oppgave 3a
T E K S T MARKDOWN!!!!
"""



def generate_test(test, digits = [0,1,2], N = 800):
    
    """
    Tilfeldig genererer test sets.
    
    Input:
        test: numpy array. Test data lastet ned fra filen
        digits: python lists. Inneholder ønskede heltall
        N: Heltall mengde test data for hver klasse
    
    Output:
        test_sub: (784,len(digits)*N) numpy array. Inneholder len(digits)*N bilder
        test_labels: (len(digits)*N) numpy array. Inneholder etiketter korresponderende til bildene av test_sub
    """

    assert N <= test.shape[2] , "N må være mindre enn eller lik det totale mengden av tilgjengelig test data for hver klasse"

    assert len(digits)<= 10, "Liste av tall kan bare holde opp til 10 tall"

    # Arrays til å lagre test sets og etiketter
    test_sub = np.zeros((test.shape[0], len(digits)*N))
    test_labels = np.zeros(len(digits)*N)

    # Itererer over alle tall-klasser, og lagrer test data og etiketter
    for i, digit in enumerate(digits):
        test_sub[:, i*N:(i+1)*N] = test[:,digit,:]
        test_labels[i*N:(i+1)*N] = digit

    # Indekser til å bli shufflet 
    ids = np.arange(0,len(digits)*N)

    # Shufflet indekser
    np.random.shuffle(ids)

    # Returnerer shufflet data 
    return test_sub[:,ids], test_labels[ids]



digits = np.array([0, 1, 2])

N = 800

A_test, A_labels = generate_test(test, digits = digits, N = 800)
print("Test data shape: ", A_test.shape) # Bør være (784,2400)
print("Test labels shape: ", A_labels.shape) # Bør være (2400)
print("First 16 labels: ", A_labels[:16])
plotimgs(A_test, nplot = 4)

# ALSO SJEKK OM ALL TEKST ER NORSK ELLER IKKE




def datacollection(A, B, d):
    
    """
    Henter Dictionary, projeksjon, og distanse av en klasse bilder
    
    Input:
    A: Trenings-bilder
    B: Test-bilder
    d: Antall U-kolonner/S-singulærvektorer/V-rader som skal brukes
    
    Output:
    Wodict: Ortogonale dictionaries
    Wnndict: Ikke-negative dictionaries
    Woproj: Ortogonale projeksjoner
    Wnnproj: Ikke-negative projeksjoner
    Wodist: Ortogonale distanser
    Wnndist: Ikke-negative distanser
    """
        
    # Henter dictionaries
    Wodict = truncSVD(A, d)[0]
    Wnndict = A[:,np.random.choice(A.shape[1],d,replace=False)]
    
    # Regner ut projeksjoner
    Woproj = orthproj(Wodict, B)
    Wnnproj = nnproj(Wnndict, B)[0]
    
    # Regner ut distanser
    Wodist = ortdist(Wodict, B)
    Wnndist = nndist(Wnndict, B)

    return Wodict, Wnndict, Woproj, Wnnproj, Wodist, Wnndist


def klassifisering(A, B, c, d):
    
    """
    Klassifiserer bilder
    
    Input:
    A: Trenings-bilder
    B: Test-bilder
    c: klassen til bildene
    d: Antall U-kolonner/S-singulærvektorer/V-rader som skal brukes
    
    Output:
    Wodict: Ortogonale dictionaries
    Wnndict: Ikke-negative dictionaries
    Woproj: Ortogonale projeksjoner
    Wnnproj: Ikke-negative projeksjoner
    Wodist: Ortogonale distanser
    Wnndist: Ikke-negative distanser
    """
    
    # liste thingy
    Odictlist = np.zeros((len(c), len(B[:,0]), d))
    Ndictlist = np.zeros((len(c), len(B[:,0]), d))
    
    Oprojlist = np.zeros((len(c), len(B[:,0]), N*len(c)))
    Nprojlist = np.zeros((len(c), len(B[:,0]), N*len(c)))
    
    Odistlist = np.zeros((len(c), len(B[0])))
    Ndistlist = np.zeros((len(c), len(B[0])))
    
    # Henter distansene
    
    for i in range(len(c)):
        Ac = A[:, n*i : n*(i+1)]
        Odictlist[i], Ndictlist[i], Oprojlist[i], Nprojlist[i], Odistlist[i], Ndistlist[i] = datacollection(Ac, B, d)
    
    
    classifyOlabels = np.zeros(len(B[0]))
    classifyNlabels = np.zeros(len(B[0]))
    
    
    for i in range(len(B[0])):
        classifyOlabels[i] = c[np.argmin(Odistlist[:,i])]
        classifyNlabels[i] = c[np.argmin(Ndistlist[:,i])]
    
    return classifyOlabels, classifyNlabels, Odictlist, Ndictlist, Oprojlist, Nprojlist, Odistlist, Ndistlist




# Henter verdier
c = digits

A = np.zeros((len(train[:,0,0]), n*len(c)))
              
for i in range(len(c)):
    A[:, n*i : n*(i+1)] = train[:,c[i],:n]

B = A_test
d = 32



truelabel = A_labels
predictions = klassifisering(A, B, c, d)

def recallandacc(c, truelabel, predictions):
    
    # recall
    Orecall = np.zeros(len(c))
    Nrecall = np.zeros(len(c))
    
    for i in range(len(c)):
        OErI = truelabel == c[i]
        OmenerErI = predictions[0][OErI]
        OsammenlignErI = OmenerErI == c[i]
        Oantallriktige = OmenerErI[OsammenlignErI]
        Orecall[i] = len(Oantallriktige) / len(OmenerErI)
        
        NErI = truelabel == c[i]
        NmenerErI = predictions[1][NErI]
        NsammenlignErI = NmenerErI == c[i]
        Nantallriktige = NmenerErI[NsammenlignErI]
        Nrecall[i] = len(Nantallriktige) / len(NmenerErI)
        
    Oacc = sum(Orecall) / len(c)
    Nacc = sum(Nrecall) / len(c)
    
    return Orecall, Nrecall, Oacc, Nacc

print(recallandacc(c, truelabel, predictions))




# class 0

Class = 0

ODist = np.argmin(predictions[6][Class])
NDist = np.argmin(predictions[7][Class])

if predictions[6][Class][ODist] < predictions[7][Class][NDist]:
    
    b = B[:,ODist]
    proj = predictions[4][Class][:,ODist]
    Type = 0
    
else:
    b = B[:,NDist]
    proj = predictions[5][Class][:,NDist]
    Type = 1

def comparepic(b, proj):
    b = b[np.newaxis, :]
    proj = proj[np.newaxis, :]
    
    zeros = np.zeros((1, b.shape[1]))
    totimage = np.transpose(np.concatenate((proj, zeros, zeros, b), axis = 0))
    plotimgs(totimage, 2)

comparepic(b, proj)




# class 0
predict = predictions[0 + Type]

indexfind = np.arange(len(B[0]))

foundindex = indexfind[(truelabel == Class) & (predict != truelabel)]
try:
    index = foundindex[0]
except:
    index = foundindex

b = B[:,index]
proj = predictions[4 + Type][Class][:,index]


comparepic(b, proj)




# Henter verdier
c = np.array([0, 1, 2, 3])

A = np.zeros((len(train[:,0,0]), n*len(c)))
              
for i in range(len(c)):
    A[:, n*i : n*(i+1)] = train[:,c[i],:n]

    
A_test, A_labels = generate_test(test, digits = c, N = 800)

B = A_test
d = 32

truelabel = A_labels
predictions = klassifisering(A, B, c, d)


print(recallandacc(c, truelabel, predictions))




# do safediv = 10^-2? idk

# Henter verdier
c = np.array([0, 1, 2, 3])

A = np.zeros((len(train[:,0,0]), n*len(c)))
              
for i in range(len(c)):
    A[:, n*i : n*(i+1)] = train[:,c[i],:n]

    
A_test, A_labels = generate_test(test, digits = c, N = 800)

B = A_test
truelabel = A_labels

exp = np.arange(10)
d = 2**exp

for i in d:
    predictions = klassifisering(A, B, c, i)
    print(recallandacc(c, truelabel, predictions))

"""
Gosh this was alot >.<
"""
