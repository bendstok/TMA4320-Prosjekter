#Oppgave 1a

#importerer biblioteker
import numpy as np
import matplotlib.pyplot as plt
import os 

#Skriver inn test-datasetter A1 og A2, hver med kolonner som datapunkter
A1 = np.array([[1000, 1], [0, 1], [0, 0]])
A2 = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]])

#Skriver inn test-vektorer, og setter det sammen til en matrise
b1 = np.array([[2], [1], [0]])
b2 = np.array([[0], [0], [1]])
b3 = np.array([[0], [1], [0]])
B = np.concatenate((b1, b2, b3), axis=1)

def A1SVD(A1):
    
    """
    Regner ut SVD til A1, sjekker om de rekonstrueres til A, og sjekker hviken kolonne som er viktigst
    
    ...
    
    input:
    A1; datasett A1
    
    Output:
    Printer ut A1 med henholdsvis full rekonstruering (uten kolonne 3), og med kolonne, 2, og 1 vekk
    """
    
    # Henter svd
    U, S, Vt = np.linalg.svd(A1, full_matrices = False)
    
    # Printer ut svd-matrisene
    print(f"U = {U}")
    print("Formen på U: ", U.shape)
    print(f"S = {S}")
    print("Formen på S: ", S.shape)
    print(f"Vt = {Vt}")
    print("Formen på Vt: ", Vt.shape)
    
    # Gjør diagonalvektoren til en matrise
    FS = np.diag(S)

    # Rekonstruerer, og printer
    A = U @ FS @ Vt
    print(f"A1, kolonne 3 vekk = {A}")
    print("Den første kolonnen i u er vekk, og likevel får vi rekonstruert A nøyaktig lik den fullt rekonstruerte A")
    
    
    
    # Henter svd
    U, S, Vt = np.linalg.svd(A1, full_matrices = False)
    
    # Fjerner kolonne 2
    U[:,1] = 0
    
    # Rekonstruerer, og printer
    A = U @ FS @ Vt
    print(f"A1 kolonne 3 + kolonne 1 vekk = {A}")
    print("Andre vekk gir et større feil, hvertfall for indeks [2][2], som er nå 10^-6")
    
    
    
     # Henter svd
    U, S, Vt = np.linalg.svd(A1, full_matrices = False)
    
    # Fjerner kolonne 2
    U[:,0] = 0
    
    # Rekonstruerer, og printer
    A = U @ FS @ Vt
    print(f"A1 kolonne 3 + kolonne 1 vekk = {A}")
    print("Første vekk gir en matrise som ikke på noen måte har de samme verdiene der verdiene ikke er 0.")
    print("Det betyr at denne første kolonnen er viktigst for å rekonstruere matrisen")


    
print(A1SVD(A1))


#Oppgave 1b

def A2SVD(A2):
    
    """
    Regner ut SVD til A2, printer ut SVD-matrisene, sjekker om de rekonstrueres til A, og sjekker hviken kolonne som er viktigst
    
    ...
    
    input:
    A2: datasett A2
    
    Output:
    Printer ut SVD-matriser, og A1 med henholdsvis full rekonstruering (uten kolonne 3), og med kolonne, 2, og 1 vekk
    """
    
    # Henter svd
    U, S, Vt = np.linalg.svd(A2, full_matrices = False)
    
    # Printer ut svd-matrisene
    print(f"U = {U}")
    print("Formen på U: ", U.shape)
    print(f"S = {S}")
    print("Formen på S: ", S.shape)
    print(f"Vt = {Vt}")
    print("Formen på Vt: ", Vt.shape)
    
    # Gjør diagonalvektoren til en matrise
    FS = np.diag(S)

    # Rekonstruerer, og printer
    A = U @ FS @ Vt
    print(f"A2 = {A}")
    
    # Fjerner kolonne 3
    U[:,2] = 0

    # Rekonstruerer, og printer
    A = U @ FS @ Vt
    print(f"A2, u3 vekk = {A}")
    print("Med å ta vekk u3, ser vi at vi får den nøyaktige samme rekonstruerte A som før") 
    print("siden u# = [0, 1, 0], viser det at A2 ikke har noen tall ved andre kolonne.")
    print("Dermed kan vi fjerne denne delen ved å ta vekk u[:,3], og de korresponderende delene ved s, og vt,")
    print("og trygt få det samme resultatet: A2 = u_d s_d vt_d")



print(A2SVD(A2))

def truncSVD(A, d):
    
    """
    Gjør et svd med de d viktigste leddene i matrisen, altså et trunktert versjon av vanlig SVD-regning,
    
    ...
    
    input:
    A: datasett-martise
    d: antal vektorer/kolonner/singulærvektorer som skal brukes
    
    Output:
    
    W: Dictionaries
    H: Vekt på dictionarien
    """
    
    # Henter svd
    U, S, Vt = np.linalg.svd(A, full_matrices = False)

    # Velger de første d relevante vektorer og verdier
    U = U[:, :d]
    S = S[:d]
    FS = np.diag(S)
    Vt = Vt[:d]
    
    # setter inn dictionary og vekt, og returnerer
    W = U
    H = FS @ Vt
    return W, H, S, FS, Vt


#Oppgave 1c

def orthproj(W,A):
    
    """
    Tar inn et dictionary med ortogonale kolonner W og et datasett A og prosjekterer A på W.
    
    input:
    W: Dictionary med ortogonale kolonner
    A: datasett-martise
    
    Output:
    orthproj: En projektert versjon av A på W
    """
    Wt = np.transpose(W)
    orthproj = W@Wt@A
    return orthproj


def ortdist(W,A):
    """
    Regner ut kolonnevis avstand fra matrise A til dictionary W.
    
    input:
    W: Dictionary med ortogonale kolonner
    A: Datasett-martise
    
    Output:
    dist: Distanse fra A til W.
    """
    dist = np.zeros(len(A[0]))
    proj=orthproj(W,A)
    for i in range(len(dist)):
        dist[i] = np.linalg.norm(A[:,i]-proj[:,i])
    return dist


#Oppgave 1d

def nnproj(W,A,maxiter=50,safeDiv=10e-10):
    """
    Tar inn et ikke-negativ dictionary W og matrise A og returnerer den ikke negative projeksjonen av A på W.
    
    input:
    W: Ikke-negativ dictionary
    A: Datasett-martise
    maxiter: Antall iterasjoner brukt for å regne ut den ikke-negative vekt matrisen H
    safeDiv: Konstant ledd i divisor for å unngå null-divisjon
    
    Output:
    proj: Den ikke-negative projeksjonen av A på W.
    """
    H = np.random.uniform(0,1,[len(W[0,:]),len(A[0])])
    Wt = np.transpose(W)
    WtA = Wt@A
    WtW = Wt@W
    for k in range(maxiter):
        H = H*WtA/(WtW@H+safeDiv)
    proj = W@H
    return proj


def nndist(W,A):
    """
    Regner ut kolonnevis avstand fra ikke-negative matrise A til ikke-negativ dictionary W.
    
    input:
    W: Ikke-negativ dictionary
    A: Ikke-negativ datasett-martise
    
    Output:
    dist: Distanse fra A til W.
    """
    dist = np.zeros(len(A[0]))
    proj = nnproj(W,A)
    for i in range(len(dist)):
        dist[i] = np.linalg.norm(A[:,i]-proj[:,i])
    return dist

"Tester de forskjellige distanse funksjonene"
print(ortdist(W1,B))
print(nndist(A1,B))


"""OPPGAVE 2"""

#a)


# Load the data and rescale

# Finner 'veien' til der python filen er på din maskin, for å finne
# train og test filene
# Altså, må train og test filene være i samme mappe som denne fila
dir_path = os.path.dirname(os.path.realpath(__file__))

train = np.load(dir_path + '/train.npy')/255.0
test = np.load(dir_path + '/test.npy')/255.0

def plotimgs(imgs, nplot = 4):
    """
    Plots the nplot*nplot first images in imgs on an nplot x nplot grid. 
    Assumes heigth = width, and that the images are stored columnwise
    input:
        imgs: (height*width,N) array containing images, where N > nplot**2
        nplot: integer, nplot**2 images will be plotted
    """

    n = imgs.shape[1]
    m = int(np.sqrt(imgs.shape[0]))

    assert(n >= nplot**2), "Need amount of data in matrix N >= nplot**2"

    # Initialize subplots
    fig, axes = plt.subplots(nplot,nplot)

    # Set background color
    plt.gcf().set_facecolor("lightgray")

    # Iterate over images
    for idx in range(nplot**2):

        # Break if we go out of bounds of the array
        if idx >= n:
            break

        # Indices
        i = idx//nplot; j = idx%nplot

        # Remove axis
        axes[i,j].axis('off')

        axes[i,j].imshow(imgs[:,idx].reshape((m,m)), cmap = "gray")
    
    # Plot

    fig.tight_layout()
    plt.show()

plotimgs(train[:,1,:], nplot=4)


"2b"


n = 1000 # Antall datapunkter
c = 0 # Klasse
d = 500 # 16 viktigste kolonner

A = train[:,c,:n]
W, H, S, FS, Vt = truncSVD(A, d)

plotimgs(W, nplot = 4)

plt.semilogy(S)


"interesting"
"Her, kommenter om resultatet vi fill nedenfor"
"ikke matchende som over? mulig fordi de første ikke er de beste?" "probably"

"Den går raskt ned, og går veldig sakte nedover. med en høyere d, ser vi at den begynner å raskt gå ned igjen"
"Dette viser oss en funksjon som har mange gjennomsnittelige, noen få gode, og noen får dårlige matriser"
"Funnksjon? nei ikke helt"
