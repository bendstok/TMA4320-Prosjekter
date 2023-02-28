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



print(f"trunctSVD på a2 med d = 2 gir: {truncSVD(A2, 2)}")



def orthproj(A, d, B):
    
    """
    Projekterer en datastt-matrise på en dictionary
    
    ...
    
    input:
    A: datasett-martise
    d: antal vektorer/kolonner/singulærvektorer som skal brukes
    B: test-vektor(er) som projekteres
    
    Output:
    orthproj: En projektert versjon av b til datasettet A
    
    """
    
    #Henter dictionary og vekt
    W = truncSVD(A, d)[0]
    
    #Transponerer W
    Wt = np.transpose(W)
    
    # Projiserer B på A, og returnerer
    orthproj = W @ Wt @ B
    return orthproj



#Printer en projeksjon av b til A1 og A2
print(f"A1 W to B: {orthproj(A1, 2, B)}")
print(f"A2 W to B : {orthproj(A2, 2, B)}")



def distSVD(A, d, B):
    
    """
    Regner ut distanse fra test-vektor(er) B til den projiserte versjonen av B
    
    input:
    A: datasett-martise
    d: antal vektorer/kolonner/singulærvektorer som skal brukes
    B: test-vektor(er) som projekteres
    
    Output:
    dist: Distansen fra test-vektor(er) B til den projuserte versjonen av B
    
    """
    
    #Henter projeksjon av B på A
    proj = orthproj(A, d, B)
    
    # regner ut distansene til B og de projiserte versjonene, og returnerer
    dist = np.zeros(B.shape[1])
    for i in range(B.shape[1]):
        dist[i] = np.linalg.norm((B[:,i] - proj), 2)
    return dist


# Printer alle distansene fra b-vektorene til A1 og A2
d = 2
print(f"A1 W to b2 distance: {distSVD(A1, d, B)}")
print(f"A1 W to b2 distance: {distSVD(A2, d, B)}")



def nnproj(W, A, maxiter, safeDiv, B):
    
    
    """
    W: Dictionaries
    A: Datasett-matrise
    maxiter: Antall ganger en iterasjonsløkke skal skje
    safeDiv: for å unngå nulldivisjon
    B: test-vektor(er) som projekteres
    
    Output:
    En ikke-negativ projektert versjon av b til datasettet A...?
    """
    
    # Tansponerer W
    Wt = np.transpose(W)
    
    # Regner ut Wt@A og Wt@W
    WtB = Wt@B
    WtW = Wt@W
    
    # hager ne tilfeldig H-matrise med riktig forn, med tilfeldig innhold fra 0 til 1
    H = np.random.uniform(0,1,(len(W[0]),len(B[0])))
    
    # Itererer gjennom en konvergensrekke for å få den reelle H, og returnerer
    for i in range(maxiter):
        H = H * WtB / ((WtW @ H) + safeDiv)
    
    # Projekterer b på a med W@H, og returnerer
    P = W@H
    return P


def nndist(W, A, maxiter, safeDiv, B):
    
    """
    Regner ut distanse  fra test-vektor(er) B til den projiserte versjonen av B, 
    
    input:
    W: Dictionaries
    A: Datasett-matrise
    maxiter: Antall ganger en iterasjonsløkke skal skje
    safeDiv: for å unngå nulldivisjon
    B: test-vektor(er) som projekteres
    
    Output:
    dist: Distansen (ikke-negativt) fra test-vektor(er) B til den projuserte versjonen av B
    
    """
    
    #Henter projeksjon av B på A
    P = nnproj(W, A, maxiter, safeDiv, B)
    
    # regner ut distansene til B og de projiserte versjonene, og returnerer
    dist = np.zeros(B.shape[1])
    for i in range(B.shape[1]):
        dist[i] = np.linalg.norm((B[:,i] - P), 2)
    return dist

#Henter Dictionary, of andre tallverdier
maxiter = 50
safeDiv = 10**-10



# Printer P_A1 og P_A2 for å sjekke om de funker:
W = truncSVD(A1, 3)[0]
print(W)
print(b1)
print(b2)
print(b3)
print(nndist(W, A1, maxiter, safeDiv, B))
W = truncSVD(A2, 3)[0]
print(nndist(W, A2, maxiter, safeDiv, B))

def orthproj(A,W):
    Wt = np.transpose(W)
    WtA = Wt@A
    P_wA=W@WtA
    return P_wA

W1,H=truncSVD(A1, d)[0:2]
W2,H=truncSVD(A2, d)[0:2]
#print(orthproj(B,W))

#Oppgave 1c-d:

def nndist(B,W):
    D = np.zeros(len(B[0]))
    P_WB=nnproj(B,W)
#     print(D)
#     print(P_WB)
#     print(B)
    for i in range(len(D)):
#         print(B[:,i])
#         print(P_WB[:,i])
#         print()
        D[i] = np.linalg.norm(B[:,i]-P_WB[:,i])
    return D
#print(dist(B,W))
def nnproj(A,W,maxiter=50,delta=10e-10):
    H = np.random.uniform(0,1,[len(W[0,:]),len(A[0])])
    Wt = np.transpose(W)
    WtA = Wt@A
    WtW = Wt@W
    for k in range(maxiter):
        H = H*WtA/(WtW@H+delta)
    P_WA = W@H
    return P_WA

"""


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
