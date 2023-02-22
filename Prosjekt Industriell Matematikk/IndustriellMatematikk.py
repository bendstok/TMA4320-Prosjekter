import numpy as np
import matplotlib.pyplot as plt

A1 = np.array([[1000, 1], [0, 1], [0, 0]])
A2 = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]])

b1 = np.array([[2], [1], [0]])
b2 = np.array([[0], [0], [1]])
b3 = np.array([[0], [1], [0]])

B = np.concatenate((b1, b2, b3), axis=1)

def A1SVD(A1):
    
    """
    a la text
    """
    
    u, s, vt = np.linalg.svd(A1, full_matrices = False)
    
    print(f"u = {u}")
    print(f"s = {s}")
    print(f"vt = {vt}")
    
    fs = np.diag(s)

    A = u @ fs @ vt
    print(f"A1, u3 vekk = {A}")
    
    
    """
    Den første kolonnen i u er vekk, og likevel får vi rekonstruert A nøyaktig lik den fullt rekonstruerte A.
    """
    
    u, s, vt = np.linalg.svd(A1, full_matrices = False)
    
    u[:,0] = 0
    A = u @ fs @ vt
    print(f"A1 u3 + u1 vekk = {A}")
    
    
    """
    Andre vekk gir et større feil, hvertfall for indeks [2][2], som er nå 10^-6
    """
    
    u, s, vt = np.linalg.svd(A1, full_matrices = False)
    
    u[:,0] = 0
    A = u @ fs @ vt
    print(f"A1 u3 + u1 vekk = {A}")
    
    
    """
    Første vekk gir en matrise som ikke på noen måte har de samme verdiene der verdiene ikke er 0.
    Det betyr at denne første kolonnen er viktigst for å rekonstruere matrisen.
    """

# print(A1SVD(A1))



def A2SVD(A2):
    
    """
    a la text
    """
    
    u, s, vt = np.linalg.svd(A2, full_matrices = False)
    
    print(f"u = {u}")
    print(f"s = {s}")
    print(f"vt = {vt}")
    
    fs = np.diag(s)

    A = u @ fs @ vt
    print(f"A2 = {A}")
    
    u[:,2] = 0
    A = u @ fs @ vt
    print(f"A2, u3 vekk = {A}")
    
    """
    Med å ta vekk u3, ser vi at vi får den nøyaktige samme rekonstruerte A som før
    dermed 
    
    siden u# = [0, 1, 0], viser det at A2 ikke har noen tall ved andre kolonne.
    Dermed kan vi fjerne denne delen ved å ta vekk u[:,3], og de korresponderende delene ved s, og vt,
    og trygt få det samme resultatet: A2 = u_d s_d vt_d
    """
    
# print(A2SVD(A2))

def truncSVD(A, d):
    
    """
    de glorious tex
    """
    
    u, s, vt = np.linalg.svd(A, full_matrices = False)

    # Velger de første d relevante vektorer og verdier
    u = u[:, :d-1]
    fs = np.diag(s[:d-1])
    vt = vt[:d-1]
    
    W = u
    H = fs @ vt
    
    return W, H
    
print(truncSVD(A2, 3))


def orthproj(A, d, B):
    
    """
    Projekterer en matrise på en dictionary (ordbok)
    """
    
    W, H = truncSVD(A, d)
    Wt = np.transpose(W)
    return(W @ Wt @ B)

print(f"A1 W to B: {orthproj(A1, 2, B)}")
print(f"A2 W to B : {orthproj(A2, 2, B)}")

print(f"A1 W to b1 distance: {np.linalg.norm((b1 - orthproj(A1, 2, b1)), 2)}")
print(f"A1 W to b1 distance: {np.linalg.norm((b2 - orthproj(A1, 2, b2)), 2)}")
print(f"A1 W to b1 distance: {np.linalg.norm((b3 - orthproj(A1, 2, b3)), 2)}")
print(f"A2 W to b1 distance: {np.linalg.norm((b1 - orthproj(A2, 2, b1)), 2)}")
print(f"A2 W to b1 distance: {np.linalg.norm((b2 - orthproj(A2, 2, b2)), 2)}")
print(f"A2 W to b1 distance: {np.linalg.norm((b3 - orthproj(A2, 2, b3)), 2)}")

print("intewesting")
