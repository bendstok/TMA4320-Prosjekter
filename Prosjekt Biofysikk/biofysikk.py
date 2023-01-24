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
N_virrevandrere = np.zeros((N, M))
for i in range(N):
    N_virrevandrere[i] = virrevandring(M, høyreSannsynlighet, randomnums, dx, dt)[0]
print(N_virrevandrere)
