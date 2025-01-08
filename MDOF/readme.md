# Experimentelle Modalanalyse (EMA) | Prof. J. Grabow
## MDOF (Systeme mit einem Freiheitsgrad f>1) und Rayleigh-Dämpfung
### modale Entkopplung


```python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:13:41 2023

Program history
01.09.2023    V. 1.0    Start
09.10.2023    V. 1.1    Anfangsbedingungen
10.10.2023    V  1.2    Lösung des entkoppelten Systems im Zeitbereich
16.11.2023    V  1.3    Frequenzgang
23.11.2023    V  1.4    Dokumentation
31.12.2024    V  1.5    Scale y-Achse 
08.01.2025    V  1.6    Rayleigh-Quotient eingefügt

@author: Prof. Jörg Grabow (grabow@amesys.de)
"""
__version__ = '1.6'
__author__ = 'Joe Grabow'

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import *
from sympy.physics.mechanics import dynamicsymbols, mlatex
from IPython.display import Math, Latex
from matplotlib.widgets import MultiCursor
```

### Hilfsfunktion normiere_matrix(matrix)
Normiert in einer Matrix in jedem Spaltenvektor den betragsmäßig größten Wert auf eins (Darstellungsnorm).


```python
def normiere_matrix(matrix):
    """normiert in einer Matrix in jedem Spaltenvektor den betragsmäßig größten
    Wert auf eins"""
    max_abs_werte = np.max(np.abs(matrix), axis=0)
    normierte_matrix = matrix / max_abs_werte
    return normierte_matrix
```

### Hilfsfunktion zur Berechnung des komplexen Frequenzganges


```python
def frf(ordnung,i,k,omega_e):
    # Estimate the frequency-response functions of the model.
    # erzeuge komplexen Null-Vektor in Länge des Vektors der Erregerfrequenz
    f_vektor = np.zeros(len(omega_e), dtype=complex)
    for n in range(ordnung):
        S_1 = (Residuum[n] * phi[i,n] * phi[k,n]) / \
            ((0+1j) * omega_e - eigenwert_k[n])
        S_2 = (np.conj(Residuum[n]) * phi[i,n] * phi[k,n]) / \
            ((0+1j) * omega_e - np.conj(eigenwert_k[n] ))
        f_vektor = f_vektor + S_1 + S_2
    return f_vektor
```

## Definition der Systemparameter
Alle Systemparameter wie Massen, Steifigkeiten oder Dämpfungen werden in SI-Einheiten angegeben.


```python
# Eingabe der Systemmatrizen M,C,K
m1, m2, m3 = (60e3, 60e3, 15e3)
c1, c2, c3 = (30e6, 30e6, 10e6)
k1, k2, k3 = (100, 250, 80)
alpha, beta = 0.9, 1e-4
```

## Definition der Systemmatrizen
Eingabe der Massenmatrix, Steifigkeitsmatrix und Dämpfungsmatrix

### Eingabe der Matrizen und Anfangsbedingungen
---
- *M - Massenmatrix als Array*
- *C - Steifigkeitsmatrix als Array*
- *K - Dämfungsmatrix als Array*
---
Bei proportionaler Dämpfung (Rayleigh-Dämpfung): $K = \alpha M + \beta C$.

<div class="alert alert-block alert-info">
<b>Systemmatritzen:</b> 
</div>


```python
M = np.array([[m1, 0, 0],
              [0, m2, 0],
              [0, 0, m3]])
C = np.array([[c1+c2, -c2,   0],
              [-c2, c2+c3, -c3],
              [0, -c3,  c3]])
K = np.array([[k1+k2, -k2,   0],
              [-k2, k2+k3, -k3],
              [0, -k3,  k3]])
# nur bei Rayleigh-Dämpfung
K = alpha*M + beta*C
```

### Massenmatrix


```python
Math(rf'M = {sym.latex(Matrix(M))}')
```




$\displaystyle M = \left[\begin{matrix}60000.0 & 0 & 0\\0 & 60000.0 & 0\\0 & 0 & 15000.0\end{matrix}\right]$



### Steifigkeitsmatrix


```python
Math(rf'C = {sym.latex(Matrix(C))}')
```




$\displaystyle C = \left[\begin{matrix}60000000.0 & -30000000.0 & 0\\-30000000.0 & 40000000.0 & -10000000.0\\0 & -10000000.0 & 10000000.0\end{matrix}\right]$



### Dämpfungsmatrix


```python
Math(rf'K = {sym.latex(Matrix(K))}')
```




$\displaystyle K = \left[\begin{matrix}60000.0 & -3000.0 & 0\\-3000.0 & 58000.0 & -1000.0\\0 & -1000.0 & 14500.0\end{matrix}\right]$




```python
# verallgemeinerte Koordinaten q(t)
q1, q2, q3 = dynamicsymbols('q1 q2 q3')
Q = sym.Matrix([[q1],[q2],[q3]])

# Erregerkräfte f(t)
f1, f2, f3 = dynamicsymbols('f1 f2 f3')
F = sym.Matrix([[f1],[f2],[f3]])

# Zeitableitungen der verallgemeinerten Koordinaten q(t)
t = sym.Symbol('t')
Qd = Q.diff(t,1)
Qdd = Q.diff(t,2)
```

### Dgl.-System


```python
display(Math(sym.latex(Matrix(M))+mlatex(Qdd)+'+'+sym.latex(Matrix(C))+mlatex(Qd)+sym.latex(Matrix(K))+mlatex(Q)+'='+mlatex(F)))
```


$\displaystyle \left[\begin{matrix}60000.0 & 0 & 0\\0 & 60000.0 & 0\\0 & 0 & 15000.0\end{matrix}\right]\left[\begin{matrix}\ddot{q}_{1}\\\ddot{q}_{2}\\\ddot{q}_{3}\end{matrix}\right]+\left[\begin{matrix}60000000.0 & -30000000.0 & 0\\-30000000.0 & 40000000.0 & -10000000.0\\0 & -10000000.0 & 10000000.0\end{matrix}\right]\left[\begin{matrix}\dot{q}_{1}\\\dot{q}_{2}\\\dot{q}_{3}\end{matrix}\right]\left[\begin{matrix}60000.0 & -3000.0 & 0\\-3000.0 & 58000.0 & -1000.0\\0 & -1000.0 & 14500.0\end{matrix}\right]\left[\begin{matrix}q_{1}\\q_{2}\\q_{3}\end{matrix}\right]=\left[\begin{matrix}f_{1}\\f_{2}\\f_{3}\end{matrix}\right]$


 ### Anfangsbedingungen $q_{o}$ und $v_{0}$


```python
# Anfangsbedingungen
q_0 = np.array([1e-3, -2e-3, 0.5e-3])  # Anfangsweg in mm
v_0 = np.array([0.5e-3, 1e-3, -0.5e-3])  # Anfangsgeschwindigkeit in mm/s
```

## Berechnungen zur Entkopplung des Dgl.-Systems
### Die Systemmatrix A: 

$A = M^{-1} \cdot C$


```python
A = np.matmul(np.linalg.inv(M),C)
```

### Das spezielle Eigenwertproblem
$\left( A - \lambda \cdot E \right) \cdot \psi = 0$ 

$det \left( A - \lambda E \right) = 0$


```python
eigenwerte, eigenvektoren = np.linalg.eig(A)

# Sortiere die Eigenwerte und -vektoren nach den Eigenwerten
sort_indices = np.argsort(eigenwerte)[::-1]
sortierte_eigenwerte = eigenwerte[sort_indices]
sortierte_eigenvektoren = eigenvektoren[:, sort_indices]
```

Eigenkreisfrequenzen (ungedämpft) sortiert nöch Größe:

$\omega _0 = \sqrt{ \lambda }$


```python
wn = np.sqrt(sortierte_eigenwerte)
```

<div class="alert alert-block alert-success">
<b>Eigenwerte:</b> sortiert nach Größe
</div>


```python
wn
```




    array([37.62931365, 27.62475937, 12.41937023])



zugehörige Eigenvektoren $\psi$ und die Eigenvektormatrix $\Psi$

<div class="alert alert-block alert-success">
<b>Eigenvektormatrix:</b>
</div>


```python
sortierte_eigenvektoren
```




    array([[ 0.66815554, -0.28934219, -0.3389505 ],
           [-0.55585897, -0.13707452, -0.57334103],
           [ 0.49455938,  0.94736037, -0.7459173 ]])




```python
ordnung = len(wn)
```

### Rayleigh-Quotient
Berechnung des betragsmäßig  größten Eigenwertes

$\lambda_i = \frac{\mathbf{x}_i^\top \mathbf{A} \mathbf{x}_i}{\mathbf{x}_i^\top \mathbf{x}_i}, \quad \text{für } i = 1, 2, \dots, n
\$


```python
lambdas = []
for i in range(sortierte_eigenvektoren.shape[1]):  # Über alle Spalten der Eigenvektoren iterieren
    x = sortierte_eigenvektoren[:, i]
    numerator = np.dot(x.T, np.dot(A, x))  # x^T * A * x
    denominator = np.dot(x.T, x)  # x^T * x
    lambdas.append(numerator / denominator)
lambda_R1, lambda_R2, lambda_R3 = lambdas
```

<div class="alert alert-block alert-success">
<b>Eigenwerte:</b> über Rayleigh-Quotient
</div>


```python
lambdas
```




    [1415.9652458835767, 763.1273304182796, 154.24075703147724]



### Normierung der Eigenvektoren
In jedem Eigenvektor wird der betragsmäßig größte Wert immer auf eins normiert.

normierte Eigenvektoren $\psi_{N}$

<div class="alert alert-block alert-success">
<b>Eigenvektormatrix:</b> Min|Max normiert auf 1 / -1 
</div>


```python
psi = normiere_matrix(sortierte_eigenvektoren)
psi
```




    array([[ 1.        , -0.30541935, -0.45440761],
           [-0.83193049, -0.144691  , -0.76863886],
           [ 0.74018601,  1.        , -1.        ]])



### Transformation der Koordinate q(t) ind den Modalraum p(t)

$ \bar{q}=\Psi \cdot \bar{p} $

$ M\Psi \ddot{\bar{p}} + K\Psi \dot{\bar{p}} + C\Psi \bar{p} = \bar{f} $

$ \Psi ^{T} M\Psi \ddot{\bar{p}} + \Psi ^{^{T}} K\Psi \dot{\bar{p}} + \Psi ^{T} C\Psi \bar{p} = \Psi ^{T} \bar{f} $

$ M_{G}\ \ddot{\bar{p}} + K_{G}\ \dot{\bar{p}} + C_{G}\ \bar{p} = \Psi ^{T} \bar{f} $

### Die generalisierte Massenmatrix $M_G$ im Modalraum
generalisierte Massenmatrix:

$M_G = \Psi ^T M \Psi$


```python
psiT = np.transpose(psi)
MG = np.dot(np.dot(psiT,M),psi)
```

<div class="alert alert-block alert-success">
<b>generalisierte Masse:</b>
</div>


```python
np.around(MG,2)
```




    array([[109744.63,     -0.  ,     -0.  ],
           [    -0.  ,  21852.99,     -0.  ],
           [    -0.  ,     -0.  ,  62837.52]])



### Die Modalmatrix $\phi$

$ \Phi^{T} M \Phi \ \ddot{\bar{p}} + \Phi ^{^{T}} K\Phi\ \dot{\bar{p}} + \Phi ^{T} C\Phi \ \bar{p} = \Phi ^{T} \bar{f} $

$ E \ddot{\bar{p}} + K_{m} \dot{\bar{p}} + \Omega  \bar{p} = \Phi ^{T} \bar{f} $

Normierung der Eigenvektoren $\psi \text{  auf  } \phi$

$\phi ^{(i)} = \frac{\psi ^{(i)}} { \sqrt{{M_G}_{i,i} } }$


```python
phi = psi/(np.sqrt(np.diag(MG)))
```

Modalmatrix $\Phi$

<div class="alert alert-block alert-success">
<b>Modalmatrix:</b>
</div>


```python
phi
```




    array([[ 0.00301862, -0.00206605, -0.00181274],
           [-0.00251128, -0.00097878, -0.00306629],
           [ 0.00223434,  0.00676464, -0.00398924]])



Proberechnung

$\Phi^{T} \cdot M \cdot \Phi = E$


```python
E = np.dot(np.dot(np.transpose(phi),M),phi)
np.around(E,2)
```




    array([[ 1., -0., -0.],
           [-0.,  1., -0.],
           [-0., -0.,  1.]])



### Die modale Dämpfungsmatrix $K_m$
$K_m =\Phi^T \cdot K \cdot \Phi$



```python
Km = np.dot(np.dot(np.transpose(phi),K),phi)
```

<div class="alert alert-block alert-success">
<b>modale Dämpfung:</b>
</div>


```python
np.around(Km,4)
```




    array([[ 1.0416, -0.    , -0.    ],
           [-0.    ,  0.9763, -0.    ],
           [-0.    , -0.    ,  0.9154]])



### Die Spektralmatrix $ \Omega $
$ \Omega =\Phi^T \cdot C \cdot \Phi $


```python
CSp = np.dot(np.dot(np.transpose(phi),C),phi)
```

<div class="alert alert-block alert-success">
<b>Spektralmatrix:</b>
</div>


```python
np.around(CSp,4)
```




    array([[1415.9652,   -0.    ,   -0.    ],
           [  -0.    ,  763.1273,   -0.    ],
           [  -0.    ,   -0.    ,  154.2408]])



### Entkoppeltes Dgl.-System im Modalraum

$1\cdot \ddot{p_{1}}+2 D_{1} \omega _{01} \ \dot{p_{1}} + \omega _{01}^{2} \ p{_{1}} =\Phi _{11} f_{1} + \Phi _{21} f_{2} +\Phi _{31} f_{3} $

$ 1\cdot \ddot{p_{2}}+2 D_{2} \omega _{02} \ \dot{p_{2}} + \omega _{02}^{2} \ p{_{2}} =\Phi _{12} f_{1} + \Phi _{22} f_{2} +\Phi _{32} f_{3} $

$ 1\cdot \ddot{p_{3}}+2 D_{3} \omega _{03} \ \dot{p_{3}} + \omega _{03}^{2} \ p{_{3}} =\Phi _{13} f_{1} + \Phi _{23} f_{2} +\Phi _{33} f_{3} $

### Die Lehrschen Dämpfungsmaße D
$D_i = \frac{ {K_m}_{i,i} }{2 \cdot {\omega_0}_i  }$


```python
D = np.diag(Km) / (2*wn)
```

<div class="alert alert-block alert-success">
<b>Dämpfungsmaße:</b>
</div>


```python
D
```




    array([0.01384023, 0.01767097, 0.03685469])



### Eigenkreisfrequenzen (gedämpft) $\omega$
$\omega = \omega_0 \cdot \sqrt{ 1-D^2 }$



```python
wd = wn * np.sqrt(1-D*D)
```

<div class="alert alert-block alert-success">
<b>Eigenkreisfrequenzen:</b> gedämpft
</div>


```python
wd
```




    array([37.6257095 , 27.62044594, 12.41093295])



### Abklingkonstanten $\delta$ 
$\delta = D \cdot \omega_0$


```python
delta = D * wd
```

<div class="alert alert-block alert-success">
<b>Abklingkonstanten:</b>
</div>


```python
delta
```




    array([0.52074838, 0.48808014, 0.45740108])



### Darstellung der Eigenvektoren als Knotenbilder(mode shapes)


```python
x = np.arange(ordnung+1)
x = np.array([0, 3, 6, 8])  # Orte der darzustellenden Eigenvektoren
psi_1 = psi[:,0]
psi_2 = psi[:,1]
psi_3 = psi[:,2]
nullvektor = np.zeros(1)
y1 = np.insert(psi_1,0,0)
y2 = np.insert(psi_2,0,0)
y3 = np.insert(psi_3,0,0)

# Plot Knotenbild (mode shape)
fig, axs = plt.subplots(3, 1,figsize=(10,10))
fig.suptitle('Knotenbilder der Eigenvektoren')

axs[0].plot(x,y3, 'o-', color='cornflowerblue',label="psi 3 (Grundwelle)")
axs[0].legend(loc=2)
axs[0].set_xlabel('Höhe in m')
axs[0].set_ylabel('Eigenvektor')
axs[0].plot(x, -y3,'o-', color='cornflowerblue')
axs[0].stem(x,y3, linefmt='red', basefmt='lightgray')
axs[0].stem(x,-y3, linefmt='red', basefmt='lightgray')

axs[1].plot(x,y2, 'o-', color='cornflowerblue',label="psi 2 (1. Oberwelle)")
axs[1].legend(loc=2)
axs[1].set_xlabel('Höhe in m')
axs[1].set_ylabel('Eigenvektor')
axs[1].plot(x,-y2,'o-', color='cornflowerblue')
axs[1].stem(x,y2, linefmt='red', basefmt='lightgray')
axs[1].stem(x,-y2, linefmt='red', basefmt='lightgray')

axs[2].plot(x,y1, 'o-', color='cornflowerblue',label="psi 1 (2. Oberwelle)")
axs[2].legend(loc=2)
axs[2].set_xlabel('Höhe in m')
axs[2].set_ylabel('Eigenvektor')
axs[2].plot(x,-y1,'o-', color='cornflowerblue')
axs[2].stem(x,y1, linefmt='red', basefmt='lightgray')
axs[2].stem(x,-y1, linefmt='red', basefmt='lightgray')

plt.show()
```


    
![png](output_76_0.png)
    


### Lösung der entkoppelten Gleichungen im Zeitbereich 


```python
# Lösung im Zeitbereich über modale Entkopplung

# modale Transformation der Anfangsbedingungen
p_0 = np.dot(np.linalg.inv(phi), q_0)  # Anfangsweg im Modalraum
v_0p = np.dot(np.linalg.inv(phi), v_0)  # Anfangsgeschwindigkeit im Modalraum
C_1 = p_0  # modale Integrationskonstante C1
C_2 = (v_0p + delta * C_1) / wd  # modale Integrationskonstante C2

t = np.arange(0, 5, 0.01)  # Zeitreife für Darstellung

# Schwingwege im Modalraum
p_1 = np.exp(-delta[0] * t) * (C_1[0] * np.cos(wn[0] * t) + C_2[0] * np.sin(wn[0] * t))
p_2 = np.exp(-delta[1] * t) * (C_1[1] * np.cos(wn[1] * t) + C_2[1] * np.sin(wn[1] * t))
p_3 = np.exp(-delta[2] * t) * (C_1[2] * np.cos(wn[2] * t) + C_2[2] * np.sin(wn[2] * t))

# Rücktransformation durch Superposition in den realen physikalischen Raum
q_1 = phi[0,0] * p_1 + phi[0,1] * p_2 + phi[0,2] * p_3
q_2 = phi[1,0] * p_1 + phi[1,1] * p_2 + phi[1,2] * p_3
q_3 = phi[2,0] * p_1 + phi[2,1] * p_2 + phi[2,2] * p_3

# Lösungen im Zeitbereich grafisch darstellen
fig, axs = plt.subplots(3, 1,figsize=(10,10))
fig.suptitle('Schwingwege der Koordinaten q1 - q3')

axs[0].plot(t, q_1, '-', color='cornflowerblue',label="q1")
axs[0].legend(loc='upper right')
axs[0].set_xlabel('Zeit in Sekunden')
axs[0].set_ylabel('Weg in m')

axs[1].plot(t, q_2, '-', color='cornflowerblue',label="q2")
axs[1].legend(loc='upper right')
axs[1].set_xlabel('Zeit in Sekunden')
axs[1].set_ylabel('Weg in m')

axs[2].plot(t, q_3, '-', color='cornflowerblue',label="q3")
axs[2].legend(loc='upper right')
axs[2].set_xlabel('Zeit in Sekunden')
axs[2].set_ylabel('Weg in m')

# Zeige ein Gitternetz im Hintergrund an
for ax in axs:
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

# Markierung der Anfangswege
axs[0].scatter(0,q_0[0], color='red', zorder=5)
axs[1].scatter(0,q_0[1], color='red', zorder=5)
axs[2].scatter(0,q_0[2], color='red', zorder=5)

plt.show()
```


    
![png](output_78_0.png)
    


### Frequenzgangmatrix berechnen und darstellen


```python
# Frequenzgangmatrix

# Residuum berechnen
Residuum = np.zeros(ordnung, dtype=complex) # erzeuge komplexen Null-Vektor
for i in range(ordnung):
    Residuum[i] = -1j / (2 * wd[i] )

# komplexe Eigenwerte berechnen
eigenwert_k = -delta + 1j * wd

# Erregerkreisfrequenz
omega_e = np.arange(0, 60, 0.1)

# Wahl des Frequenzgangmatrixelementes zur Berechnung und Darstellung
out_frf = 0
in_frf = 0

frequenzgang = frf(ordnung, out_frf, in_frf, omega_e)
amplitude = np.abs(frequenzgang)  # Amplitude
phase_radian = np.angle(frequenzgang)
phase_degree = np.degrees(phase_radian)

# Plot Amplitudenfrequenzgang
fig, axs = plt.subplots(2, 1,figsize=(10,7))
fig.suptitle('Amplitudenfrequenzgang')
axs[0].set_yscale("log")
axs[0].set_xlabel('Erregerkreisfrequenz in rad/s')
axs[0].set_ylabel('Amplitude in Meter')
axs[0].plot(omega_e, amplitude, '-', color='cornflowerblue',
            label='FRF '+ str(out_frf) + str(in_frf))

# Für logarithmischen Maßstab nur positive Werte berücksichtigen
positive_amplitude = [amp for amp in amplitude if amp > 0]
y_min, y_max = min(positive_amplitude), max(positive_amplitude)
axs[0].set_ylim([y_min * 0.9, y_max * 1.1])

axs[0].legend(loc='upper right')

#axs[1].set_ylim(-180)
axs[1].set_xlabel('Erregerkreisfrequenz in rad/s')
axs[1].set_ylabel('Phase in Grad')
axs[1].plot(omega_e, phase_degree, '-', color='cornflowerblue')

multi = MultiCursor(fig.canvas, (axs[0], axs[1]), color='g', lw=2,
                    horizOn=True, vertOn=True)
# Zeige ein Gitternetz im Hintergrund an
for ax in axs:
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

plt.show()
```


    
![png](output_80_0.png)
    



```python

```
