---
title: 'pyMMF: A Python package for fast mulimode fiber mode solving and simulations'
tags:
  - Python
  - physics
  - fibers
  - wave equation
  - transmission matrix
authors:
  - name: Sébastien M. Popoff
    orcid: 0000-0002-7199-9814
    affiliation: 1

affiliations:
 - name: Institut Langevin, ESPCI Paris, PSL University, CNRS, France
   index: 1

date: 2024
bibliography: paper.bib
---

# Summary

# Statement of need section

# The mathematical problem

We consider the scalar Helmholtz equation:

$$
\left[\Delta_{\perp} +n^2(x,y)k_0^2-\beta^2\right] \ket{\psi} = 0
$$

Finding the modes of the fiber correspond to finding 
all the possible pairs of index profiles $\ket{\psi}$ 
and propagation constants $\beta$ that satisfy this solution.
We are mainly interested in propagating modes, i.e. modes 
for which $\beta$ is real.

# Solvers




## 2D Eigenvalue solver

### Straight fibers

Under the weakly guided approximation, analytical solutions for the mode profiles 
of step-index (SI) and graded-index (GRIN) multimode fibers (MMF) can be found [@okamoto2021fundamentals]. 
It also gives a semi-analytical solution for the dispersion relation in SI MMFs, and, by adding stronger approximations, an analytical solution for the parabolic profile GRIN MMFs [2] (note that those approximations do fail for lower order modes). An arbitrary index profile requires numerical simulations to estimate the mode profiles and the corresponding propagation constants of the modes. I present in this tutorial how to numerically estimate the scalar solution for the profiles and propagation constants of guides modes in multimode circular waveguide with arbitrary index profile and in the presence of bending. I released a beta version of the Python module pyMMF based on such an approach [3]. It relies on expressing the transverse Helmholtz equation as an eigenvalue problem. Solutions are found by finding the eigenvectors of a large but sparse matrix representing the equation on the discretized space.


Finding the modes

We consider the scalar Helmholtz equation:

$$\left[\Delta_{\perp} +n^2(x,y)k_0^2-\beta^2\right] \ket{\psi} = 0$$

Which can be directly expressed as an eigenvalue problem:


$$
\mathbf{A} \ket{\psi} = \beta^2 \ket{\psi} \label{eq:EVP}\\
\text{with } \mathbf{A} = \Delta_{\perp} +n^2(x,y)k_0^2
$$

The solutions are the eigenvalues \(\beta_p^2\) and the corresponding eigenvectors \(\ket{\psi_p}\) are the transverse mode profiles.

Let's consider a discretization of the \((x,y)\) plane with a square mesh so that \(x(m) = m\, dh\) and \(y(n) = n\, dh\). For \(dh\) small enough compared to the wavelength, the transverse part of the Laplacian in the Cartesian coordinate system reads:

\begin{gather}
\Delta_{\perp} = \partial_x^2 + \partial_y^2\\
\text{with} \left[\partial_x^2\ket{\psi}\right]_m \approx \frac{1}{dh^2}\left[\ket{\psi}_{m+1}-2\ket{\psi}_m+\ket{\psi}_{m-1}\right]\\
\text{and} \left[\partial_y^2\ket{\psi}\right]_n \approx \frac{1}{dh^2}\left[\ket{\psi}_{n+1}-2\ket{\psi}_n+\ket{\psi}_{n-1}\right]
\end{gather}

The matrix representation of the operator \(\mathbf{A}\) for a square area of size \(L = N\, dh\) is a \(N^2\) by \(N^2\) sparse matrix with only 5 non-zero diagonals. The main diagonal represents the central part of Laplacian plus the dielectric constant term:

\begin{equation}
\mathbf{A}_{ii} = -\frac{4}{dh^2} + k_0^2n^2(\mathbf{r}_i)
\end{equation}

The unique parameter \(i \in [0,N^2]\) indexes the transverse position. The other non-zero coefficients represent the lateral terms of the Laplacian in the \(x\) and \(y\) direction:

\begin{align}
\mathbf{A}_{i,i+1} &= \frac{1}{dh^2},\quad \text{for } i \not\equiv N-1 \pmod N\\
\mathbf{A}_{i+1,i} &= \frac{1}{dh^2},\quad \text{for } i \not\equiv N-1 \pmod N\\
\mathbf{A}_{i,i+N} &= \frac{1}{dh^2},\quad \text{for } i \in [0,N(N-1)]\\
\mathbf{A}_{i+N,i} &= \frac{1}{dh^2},\quad \text{for } i \in [0,N(N-1)]
\end{align}

These four contributions has exactly \(N(N-1)\) non-zero elements, corresponding to close boundary conditions that force the points on the edge of the square window to have a missing neighbor. Close boundary conditions may create artificial unwanted reflections. However, as we are only interested here in the guided modes, which exponentially decay as a function of \(r\) outside the fiber core, boundary conditions at the edge of the window would not affect the results as long as the window size \(L\) is sufficiently large compared to the core size.

The matrix \(\mathbf{A}\) can be very large but it is also very sparse and Hermitian, allowing numerical tools to efficiently find the eigenvalues and the eigenvectors with reasonable computing resources. In Python, we use the function scipy.sparse.eigh from the Scipy module [3]. To select only the propagating mode we only keep the solution for which the following condition is met:


\begin{equation}
\beta > \beta_{min}
\end{equation}

with \(\beta_{min} = k_0 n_{min}\) and \(n_{min}\) being the optical index in the cladding. When this condition is not satisfied, the mode propagates both in the cladding and in the core and is thus not confined in the core. As we are interested here only in the propagating modes, these modes are rejected. If we wanted to accurately account for them, more attention should be paid to the boundary conditions as their transverse mode profile would extend away from the core and would be sensitive to the edges of the observation window.


# Acknowledgement

# References