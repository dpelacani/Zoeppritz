import numpy as np
import cmath as cm

"""
Implementation of the theoretical formulations of reflection and trasmission of plane waves described in SEISMIC WAVE THEORY by Edward S. Krebbes
"""

def snell(theta1, v1, v2):
    "takes theta1 in rad, wave propagating from medium 1 to medium 2"
    theta2 = cm.asin((v2 / v1) * np.sin(theta1))
    return theta2


def zoeppritz(theta1, theta2, phi1, phi2, vp1, vp2, vs1, vs2, p1, p2, Ai, Bi):
    """
    Treats medium 1 as the incidence medium and medium 2 as the transmission
    Solves for amplitudes of the reflected and transmitted P-wave and SV-wave.
    
    Vp Vs and Rho are media parameters, theta1 the independent variable, and the other angles
    are a product of Snell's law. Angles are required to be in radians.
    
    Parameters
    ----------
    theta1: angle of incidence/reflection of P-wave on medium 1
    theta2: angle of transmission of P-wave on medium 2
    phi1: angle of incidence/reflection of SV-wave on medium 1
    phi2: angle of transmission of SV-wave on medium 2
    vp1: P-wave speed propagation on medium 1
    vp2: P-wave speed propagation on medium 2
    vs1: SV-wave speed propagation on medium 1
    vs2: SV-wave speed propagation on medium 2
    rho1: density of medium 1
    rho2: density of medium 2
    Ai: amplitude of incident P-wave
    Bi: amplitude of incident SV-wave
    
    Return
    ------
    Ar: amplitude of reflected P-wave
    At: amplitude of transmitted P-Wave
    Br: amplitude of reflected SV-wave
    Bt: amplitude of transmitted SV-Wave
    
    """
    
    # some useful variables
#     p = np.sin(theta1) / vp1 
#     chi1 = 2 * rho1 * p * (vs1**2)
#     chi2 = 2 * rho2 * p * (vs2**2)
#     gam1 = rho1 * (1 - 2*(vs1*p)**2)
#     gam2 = rho1 * (1 - 2*(vs1*p)**2)
#     z1 = rho1*vp1
#     z2 = rho2*vp2
#     w1 = rho1*vs1
#     w2 = rho2*vs2
    
    # Construct system M@U_rt = N@U_i = b, where U_rt is the vector [Ar, Br, At, Bt] and U_i is the vector [Ai, Bi]
    # See Krebes(2019), Seismic Wave Theory, Chapter 3 and Geldart, Lloyd P., and Robert E. Sheriff. Problems in 
    # exploration seismology and their solutions. Society of Exploration Geophysicists, 2004.
    
#     if vs1==0:
#         M = np.array([
#             [ np.cos(theta1), np.cos(theta2), np.sin(phi2)],
#             [ z1, -z2*np.cos(2*phi2), -w2*np.sin(2*phi2)],
#             [ 0, (vs2/vp2)*np.sin(2*theta2), -np.cos(2*phi2)]
#         ])

#         N = np.array([
#             [np.cos(theta1)],
#             [-z1],
#             [0]
#         ])

#         b = Ai * N
#         U_rt = np.linalg.solve(M, b[:, 0])
#         Ar, At, Bt = np.abs(U_rt)
#         Br = 0.
        
#     else:
#         M = np.array([
#             [ -vp1*p, -np.cos(phi1) , vp2*p , np.cos(phi2) ],
#             [ np.cos(theta1), -vs1*p , np.cos(theta2) , -vs2*p ],
#             [ chi1 * np.cos(theta1) , vs1*gam1 , chi2*np.cos(theta2) , vs2*gam2],
#             [ -vp1*gam1, chi1*np.cos(phi1) ,vp2*gam2 ,-chi2*np.cos(phi2)]
#         ])

#         N = np.array([
#             [vp1*p , np.cos(phi1)],
#             [np.cos(theta1) , -vs1*p],
#             [chi1*np.cos(theta1) , vs1*gam1],
#             [vp1*gam1 , -chi1*np.cos(phi1)]
#         ])

#         U_i = np.array([Ai, Bi])
#         b = N@U_i    
#         U_rt = np.linalg.solve(M, b)
#         Ar, Br, At, Bt = np.abs(np.abs(U_rt))
        
#     print(U_rt, np.sum(U_rt))

    # initialise P and R from the matrix form of the zoeppritz equations
    P = np.array([[-np.sin(theta1), -np.cos(phi1), np.sin(theta2), np.cos(phi2)],
                [np.cos(theta1), -np.sin(phi1), np.cos(theta2), -np.sin(phi2)],
                [2 * p1 * vs1 * np.sin(phi1) * np.cos(theta1), p1 * vs1 *
                (1 - 2 * (np.sin(phi1) ** 2)), 2 * p2 * vs2 * np.sin(phi2) *
                np.cos(theta2), p2 * vs2 * (1 - 2 * (np.sin(phi2) ** 2))],
                [-p1 * vp1 * (1 - 2 * (np.sin(phi1) ** 2)), p1 * vs1 *
                np.sin(2 * phi1), p2 * vp2 * (1 - 2 * (np.sin(phi2) ** 2)),
                -p2 * vs2 * np.sin(2 * phi2)]])

    R = np.array([[np.sin(theta1), np.cos(phi1), -np.sin(theta2), -np.cos(phi2)],
                [np.cos(theta1), -np.sin(phi1), np.cos(theta2), -np.sin(phi2)],
                [2 * p1 * vs1 * np.sin(phi1) * np.cos(theta1), p1 * vs1 *
                (1 - 2 * (np.sin(phi1) ** 2)), 2 * p2 * vs2 * np.sin(phi2) *
                np.cos(theta2), p2 * vs2 * (1 - 2 * (np.sin(phi2) ** 2))],
                [p1 * vp1 * (1 - 2 * (np.sin(phi1) ** 2)), -p1 * vs1 *
                np.sin(2 * phi1), - p2 * vp2 * (1 - 2 * (np.sin(phi2) ** 2)),
                p2 * vs2 * np.sin(2 * phi2)]])

    # invert P and solve for the scattering matrix: Q = P^{-1} R
    Q = np.dot(np.linalg.inv(P), R)
    
    # fix the sign of the imaginary component
    for i in range(4):
        for j in range(4):
            Q[i][j] = complex(Q[i][j].real, -Q[i][j].imag)
            
    Rpp = np.abs(Q[0][0]) # p reflection coeff from incident p
    Rps = np.abs(Q[1][0]) # s reflection coeff from incident p
    Tpp = np.abs(Q[2][0]) # p transmission coeff from incident p
    Tps = np.abs(Q[3][0]) # s transmission coeff from incident p
    
    Rsp = np.abs(Q[0][1]) # p reflection coeff from incident s
    Rss = np.abs(Q[1][1]) # s reflection coeff from incident s
    Tsp = np.abs(Q[2][1]) # p transmission coeff from incident s
    Tss = np.abs(Q[3][1]) # s transmission coeff from incident s
    
    Ar = Ai*Rpp + Bi*Rsp
    At = Ai*Tpp + Bi*Tsp
    Br = Ai*Rps + Bi*Rss
    Bt = Ai*Tps + Bi*Tss
    
    return Ar, At, Br, Bt