import numpy as np
import selection.constraints as C
from scipy.stats import chi
import matplotlib.pyplot as plt
import statsmodels.api as sm

def tangent_space(A, y):

    soln = np.dot(A, y)
    norm_soln = np.linalg.norm(soln)
    eta = np.dot(A.T, soln) / norm_soln

    p, q = A.shape
    tangent_vectors = np.identity(p)[:-1]
    for tv in tangent_vectors:
        tv[:] = tv - np.dot(tv, soln) * soln / norm_soln**2

    return eta, np.dot(tangent_vectors, A)

if __name__ == "__main__":

    def full_sim(L, b, p):
        k, q = L.shape
        A1 = np.random.standard_normal((p,q))
        A2 = L[:p]
        A3 = np.array([np.arange(q)**(i/2.) for i in range(1,4)])

        con = C.constraint((L, b), None)
        
        def sim(A):

            y = C.simulate_from_constraints(con) 
            p = A.shape[0]
            eta, B = tangent_space(A, y)

            newcon = C.constraint((L, b), (B, np.zeros(B.shape[0])))
            newcon = newcon.impose_equality()
            P = np.identity(q) - np.dot(np.linalg.pinv(B), B)
            eta = np.dot(P, eta)
            Vp, V, Vm, sigma = newcon.pivots(eta, y)[:4]

            Vp = max(0, Vp)
            pval = ((chi.sf(Vm/sigma, p) - chi.sf(V/sigma,p)) / 
                    (chi.sf(Vm/sigma, p) - chi.sf(Vp/sigma,p)))

            V2 = np.sqrt((y*np.dot(np.linalg.pinv(A), np.dot(A, y))).sum())
            return np.clip(pval,0,1)

        return sim(A1), sim(A2), sim(A3)

    nsim = 10000
    P = []

    p, q, k = 4, 20, 6
    L, b = np.random.standard_normal((k,q)), np.ones(k) * 0.2

    for _ in range(nsim):
        P.append(full_sim(L, b, p))
    P = np.array(P)

    ecdf = sm.distributions.ECDF(P[:,0])
    ecdf2 = sm.distributions.ECDF(P[:,1])
    ecdf3 = sm.distributions.ECDF(P[:,2])

    plt.clf()
    plt.plot(ecdf.x, ecdf.y, linewidth=4, color='black', label='Fixed (but random) $A$')
    plt.plot(ecdf2.x, ecdf2.y, linewidth=4, color='purple', label='Selected $A$')
    plt.plot(ecdf3.x, ecdf3.y, linewidth=4, color='green', label='Deterministic $A$')

    plt.plot([0,1],[0,1], linewidth=3, linestyle='--', color='red')
    plt.legend(loc='lower right')
    plt.savefig('chisq.pdf')

    # deterministic 

    L2, b2 = np.identity(q)[:4], np.zeros(4)
    P2 = []
    for _ in range(nsim):
        P2.append(full_sim(L2, b2, 3))
    P2 = np.array(P2)

    ecdf = sm.distributions.ECDF(P2[:,0])
    ecdf2 = sm.distributions.ECDF(P2[:,1])
    ecdf3 = sm.distributions.ECDF(P2[:,2])

    plt.clf()
    plt.plot(ecdf.x, ecdf.y, linewidth=4, color='black', label='Fixed (but random) $A$')
    plt.plot(ecdf2.x, ecdf2.y, linewidth=4, color='purple', label='Selected $A$')
    plt.plot(ecdf3.x, ecdf3.y, linewidth=4, color='green', label='Deterministic $A$')

    plt.plot([0,1],[0,1], linewidth=3, linestyle='--', color='red')
    plt.legend(loc='lower right')
    plt.savefig('chisq_det.pdf')

