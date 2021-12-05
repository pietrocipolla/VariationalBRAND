import numpy as np
import scipy.sparse as sp
from scipy.special import psi, gammaln
import sys

#EXAMPLE CODE
#-load data
#-call cavi
#-print data

#CAVI
#-init parameter
#-for
# —update parameter
# —elbo

def load_ap_data(ap_data, ap_vocab):
    n = len(open(ap_data).readlines())
    m = len(open(ap_vocab).readlines())

    X = sp.lil_matrix((n,m))

    for i, line in enumerate(open(ap_data)):
        words = line.split()
        idxs = []
        vals = []
        for w in words[1:]:
            idx, val = map(int, w.split(':'))
            idxs.append(idx)
            vals.append(val)
        
        X[i,idxs] = vals

    X = X.tocsr()
    return X

def load_bow_data(bow_data):
    f = open(bow_data)
    n = int(f.readline().strip())
    m = int(f.readline().strip())
    _ = f.readline()

    d = np.array([map(int, line.split()) for line in f])

    X = sp.csr_matrix((d[:,2], d[:,:2].T - 1), shape=(n,m))
    return X

def logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis)
    try:
        return a_max + np.log(np.sum(np.exp(a - a_max), axis=axis))
    except:
        return a_max + np.log(np.sum(np.exp(a - a_max[:,np.newaxis]), axis=axis))

def var_dpmm_multinomial(X, alpha, base_dirichlet, T=50, n_iter=100, Xtest=None):
    '''
    runs variational inference on a DP mixture model where each
    mixture component is a multinomial distribution.

    X: observed data, (N,M) matrix, can be sparse
    alpha: concentration parameter
    base_dirichlet: base measure (Dirichlet (1,M) in this case)
    '''
    N, M = X.shape

    # variational multinomial parameters for z_n
    phi = np.matrix(np.random.uniform(size=(T,N)))
    phi = np.divide(phi, np.sum(phi, axis=0))

    # variational beta parameters for V_t
    gamma1 = np.matrix(np.zeros((T-1,1)))
    gamma2 = np.matrix(np.zeros((T-1,1)))

    # variational dirichlet parameters for \eta_t
    tau = np.matrix(np.zeros((T,M)))

    ll = []
    held_out = []
    for it in range(n_iter):
        sys.stdout.write(''); sys.stdout.flush()
        gamma1 = 1. + np.sum(phi[:T-1,:], axis=1)
        phi_cum = np.cumsum(phi[:0:-1,:], axis=0)[::-1,:]
        gamma2 = alpha + np.sum(phi_cum, axis=1)

        tau = base_dirichlet + phi * X

        lV1 = psi(gamma1) - psi(gamma1 + gamma2)  # E_q[log V_t]
        lV1 = np.vstack((lV1, 0.))
        lV2 = psi(gamma2) - psi(gamma1 + gamma2)  # E_q[log (1-V_t)]
        lV2 = np.cumsum(np.vstack((0., lV2)), axis=0)  # \sum_{i=1}^{t-1} E_q[log (1-V_i)]

        eta = psi(tau) - psi(np.sum(tau, axis=1)) # E_q[eta_t]

        S = lV1 + lV2 + eta * X.T
        S = S - logsumexp(S, axis=0)
        phi = np.exp(S)

        ll.append(log_likelihood(X, gamma1, gamma2, tau,
            alpha, base_dirichlet, phi=phi, eta=eta))
        if Xtest is not None:
            held_out.append(mean_log_predictive(Xtest, gamma1, gamma2, tau,
                alpha, base_dirichlet, eta=eta))

    return gamma1, gamma2, tau, phi, ll, held_out

def log_likelihood(X, gamma1, gamma2, tau, alpha, base_dirichlet, phi, eta=None):
    '''computes lower bound on log marginal likelihood'''
    lV1 = psi(gamma1) - psi(gamma1 + gamma2)  # E_q[log V_t]
    lV11 = np.vstack((lV1, 0.))
    lV2 = psi(gamma2) - psi(gamma1 + gamma2)  # E_q[log (1-V_t)]
    lV22 = np.cumsum(np.vstack((0., lV2)), axis=0)  # \sum_{i=1}^{t-1} E_q[log (1-V_i)]
    lambda1 = np.matrix(base_dirichlet).T

    T = tau.shape[0]

    if eta is None:
        eta = psi(tau) - psi(np.sum(tau, axis=1))

    phi_cum = np.cumsum(phi[:0:-1,:], axis=0)[::-1,:]

    # E_q[log p(V|alpha)]
    ll = np.sum((alpha - 1) * lV2) - \
            (T-1) * (gammaln(alpha) - gammaln(1.+alpha))

    # E_q[log p(eta|lambda)]
    ll += np.sum(eta * (lambda1 - 1)) - \
            T * (np.sum(gammaln(lambda1)) - gammaln(np.sum(lambda1)))

    # \sum_n E_q[log p(Z_n|V)]
    ll += np.sum(np.multiply(phi[:-1,:], lV1) + np.multiply(phi_cum, lV2))

    # \sum_n E_q[log p(x_n | Z_n)]
    ll += np.sum(np.multiply(phi.T, X * eta.T))

    # - E_q[log q(V)]
    ll -= ((gamma1 - 1).T * lV1 + (gamma2 - 1).T * lV2).item() - \
            np.sum(gammaln(gamma1) + gammaln(gamma2) - gammaln(gamma1 + gamma2))

    # - E_q[log q(eta)]
    ll -= np.sum(np.multiply(tau - 1, eta)) - \
            np.sum(np.sum(gammaln(tau), axis=1) - gammaln(np.sum(tau, axis=1)))

    # - E_q[log q(z)]
    ll -= np.sum(np.nan_to_num(np.multiply(phi, np.log(phi))))

    return ll

def mean_log_predictive(X, gamma1, gamma2, tau, alpha, base_dirichlet, eta=None):
    '''Computes the mean of the log predictive distribution over sample X,
    typically held out data.'''
    lV1 = psi(gamma1) - psi(gamma1 + gamma2)  # E_q[log V_t]
    lV11 = np.vstack((lV1, 0.))
    lV2 = psi(gamma2) - psi(gamma1 + gamma2)  # E_q[log (1-V_t)]
    lV22 = np.cumsum(np.vstack((0., lV2)), axis=0)  # \sum_{i=1}^{t-1} E_q[log (1-V_i)]

    if eta is None:
        eta = psi(tau) - psi(np.sum(tau, axis=1))

    # E_q[log pi(V)]
    lPi = lV11 + lV22

    # p(x_N|x) = \sum_t E_q[pi_t(V)] E_q[p(x_N|eta_t)]
    lPred = logsumexp(lPi.T + X * eta.T, axis=1)
    return lPred.mean()

def print_top_words_for_topics(topics, tau, counts=None, n_words=10):
    voc = np.array(open(vocab_file).read().strip().split('\n'))

    if isinstance(topics, tuple):
        for topic, prob in zip(*topics):
            idx = np.argsort(tau[topic,:].A1)[::-1]
            # print '{} ({}): {}'.format(topic, float(prob),
            #         ', '.join(voc[idx[:n_words]]))
    elif counts is not None:
        for topic, count in zip(topics, counts):
            idx = np.argsort(tau[topic,:].A1)[::-1]
            # print '{} ({}): {}'.format(topic, count,
            #         ', '.join(voc[idx[:n_words]]))
    else:
        for topic in topics:
            idx = np.argsort(tau[topic,:].A1)[::-1]
            # print '{}: {}'.format(topic, ', '.join(voc[idx[:n_words]]))

def top_topics_of_document(n, phi, n_topics=None):
    idx = np.argsort(phi[:,n].A1)[::-1]
    return idx[:n_topics], phi[idx[:n_topics],n]

if __name__ == '__main__':
    # load AP data - http://www.cs.princeton.edu/~blei/lda-c/ap.tgz
    data_file = '../ap/ap.dat'
    vocab_file = '../ap/vocab.txt'
    X = load_ap_data(data_file, vocab_file)

    # load BOW data - http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/
    # data_file = '../bow/docword.nips.txt'
    # vocab_file = '../bow/vocab.nips.txt'
    # X = load_bow_data(data_file)
    N, M = X.shape
    # print 'Data loaded: {} documents, {} words in vocabulary'.format(N, M)

    alpha = 1
    base_dirichlet = np.ones(M)

    g1, g2, tau, phi, ll, held_out = var_dpmm_multinomial(X[:1000,:], alpha, base_dirichlet, T=50, n_iter=50, Xtest=X[1000:,:])

    # print topics/clusters, ordered by decreasing number of assigned documents
    assigned_topics = np.argmax(phi, axis=0).A1
    counts = np.bincount(assigned_topics)
    idx = np.argsort(counts)[::-1]
    print
    print_top_words_for_topics(idx, tau, counts=counts[idx])
