def init_cavi_old(hyperparameters_model):
    #########
    # INIT
    # cavi hyperparameters_model & inizialization
    # J: num di cluster learning set
    J = num_classes_learning

    # dirichlet hyperparameters_model
    dirichlet_hyperparameters_ai = np.ones(num_classes)

    # - eta
    # Rappresentano la probabilità di essere o in una delle j classi
    # precedentemente viste o #probabilità di essere appartenente ad una nuova (circa)
    eta_0 = np.ones(num_classes)  # todo, to change: tutti alfa_k uguali; 1/ (flat prior)

    # - ak, bk
    # parametri dirichlet
    a_0 = 1  # per tenerle flat = 1 (a priori le vk sono dell uniforme)
    b_0 = 1  # per tenerle flat = 1 (a priori le vk sono dell uniforme)
    from scipy.stats import beta
    beta_0 = beta(a_0, b_0)  # todo check

    # - fi
    # parametro multinomiale
    # probabilità che n-esimo dato appartenga al gruppo k
    fi_0 = []
    J = num_classes_learning
    for i in num_classes_learning:
        fi_0.append(1 / (J + 1))
    for i in num_classes_test:
        fi_0.append(get_fi_novelty(J))

    # hyperparameters_model pink
    lambda_0 = 1
    mu_0 = robust_mean  # inizializzazione tenendo conto che ho gia gruppi noti, con le leoor medie e uso quelle medie
    nu_0 = 2  # con 2 = p numero di componenti del vettore di y
    psi_0 = np.identity(2 * n_samples)  # matrice identità, dim = da dim vettore y due compoentni e 500 valori

    # - gamma
    # alfa del DP, hyperparameter
    # iperparametro tra 1 e 50 tipo oppure buttarci su una distribuzione e una prior
    gamma = 5

    return initialized_cavi_parameters