from model.hyperparameters_model.old.NIW_MIX_0.NIW_MIX_0_parameters import Nu_0_MIX, Mu_0_MIX, Lambda_0_MIX, PHI_0_MIX


class NIW_MIX_VAR:
    # parametri NIW MIX:
    # mu_0_MIX -> matrice (Jxp)
    #           -> riga per riga ci sono le medie delle componenti della mistura
    #               (mu_0_MIX[i,:] -> media della (i+1)-esima NIW della mistura)
    #
    # nu_0_MIX -> vettore (J) componenti
    #               (nu_0_MIX[i] = nu_0 della (i+1) esima NIW della mistura)
    #
    # lambda_0_MIX -> vettore (J) componenti
    #               (lambda_0_MIX[i] = lambda_0 della (i+1)  esima NIW della mistura)
    #
    # PHI_0_MIX -> vettore di matrici (Jxpxp), sostanzialmente un ndarray
    #           -> PHI_0_MIX[i,:,:] = Matrice PHI della (i+1)esima NIW della mistura


    def __init__(self, mu_0_MIX: Mu_0_MIX, nu_0_MIX : Nu_0_MIX, lambda_0_MIX : Lambda_0_MIX, phi_0_MIX: PHI_0_MIX):
        self.mu_0_MIX = mu_0_MIX
        self.nu_0_MIX = nu_0_MIX
        self.lambda_0_MIX = lambda_0_MIX
        self.phi_0_MIX = phi_0_MIX
