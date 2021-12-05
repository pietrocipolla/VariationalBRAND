from model.variational_parameters.NIW_DP_VAR.NIW_DP_VAR_parameters import Mu_VAR_DP, Nu_VAR_DP, Lambda_VAR_DP, \
    PHI_VAR_DP

class NIW_DP_VAR:
    # parametri NIW DP
    #mu_var_DP -> matrice (Txp)
    # -> riga per riga ci sono le medie delle componenti della misturaDP
    #    (mu_var_DP[i,:] -> media della (i+1)-esima NIW della misturaDP)

    #nu_var_DP -> vettore (T) componenti
    # (nu_var_MIX[i] = nu_var della (i+1) esima NIW della misturaDP)

    #lambda_var_DP -> vettore (T) componenti
    # (lambda_var_DP[i] = lambda_var della (i+1) esima NIW della misturaDP)

    #PHI_var_DP -> vettore di matrici (Txpxp), sostanzialmente un ndarray
    # -> PHI_var_DP[i,:,:] = Matrice PHI della (i+1)esima NIW della misturaDP


    def __init__(self, mu_VAR_DP: Mu_VAR_DP, nu_VAR_DP : Nu_VAR_DP, lambda_VAR_DP : Lambda_VAR_DP, phi_VAR_dp: PHI_VAR_DP):
        self.mu_VAR_DP = mu_VAR_DP
        self.nu_VAR_DP = nu_VAR_DP
        self.lambda_VAR_DP = lambda_VAR_DP
        self.phi_VAR_dp = phi_VAR_dp
