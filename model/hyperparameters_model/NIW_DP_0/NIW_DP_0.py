from model.hyperparameters_model.NIW_DP_0.NIW_DP_0_parameters import Lambda_0_DP, Nu_0_DP, PHI_0_DP
from model.hyperparameters_model.NIW_DP_0.NIW_DP_0_parameters.Mu_0_DP import Mu_0_DP


class NIW_DP_0:
    # parametri NIW DP:
    # 	mu_0_DP -> vettore (p) componenti
    # 	nu_0_DP -> scalare
    # 	lambda_0_DP -> scalare
    # 	PHI_0_DP -> Matrice (pxp)

    def __init__(self, mu_0_DP: Mu_0_DP, nu_0_DP : Nu_0_DP, lambda_0_DP : Lambda_0_DP, phi_0_dp: PHI_0_DP):
        self.mu_0_DP = mu_0_DP
        self.nu_0_DP = nu_0_DP
        self.lambda_0_DP = lambda_0_DP
        self.phi_0_dp = phi_0_dp
