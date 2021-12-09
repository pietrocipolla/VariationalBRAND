class UserInput:
    def __init__(self, gamma, a_dir_k,
                 mu_0_DP,nu_0_DP,lambda_0_DP,PHI_0_DP,
                 mu_0_MIX,nu_0_MIX,lambda_0_MIX,PHI_0_MIX,
                 Phi_m_k,eta_k,a_k_beta,b_k_beta,
                 mu_var_DP,nu_var_DP,lambda_var_DP,PHI_var_DP,
                 mu_VAR_MIX,nu_VAR_MIX,lambda_VAR_MIX,PHI_VAR_MIX):
        self.gamma = gamma
        self.a_dir_k = a_dir_k

        self.mu_0_DP = mu_0_DP
        self.nu_0_DP = nu_0_DP
        self.lambda_0_DP = lambda_0_DP
        self.PHI_0_DP = PHI_0_DP

        self.mu_0_MIX = mu_0_MIX
        self.nu_0_MIX = nu_0_MIX
        self.lambda_0_MIX = lambda_0_MIX
        self.PHI_0_MIX = PHI_0_MIX

        self.Phi_m_k = Phi_m_k
        self.eta_k = eta_k
        self.a_k_beta = a_k_beta
        self.b_k_beta = b_k_beta

        self.mu_var_DP = mu_var_DP
        self.nu_var_DP = nu_var_DP
        self.lambda_var_DP = lambda_var_DP
        self.PHI_var_DP = PHI_var_DP

        self.mu_VAR_MIX = mu_VAR_MIX
        self.nu_VAR_MIX = nu_VAR_MIX
        self.lambda_VAR_MIX = lambda_VAR_MIX
        self.PHI_VAR_MIX = PHI_VAR_MIX
