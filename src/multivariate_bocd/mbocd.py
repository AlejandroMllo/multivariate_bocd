import torch


class BOCD:
    """
    Bayesian Online Changepoint Detection (BOCD) Module.

    Based on:
    Adams, Ryan Prescott, and David JC MacKay. "Bayesian online changepoint detection."
    arXiv preprint arXiv:0710.3742 (2007).

    Code adapted by Alejandro Murillo-Gonzalez, from the following
    works by Gregory Gundersen:
        https://github.com/gwgundersen/bocd/blob/master/bocd.py
        http://gregorygundersen.com/blog/2019/08/13/bocd/
        http://gregorygundersen.com/blog/2020/10/20/implementing-bocd/
    """

    def __init__(self, model, hazard, device='cuda'):

        self.__model: MultivariateGaussianWishartModel = model     # TODO: Abstract class for the models, i.e., BOCDModel
        self.__device = device
        self.__hazard = hazard
        # self.__data = None
        self.__num_observations = 0

        # Initialize algorithm
        self.__log_H = torch.tensor(self.__hazard).log()
        self.__log_1mH = torch.tensor(1 - self.__hazard).log()

        self.__log_message = torch.tensor(1.0).log()       # P(r_0 = 0, x_{0:0} = {}) = 1
        self.__log_R = [self.__log_message]

        self.changepoints = []

    def add_observation(self, obs):
        """
        Perform one iteration of Algorithm 1 from Adams & MacKay (2007),
        to include the latest observed data into the changepoint analysis.
        """
        assert len(obs.shape) == 1

        # Step 2. "Observe New Datum"
        self.__num_observations += 1

        # Step 3. "Evaluate Predictive Probability"
        log_pis = self.__model.log_pred_prob(self.__num_observations, obs)  # Vector of length `num_observations`

        # Step 4. "Calculate Growth Probabilities"
        log_growth_probs = log_pis + self.__log_message + self.__log_1mH  # Vector of length `num_observations`

        # Step 5. "Calculate Changepoint Probability"
        log_cp_prob = torch.logsumexp(log_pis + self.__log_message + self.__log_H, dim=0)  # Float Scalar

        # Step 6. "Calculate Evidence"
        new_log_joint = torch.cat([log_cp_prob.unsqueeze(dim=0), log_growth_probs], dim=0)   # Vector of length `num_observations` + 1
        evidence = torch.logsumexp(new_log_joint, dim=0)             # P(data)

        # Step 7. "Determine Run Length Distribution"
        log_run_length_prob = new_log_joint - evidence
        self.__log_R.append(log_run_length_prob)

        # Step 8. "Update Sufficient Statistics"
        self.__model.update_params(self.__num_observations, obs)

        # Misc. Steps:
        if log_run_length_prob[0] > log_run_length_prob[-1]:      # if changepoint prob > run growth prob:
            cp = self.__num_observations
            self.changepoints.append(cp)
            self.__model.register_changepoint(cp)
            self.__log_message = torch.tensor(1.0).log()
        else:
            self.__log_message = new_log_joint

        return dict(
            cp_prob=log_run_length_prob[0].exp(),
            grow_prob=log_run_length_prob[-1].exp(),
            run_data=self.__model.current_run_data()
        )
    
    def run_length_probabilities(self):

        l = len(self.__log_R)
        run_probs = -torch.inf * torch.ones(l, l)
        for t, t_probs in enumerate(self.__log_R):
            
            r_len = len(t_probs) if len(t_probs.shape) > 0 else 1
            run_probs[-r_len:, t] = t_probs

        return run_probs.exp().numpy()


class MultivariateGaussianWishartModel:

    def __init__(self, mu0, kappa0, nu0, T0, reset_prior_on_changepoint):
        """
        Multivariate Gaussian-Wishart prior where both mean and precision parameters are unknonw.

        Derivations obtained from:
        "Conjugate Bayesian analysis of the Gaussian distribution"
        https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

        Code adapted by Alejandro Murillo-Gonzalez, from the following
        works by Gregory Gundersen:
        https://github.com/gwgundersen/bocd/blob/master/bocd.py
        http://gregorygundersen.com/blog/2019/08/13/bocd/
        http://gregorygundersen.com/blog/2020/10/20/implementing-bocd/

        :param mu0:    prior mean vector.
        :param kappa0: prior scaling factor for the precision matrix of the Gaussian distribution. (Equivalent to prior sample size).
        :param nu0:    prior degrees of freedom of the Wishart distribution.
        :param T0:     prior precision matrix.
        """

        assert len(mu0.shape) == 1
        assert kappa0 > 0
        assert nu0 > mu0.shape[0] - 1

        self.__reset_prior_on_changepoint = reset_prior_on_changepoint

        self.d = mu0.shape[0]     # Dimensionality of the data modeled by this distribution.
        self.mu0 = mu0
        self.kappa0 = kappa0 if torch.is_tensor(kappa0) else torch.tensor(kappa0)
        self.nu0 = nu0 if torch.is_tensor(nu0) else torch.tensor(nu0)
        self.T0 = T0

        self.data = torch.tensor([])
        self.mu_history = self.mu0.clone().unsqueeze(dim=0)
        self.kappa_history = torch.tensor([self.kappa0], dtype=torch.float64)
        self.nu_history = torch.tensor([self.nu0], dtype=torch.float64)
        self.T_history = self.T0.clone().unsqueeze(dim=0)

        self.__changepoints = [0]

    def register_changepoint(self, timestep):

        if self.__reset_prior_on_changepoint:
            # self.mu0 = self.mu_history[-1]
            # self.kappa0 = self.kappa_history[-1]
            # self.nu0 = self.nu_history[-1]
            # self.T0 = self.T_history[-1]

            t = self.latest_changepoint()
            data = self.data[t:]
            self.mu0 = data.mean(dim=0)
            self.kappa0 = torch.tensor(data.shape[0])  # torch.tensor(2)
            self.nu0 = torch.tensor(data.shape[1])  # torch.tensor(data.shape[0]+1)
            self.T0 = data.T.cov().inverse() if data.shape[0] > data.shape[1] else self.T0

        self.__changepoints.append(timestep)
        # print(self.__changepoints)

    def current_run_data(self):
        t = self.latest_changepoint()
        return self.data[t:]

    def latest_changepoint(self):
        return self.__changepoints[-1]
    
    def latest_params(self):
        return self.mu_history[-1], self.kappa_history[-1], self.nu_history[-1], self.T_history[-1]
    
    def changepoints(self):
        return self.__changepoints

    def log_pred_prob(self, t, x):
        """
        Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.

        E.g., if t = 5, the function returns the log-likelihood of `x` for 5
              parametrizations of the likelihood function (Gaussian in this case),
              where the parameters \eta_i correspond to \eta_0 = prior, and for
              i > 0 they are the updated parameters after receiving observation i-1.
        """

        if not torch.is_tensor(x):
            x = torch.tensor(x)

        t0 = self.latest_changepoint()

        means = self.mu_history[t0:t]
        precisions = self.T_history[t0:t]

        dist = torch.distributions.MultivariateNormal(loc=means, precision_matrix=precisions)
        ll = dist.log_prob(x)

        return ll

    def update_params(self, t, x):
        """Upon observing a new datum x at time t, update all run length 
        hypotheses.

        Updating the parameters with the observation x at time t, involves
        updating the previous t parameters \eta_i (0 <= i <= t) and including
        a new one for the next timestep.
            Basically the update should be something like: [prior] + [t new parameterizations]
            where the t new parameterizations consist of applying the update to each previous \eta_i.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)

        self.data = torch.cat([self.data, x.unsqueeze(dim=0)], dim=0)

        t0 = self.latest_changepoint()

        n = 1 # x.shape[0]
        data_mean = x  # if n == 1 else torch.mean(x, dim=0)

        # mu_n
        mu_n = (self.kappa_history.unsqueeze(1) * self.mu0 + n * data_mean) / (self.kappa_history + n).unsqueeze(1)
        self.mu_history = torch.cat([self.mu0.unsqueeze(dim=0), mu_n], dim=0)

        # S_n (symmetric d x d nonnegative definite matrix)
        data_diff = self.data[t0:, :] - data_mean
        S = torch.matmul(torch.transpose(data_diff, 0, 1), data_diff)

        # T_n
        mean_diff = (self.mu0 - data_mean).reshape(-1, 1)
        mean_dif_cov = torch.matmul(mean_diff, torch.transpose(mean_diff, 0, 1))
        # T_n = (d x d) + (d x d) + () * (4 x 4)
        mean_dif_cov_scale = ((self.kappa_history * n) / (self.kappa_history + n)).view(-1, 1, 1)
        T_n = self.T0 + S + mean_dif_cov_scale * mean_dif_cov

        self.T_history = torch.cat([self.T0.unsqueeze(dim=0), T_n], dim=0)
        # print('T params', mean_diff.shape, mean_dif_cov.shape, T_n.shape, self.T_history.shape, 'T0', self.T0.shape)

        # nu_n
        nu_n = self.nu_history + n
        self.nu_history = torch.cat([self.nu0.unsqueeze(0), nu_n], dim=0)

        # kappa_n
        kappa_n = self.kappa_history + n
        self.kappa_history = torch.cat([self.kappa0.unsqueeze(0), kappa_n], dim=0)

    @staticmethod
    def init_from_data(data, reset_prior_on_changepoint):

        if not torch.is_tensor(data):
            data = torch.tensor(data)

        mean = torch.mean(data, dim=0)
        precision = torch.linalg.inv(torch.cov(torch.transpose(data, 1, 0)))

        model = MultivariateGaussianWishartModel(
            mu0=mean,
            kappa0=data.shape[0],
            nu0=mean.shape[0],
            T0=precision,
            reset_prior_on_changepoint=reset_prior_on_changepoint
        )

        # print('init_from_data', data.shape, mean.shape, precision.shape)
        
        return model
