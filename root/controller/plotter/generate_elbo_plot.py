from matplotlib import pyplot as plt

def generate_elbo_plot(elbo_values, hyperparameters_model, SEED):
    plt.step(range(len(elbo_values)), elbo_values, )
    plt.xlabel('Iterations')
    plt.ylabel('Elbo Values')
    plt.title('The Elbo')
    #plt.show()
    p = hyperparameters_model.p
    M = hyperparameters_model.M
    elbo_name = 'elbo_' + str(SEED) + '_'+ str(p) + '_' + str(M) + '.png'

    plt.savefig(elbo_name)
    plt.close()