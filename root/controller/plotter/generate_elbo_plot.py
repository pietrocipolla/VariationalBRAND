from matplotlib import pyplot as plt

def generate_elbo_plot(elbo_values):
    plt.step(range(len(elbo_values)), elbo_values, )
    plt.xlabel('Iterations')
    plt.ylabel('Elbo Values')
    plt.title('The Elbo')
    #plt.show()
    plt.savefig('elbo.png')
    plt.close()