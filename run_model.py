from experiment import VAEXperiment
import yaml
import torch 
from models import *
from dataset import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if __name__ == '__main__':
    config = yaml.safe_load(open('./configs/vae_1d.yaml'))
    model = vae_models[config['model_params']['name']](**config['model_params'])
    # ckpt = torch.load('./logs/VanillaVAE1D/version_13/checkpoints/last.ckpt')
    ckpt = torch.load('./logs/VanillaVAE1D/version_24/checkpoints/last.ckpt')
    experiment = VAEXperiment(model, config['exp_params'])
    #model.load_state_dict(ckpt['state_dict'])

    experiment.load_state_dict(ckpt['state_dict'])
    data = VAEDataset1D(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    data.setup()

    # test_data = next(iter(data.test_dataloader()))

    test_input, labels, recons, samples = experiment.sample_images(save_images=False, dataloader=data.test_dataloader())
    plt.plot(test_input[0,0])
    plt.plot(recons[0,0, 0])
    plt.show()

    # plt.subplot(1, 2, 2)
    # plt.plot(test_input[0,0],recons[0,0,0])

    # EXPLORATORY TEST #
    # Generate synthetic data that varies across single controlling variable, holding others constant #
    # EG for a given frequency sin wave, present inputs along a gradient of increasing bias #
    # 1) is there a 1:1 correspondence with a single latent mean dimension #
    # 2) while varying overall noise, what happens to the estimated variance? #
    # 3) is there zero crossing? In other words:
    #   if inputs are sorting based on ascending parameter values,
    #   the latent space, whether ascending or descending, should also be sorted

    b = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5]
    sweep = data.val_dataset.__paramsweep__(b=[-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5], sigma=[0.5], r=[64])
    mu, log_var = experiment.model.encode(sweep[0])
    mu = mu.detach().numpy()
    # mu is B x L
    # where B is the input batch size, and L is the dimension of the latent space
    # graph each dimension of the latent space as a line
    for i in range(mu.shape[1]):
        plt.plot(b,mu[:,i])
    plt.legend(np.arange(mu.shape[1]))
    plt.show()


    r = [16, 32, 48, 64, 80, 96, 112, 128]
    sweep = data.val_dataset.__paramsweep__(b=[-0.25], sigma=[0.5], r=r)
    mu, log_var = experiment.model.encode(sweep[0])
    mu = mu.detach().numpy()
    # mu is B x L
    # where B is the input batch size, and L is the dimension of the latent space
    # graph each dimension of the latent space as a line
    for i in range(mu.shape[1]):
        plt.plot(r,mu[:,i])
    plt.legend(np.arange(mu.shape[1]))
    plt.show()

    test_input = sweep[0]

    # EXPLORATORY TEST #
    # Give two inputs; interpolate in latent space; are they similar? #
    test_input, recons, samples = experiment.sample_images(save_images=False, dataloader=data.test_dataloader())
    
    interpolated = experiment.model.interpolate_inputs(test_input[3], test_input[15])
    res = interpolated[0].detach().numpy()
    for i in range(res.shape[0]):
        plt.plot(res[i,0])
    plt.legend(np.arange(res.shape[0]))
    plt.show()


    # EXPLORATORY TEST #
    # Once trained on weather data from different sites, do distances in latent space correspond to #
    # 1) Distances in geographic space
    # 2) Distances according to popular metrics of climatic similarity?
    # if not, can it be enforced in loss function?

    # TO-DO #
    # Single input to probabilistic output #
    result = experiment.model.probabilistic_generate(x=test_input[[3]], M=100).detach().numpy()
    quant = np.quantile(result, q=[0.025, 0.5, 0.975], axis=0)
    mean = np.mean(result, axis=0)
    colors = list(mcolors.BASE_COLORS)
    n_lines = 1
    offset=0
    for i in range(n_lines):
        single_test = quant[:, i+offset, 0, :] 
        plt.plot(single_test[1], color=colors[i])
        plt.fill_between(x = np.arange(single_test.shape[-1]), y1=single_test[0], y2=single_test[2], alpha=0.5, color=colors[i])

    for i in range(n_lines):
        plt.plot(test_input[i+offset, 0], color=colors[i], alpha=0.25)

    plt.show()

    # TO-DO #
    # Implement structural similarity loss function; does this capture more of variance? #





    print("done")

