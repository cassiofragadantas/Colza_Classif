import numpy as np
from matplotlib import pyplot as plt


def main():

    # fig1, axs1 = plt.subplots(5, 1, constrained_layout=True)
    # fig1.suptitle('Colza', fontsize=16)

    titles_SAR = ['VV', 'VV grid', 'VH', 'VH grid']


    for year in [2018, 2019, 2020]:
        dataset=np.load(f'Colza_DB/Colza_data_{year}.npz', allow_pickle=True)
        X_SAR, X_NDVI, y_multi, id_p=dataset["X_SAR"], dataset["X_NDVI"], dataset["y"], dataset["id_parcels"]


        # --- Colza profiles ---
        # SAR
        colza_SAR=X_SAR[y_multi == 'CZH']
        n_times=X_SAR.shape[1]
        mean=np.mean(colza_SAR, axis=0)
        std=np.std(colza_SAR, axis=0)
        for k, title in enumerate(titles_SAR):
            plt.figure(k)
            plt.title('Colza ' + title)
            plt.plot(mean[:, k])
            plt.fill_between(range(n_times), mean[:, k] +
                                std[:, k], mean[:, k]-std[:, k], alpha=0.5)        
        # plt.figure(fig1)
        # for k, title in enumerate(axs1):
        #     axs1[k].set_title(title)
        #     axs1[k].plot(mean[:, k])
        #     axs1[k].fill_between(range(n_times), mean[:, k] +
        #                         std[:, k], mean[:, k]-std[:, k], alpha=0.5)

        # NDVI
        colza_NDVI=X_NDVI[y_multi == 'CZH']
        n_times=X_NDVI.shape[1]
        mean=np.mean(colza_NDVI, axis=0)
        std=np.std(colza_NDVI, axis=0)
        plt.figure(5)
        plt.title('NDVI')
        plt.plot(mean)
        plt.fill_between(range(n_times), mean+std, mean-std, alpha=0.5)

        # colza_np = colza[:,:,0]
        # colza_norm = colza_np/np.sqrt((colza_np * colza_np).sum(axis=1))[:,np.newaxis] # normalize rows
        # cov_mtx = colza_norm.dot(colza_norm.T)

        # Dates

if __name__ == "__main__":
    main()
