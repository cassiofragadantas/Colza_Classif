import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd
import os

# BUG TESTING
# plt.plot(colza_SAR[:,43,-1])

def main():

    # fig1, axs1 = plt.subplots(5, 1, constrained_layout=True)
    # fig1.suptitle('Colza', fontsize=16)

    years = [2018, 2019, 2020]
    titles_SAR = ['VV', 'VV-grid', 'VH', 'VH-grid']

    path = "results/visualize_data"
    if not os.path.exists(path):
        os.makedirs(path)

    for year in years:
        dataset=np.load(f'Colza_DB/Colza_data_{year}.npz', allow_pickle=True)
        X_SAR, X_NDVI, y_multi, id_p=dataset["X_SAR"], dataset["X_NDVI"], dataset["y"], dataset["id_parcels"]
        dates_SAR = pd.to_datetime(dataset["dates_SAR"]) - pd.DateOffset(years=year-2018)
        dates_NDVI = pd.to_datetime(dataset["dates_NDVI"]) - pd.DateOffset(years=year-2018)

        # --- Colza profiles ---
        # SAR
        colza_SAR=X_SAR[y_multi == 'CZH']
        n_times=X_SAR.shape[1]
        mean=np.mean(colza_SAR, axis=0)
        std=np.std(colza_SAR, axis=0)
        for k, title in enumerate(titles_SAR):
            plt.figure(k)
            # X-axis: Index number
            plt.subplot(2,1,1)            
            plt.title('Colza ' + title)
            plt.plot(mean[:, k],label=year)
            plt.fill_between(np.arange(n_times), mean[:, k] +
                                std[:, k], mean[:, k]-std[:, k], alpha=0.5)
            plt.legend()
            # X-axis: Dates
            ax = plt.subplot(2,1,2)            
            plt.plot(dates_SAR, mean[:, k])
            plt.fill_between(dates_SAR, mean[:, k] +
                                std[:, k], mean[:, k]-std[:, k], alpha=0.5)
            plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
            ax.xaxis.set_major_formatter(DateFormatter("%b"))
            plt.savefig(f'{path}/SAR_mean_profile_{title}.pdf')

        # NDVI
        colza_NDVI=X_NDVI[y_multi == 'CZH']
        n_times=X_NDVI.shape[1]
        mean=np.mean(colza_NDVI, axis=0)
        std=np.std(colza_NDVI, axis=0)
        plt.figure(5)
        # X-axis: Index number
        plt.subplot(2,1,1)
        plt.title('NDVI')
        plt.plot(mean,label=year)
        plt.fill_between(np.arange(n_times), mean+std, mean-std, alpha=0.5)
        plt.legend()
        # X-axis: Dates
        ax = plt.subplot(2,1,2)
        plt.plot(dates_NDVI, mean)
        plt.fill_between(dates_NDVI, mean+std, mean-std, alpha=0.5)
        plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        plt.savefig(f'{path}/NDVI_mean_profile.pdf')

        # colza_np = colza[:,:,0]
        # colza_norm = colza_np/np.sqrt((colza_np * colza_np).sum(axis=1))[:,np.newaxis] # normalize rows
        # cov_mtx = colza_norm.dot(colza_norm.T)

        # Acquisition Dates
        plt.figure(6, figsize=(6,1))
        ax = plt.subplot(1,1,1)
        plt.title('Acquisition dates')
        ax.scatter(dates_NDVI, [0.5*(year-2017)]*len(dates_NDVI), marker='o', s=30, alpha=0.5)
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.yaxis.set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.get_yaxis().set_ticklabels([])
        plt.savefig(f'{path}/Acquisition_dates.pdf', bbox_inches = "tight")

if __name__ == "__main__":
    main()
