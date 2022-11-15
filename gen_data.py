import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# TODO list:
# OK -  Merge VV VH, merge orbites
# OK - Stats sur les zéros, enlever?
# - Remove grid average
# - NDVI
# - Labels
#   OK - Histogramme des classes.
#   OK - Search labels correspondance (using id)

# TODO: instead of manually choosing year, orb and polarization,
# just scan the folder and treat the whole data

year = 2018  # Year of interest. Options are 2018, ..., 2020
show_plots = True

X = []
default_ids = False
for polarization in ['VV', 'VH']:
    allOrbs_df = pd.DataFrame()
    for orb in [8, 30, 81]:

        print(f'\nYear {year}, orbit {orb}, type {polarization}:')

        ### LOAD DATA ###
        # Place your data at './Colza_DB/Plot/'
        # E.g.: './Colza_DB/Plot/2018/Orb30_2017_Colza2018/mean_VH_Orb30.csv'
        filepath1 = './Colza_DB/Plot/' + str(year) + '/Orb' + str(orb) + '_' \
            + str(year-1) + '_Colza' + str(year) + '/'
        filepath2 = './Colza_DB/Plot/' + str(year) + '/Orb' + str(orb) + '_' \
            + str(year) + '_Colza' + str(year) + '/'
        filename = 'mean_' + polarization + '_Orb' + str(orb) + '.csv'

        # Load csv from year and year-1
        df1 = pd.read_csv(filepath1 + filename, delimiter=';', index_col=0)
        df2 = pd.read_csv(filepath2 + filename, delimiter=';', index_col=0)

        ### SANITY CHECKS ###
        # Check if indexes match
        # Compare year-1 to year
        if not df1.index.equals(df2.index):
            print(f'WEIRD! Indexes in {year-1} do not match those in {year}!')
        # Compare to default index values
        if not default_ids:
            ids = df2.index
            default_ids = True
        if not df1.index.equals(ids):
            print(f'WEIRD! Indexes in {year-1} do not match default indexes!')
        if not df2.index.equals(ids):
            print(f'WEIRD! Indexes in {year} do not match default indexes!')

        # Check for duplicated indexes
        dup_idx = np.where(df1.index.duplicated(keep=False))
        if dup_idx[0].size > 0:
            print(f'WEIRD! Duplicate index at row {dup_idx[0]} year {year-1}.')
            # Keeping only the first occurence.
            # df1 = df1.loc[~df1.index.duplicated(keep='first')]
            # Overwriting current index by default index
            df1 = df1.set_index(ids)

        dup_idx = np.where(df2.index.duplicated(keep=False))
        if dup_idx[0].size > 0:
            print(f'WEIRD! Duplicate index at {dup_idx[0]} year {year}.')
            # Keeping only the first occurence.
            # df2 = df2.loc[~df2.index.duplicated(keep='first')]
            # Overwriting current index by default index
            df2 = df2.set_index(ids)

        ### CONCATENATE AND TRIM MONTHS OF INTEREST ###
        # Concatenate year and year-1
        df = pd.concat([df1, df2], axis=1, join='outer')
        # Sort by date (should already be sorted)
        df.sort_index(axis=1, ascending=True)

        # Keeping only months of interest (Oct/N-1 to July/N)
        df.columns = pd.to_datetime(df.columns)
        date_mask = (df.columns >= str(year-1) + '-10-01') \
            & (df.columns < str(year)+'-08-01')
        df = df.loc[:, date_mask]

        # Convert to Numpy and to dB
        data = df.to_numpy()
        print(f'Data dimensions: ', data.shape)

        ### HANDLE MISSING DATA (ZEROS) ###
        zeros_per_row = np.count_nonzero(data, axis=1)
        zeros_per_column = np.count_nonzero(data, axis=0)
        missing_ratio = zeros_per_row.sum()/data.size
        if show_plots:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(
                f'Orbit {orb}, Polarization {polarization}, {missing_ratio*100:.1f}% missing')
            # Zeros per row
            ax1.hist(zeros_per_row, bins=range(data.shape[1]+1))
            ax1.set_title('Zero entries per row')
            ax1.set(xlabel='# Missing time entries', ylabel='# Rows')
            # Zeros per column
            w = round(data.shape[0]/50)  # bin width
            ax2.hist(zeros_per_column, bins=np.arange(0, data.shape[0] + w, w))
            ax2.set_title('Zero entries per column')
            ax2.set(xlabel='# Missing samples', ylabel='# Columns')
            # ax2.set_xlim(0, data.shape[0]+1)
            # View data as colormap
            # plt.imshow(data == 0, aspect='auto',cmap='Greys')  # Visualize data
            # plt.xlabel('Time index'), plt.ylabel('Sample index')

        ### MERGE ORBITS ###
        print('\nMERGING ORBITS')
        print(f'Shape allOrbd_df: {allOrbs_df.shape}')
        print(f'Shape current df: {df.shape}')
        if not df.index.equals(allOrbs_df.index):
            print(f'WEIRD! Indexes do not match among orbits!')

        allOrbs_df = pd.concat([allOrbs_df, df], axis=1, join='outer')
        allOrbs_df.sort_index(axis=1, ascending=True)  # Sort by date

    print(f'All orbits dimensions: {allOrbs_df.shape}')

    ### MERGE POLARIZATIONS ###
    X.append(allOrbs_df.to_numpy())

X = np.stack(X, axis=-1)
# X = 10*np.log10(X) # Convert to dB
print(f'Final data dimension: {X.shape}')

### NDVI DATA ###

### LABELS ###
print('\nTREATING LABELS')
# Place your data at './Colza_DB/ID/'
# E.g.: './Colza_DB/ID/gid_5000_2018.csv'
filepath = './Colza_DB/ID/'
filename = 'gid_5000_' + str(year) + '.csv'
gid_all = pd.read_csv(filepath + filename, delimiter=',', index_col=1)
print(f'Total listed samples: {gid_all.shape[0]}')
# Filter rows corresponding to default indexes
gid = gid_all.loc[ids]
print(f'Total available samples: {gid.shape[0]}')

# Class histograms
if show_plots:
    # Full data
    plt.figure()
    class_counts = gid_all["CODE_CULTU"].value_counts()
    class_counts[:20].plot.bar()
    plt.title(
        f'Histogram of more common classes (20 out of {len(class_counts)})')
    # Filtered data
    plt.figure()
    class_counts = gid_all["CODE_CULTU"].value_counts()
    class_counts[:20].plot.bar()
    plt.title(
        f'Histogram of classes on available data')

# Check for duplicated indexes
dup_idx = np.where(gid.index.duplicated(keep=False))
if dup_idx[0].size > 0:
    print(f'WEIRD! Duplicate indexes at rows {dup_idx[0]} year {year-1}.\
            A total of {len(dup_idx[0])} duplicates out of {gid.shape[0]}.')
    # Keeping only the first occurence
    print('Keeping only the first occurence...')
    gid = gid.loc[~gid.index.duplicated(keep='first')]

y = gid["CODE_CULTU"].to_numpy()
