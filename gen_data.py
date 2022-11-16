import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# TODO list:
# OK -  Merge VV VH, merge orbites
# OK - Stats sur les zéros, enlever?
# - Remove grid average
# OK - NDVI
# OK - Labels
# OK - Histogramme des classes.
# OK - Search labels correspondance (using id)

# TODO: instead of manually choosing year, orb and polarization,
# just scan the folder and treat the whole data

year = 2018  # Year of interest. Options are 2018, ..., 2020
show_plots = True


def check_duplicates(df, id='csv'):
    # Duplicate rows
    (dup_idx, ) = np.where(df.index.duplicated(keep=False))
    if dup_idx.size > 0:
        print(
            f'WEIRD! {len(dup_idx)} duplicate indexes at rows {dup_idx+2} in {id}.')
        if not all(df.loc[df.index.duplicated(keep=False)].duplicated()):
            print(f'\tEVEN WEIRDER! Duplicate indexes have different content.')
        # Keeping only the first occurence.
        df = df.loc[~df.index.duplicated(keep='first')]
    # Duplicate rows
    dup_idx = df.columns.str.find('.')
    if any(dup_idx >= 0):
        print(
            f'WEIRD! {sum(dup_idx >= 0)} duplicate indexes at columns {dup_idx[dup_idx >= 0]+2} in {id}.')
        # Keeping only the first occurence.
        df = df.loc[:, df.columns.str.find('.') < 0]
    return df


X_SAR = []
default_idx = pd.DataFrame().index
merged_idx = pd.DataFrame().index
for polarization in ['VV', 'VH']:
    allOrbs_df = pd.DataFrame()
    for orb in [81, 8, 30]:

        print(f'\nYear {year}, orbit {orb}, type {polarization}:')

        ### LOAD DATA ###
        # Place your data at './Colza_DB/Plot/'
        # E.g.: './Colza_DB/Plot/2018/Orb30_2017_Colza2018/mean_VH_Orb30.csv'
        path = f'./Colza_DB/Plot/'
        path1 = f'{year}/Orb{orb}_{year-1}_Colza{year}/'
        path2 = f'{year}/Orb{orb}_{year}_Colza{year}/'
        filename = f'mean_{polarization}_Orb{orb}.csv'

        # Load csv from year and year-1
        df1 = pd.read_csv(path + path1 + filename, delimiter=';', index_col=0)
        df2 = pd.read_csv(path + path2 + filename, delimiter=';', index_col=0)
        # Load grid mean data
        path = f'./Colza_DB/RPG_grid_mean/'
        gridmean1 = pd.read_csv(path+path1+filename,
                                delimiter=';', index_col=0)
        gridmean2 = pd.read_csv(path+path2+filename,
                                delimiter=';', index_col=0)
        # Load id correspondance (parcel and grid)
        gid = pd.read_csv(
            f'./Colza_DB/ID/gid_5000_{year}.csv', delimiter=',', index_col=1)
        gid = gid.loc[~gid.index.duplicated(keep='first')]  # Remove duplicates

        ### SANITY CHECKS ###
        # Check if indexes match
        # Compare year-1 to year
        if not df1.index.equals(df2.index):
            print(f'WEIRD! Indexes in {year-1} do not match those in {year}!')
        # Compare to default index values
        if not default_idx.empty:
            default_idx = df1.index.copy()
        if not df1.index.equals(default_idx):
            print(f'WEIRD! Indexes in {year-1} do not match default indexes!')
        if not df2.index.equals(default_idx):
            print(f'WEIRD! Indexes in {year} do not match default indexes!')

        # Check for duplicated indexes
        df1 = check_duplicates(df1, year-1)
        df2 = check_duplicates(df2, year)

        # Overwriting current index by default index
        # df1 = df1.set_index(default_idx)

        #### GET GRID MEAN ###
        # gridmean1[gridmean1.index.isin(df2['ID_g'])]
        df1_ID_g = gid.loc[df1.index]['ID_g']
        # gridmean1.loc[df1_ID_g] #TODO uncomment
        df2_ID_g = gid.loc[df2.index]['ID_g']
        # gridmean2.loc[df2_ID_g] #TODO uncomment

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

    ### STACKING POLARIZATIONS ###
    print('\nSTACKING POLARIZATIONS')
    if merged_idx.empty:
        merged_idx = allOrbs_df.index  # Default indexes for comparison
    else:
        if not allOrbs_df.index.equals(merged_idx):
            print(f'WEIRD! Indexes do not match between polarizations !')
    print(f'All orbits dimensions: {allOrbs_df.shape}')

    X_SAR.append(allOrbs_df.to_numpy())

#### SAR DATA ###
X_SAR = np.stack(X_SAR, axis=-1)
# X_SAR = 10*np.log10(X_SAR) # Convert to dB
print(f'\nFinal SAR data dimension: {X_SAR.shape}')

### NDVI DATA ###
print('\nADDING NDVI DATA')
path1 = f'./Colza_DB/Plot/{year}/NDVI/ndvi{year-1}_wgs84_Cipan{year}/'
path2 = f'./Colza_DB/Plot/{year}/NDVI/ndvi{year}_wgs84_Cipan{year}/'
filename = f'mean_ndvi_wgs84_ndvi_wgs84.csv'
# Load csv from year and year-1
df1 = pd.read_csv(path1 + filename, delimiter=';', index_col=0)
df2 = pd.read_csv(path2 + filename, delimiter=';', index_col=0)

# Check if indexes match
# Compare year-1 to year
if not df1.index.equals(df2.index):
    print(f'WEIRD! Indexes in {year-1} do not match those in {year}!')
default_idx = df1.index.copy()

# Check for duplicated indexes
df1 = check_duplicates(df1, year-1)
df2 = check_duplicates(df2, year)

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
X_NDVI = df.to_numpy()

print(f'\nFinal NDVI data dimension: {X_NDVI.shape}')

zeros_per_row = np.count_nonzero(X_NDVI, axis=1)
zeros_per_column = np.count_nonzero(X_NDVI, axis=0)
missing_ratio = zeros_per_row.sum()/X_NDVI.size
if show_plots:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'NDVI data, {missing_ratio*100:.1f}% missing')
    # Zeros per row
    ax1.hist(zeros_per_row, bins=range(X_NDVI.shape[1]+1))
    ax1.set_title('Zero entries per row')
    ax1.set(xlabel='# Missing time entries', ylabel='# Rows')
    # Zeros per column
    w = round(X_NDVI.shape[0]/50)  # bin width
    ax2.hist(zeros_per_column, bins=np.arange(0, X_NDVI.shape[0] + w, w))
    ax2.set_title('Zero entries per column')
    ax2.set(xlabel='# Missing samples', ylabel='# Columns')
    # View data as colormap
    # plt.imshow(X_NDVI == 0, aspect='auto',cmap='Greys')  # Visualize data
    # plt.xlabel('Time index'), plt.ylabel('Sample index')


### LABELS ###
print('\nTREATING LABELS')
# Place your data at './Colza_DB/ID/'
# E.g.: './Colza_DB/ID/gid_5000_2018.csv'
path = './Colza_DB/ID/'
filename = f'gid_5000_{year}.csv'
gid = pd.read_csv(path + filename, delimiter=',', index_col=1)
print(f'Total listed samples: {gid.shape[0]}')

# Check for duplicated indexes
gid = check_duplicates(gid, 'gid')

# Filter rows corresponding to default indexes
gid = gid.loc[allOrbs_df.index]
print(f'Total available samples: {gid.shape[0]}')

# Class histograms
if show_plots:
    plt.figure()
    class_counts = gid["CODE_CULTU"].value_counts()
    class_counts[:20].plot.bar()
    plt.title(
        f'Histogram of more common classes (20 out of {len(class_counts)})')

y = gid["CODE_CULTU"].to_numpy()

### SAVE DATA ###
np.savez(f'Colza_DB/Colza_data_{year}', X_SAR=X_SAR, X_NDVI=X_NDVI, y=y)
