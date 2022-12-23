import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# TODO list:
# OK -  Merge VV VH, merge orbites
# OK - Stats sur les zéros, enlever?
# OK - Remove grid average
# OK - NDVI
# OK - Labels
# OK - Histogramme des classes.
# OK - Search labels correspondance (using id)

year = 2020 # Year of interest. Options are 2018, ..., 2020
join = 'inner' # pandas join type (inner or outer)
show_plots = True
interpolate = True
grid_mean = True
NDVI_trim_columns = False # Trim out all-zero columns


def check_duplicates(df, id='csv'):
    # Duplicate rows
    (dup_idx, ) = np.where(df.index.duplicated(keep=False))
    if dup_idx.size > 0:
        print(f'\tWEIRD! {len(dup_idx)} duplicate indexes at rows {dup_idx+2} in {id}.')
        if not all(df.loc[df.index.duplicated(keep=False)].duplicated()):
            print(f'\t\tEVEN WEIRDER! Duplicate indexes have different content.')
        # Keeping only the first occurence.
        df = df.loc[~df.index.duplicated(keep='first')]
    else:
        print(f'\tGREAT! No duplicate row indexes in {id}')
    # Duplicate columns
    dup_idx = df.columns.str.find('.')
    if any(dup_idx >= 0):
        print(f'\tWEIRD! {sum(dup_idx >= 0)} duplicate indexes at columns {dup_idx[dup_idx >= 0].values+2} in {id}.')
        # Keeping only the first occurence.
        df = df.loc[:, df.columns.str.find('.') < 0]
    else:
        print(f'\tGREAT! No duplicate columns indexes in {id}') 

    return df

def get_grid_mean(df,gridmean,id):
    df_ID_g = gid.loc[df.index]['ID_g']
    if all(df_ID_g.isin(gridmean.index)):
        print(f'\tGREAT! All grid indexes found in the grid mean at {id}.')
    else:
        missing_id_g = df_ID_g[~df_ID_g.isin(gridmean.index)].unique()
        missing_parcel_id = gid.loc[gid['ID_g'].isin(missing_id_g)].index
        print(f'\tISSUE! {len(missing_id_g)} grid indexes missing in the grid mean at {id}.')
        print(f'\t\tMissing grid indexes {missing_id_g} ' + \
              f'with {missing_parcel_id.shape} associated parcels (to be removed).')
        df=df[~df.index.isin(missing_parcel_id)]
        df_ID_g = gid.loc[df.index]['ID_g']

    gridmean = gridmean.loc[df_ID_g]
    gridmean.set_index(df.index, inplace=True)

    return df, gridmean


# Load id correspondance (parcel and grid)
print('\nGetting grid-parcel id correspondance')
gid = pd.read_csv(
    f'./Colza_DB/ID/gid_5000_{year}.csv', delimiter=',', index_col=1)
gid = check_duplicates(gid,f'gid_5000_{year}') # Remove duplicates

### NDVI DATA ###
print('\n===== ADDING NDVI DATA =====\n')
path1 = f'./Colza_DB/Plot/{year}/NDVI/ndvi{year-1}_wgs84_Cipan{year}/'
path2 = f'./Colza_DB/Plot/{year}/NDVI/ndvi{year}_wgs84_Cipan{year}/'
filename = f'mean_ndvi_wgs84_ndvi_wgs84.csv'
# Load csv from year and year-1
df1 = pd.read_csv(path1 + filename, delimiter=',', index_col=0)
df2 = pd.read_csv(path2 + filename, delimiter=';', index_col=0)
print(f'{year-1} data dimensions {df1.shape}')
print(f'{year} data dimensions {df2.shape}')

### SANITY CHECKS ###
print('\nSanity checks:')
# Check if indexes match
# Compare year-1 to year
if not df1.index.equals(df2.index):
    print(f'\tWEIRD! Indexes in {year-1} do not match those in {year}!')
    print(f'\t\tIntersection size is {df1.index.intersection(df2.index).shape}')
# Check for duplicated indexes
df1 = check_duplicates(df1, year-1)
df2 = check_duplicates(df2, year)

### CONCATENATE AND TRIM MONTHS OF INTEREST ###
print(f'\nConcatenating data from {year-1} and {year} (outer)...')
# Concatenate year and year-1
df = pd.concat([df1, df2], axis=1, join=join)
# Sort by date (should already be sorted)
df = df.sort_index(axis=1, ascending=True)
# Keeping only months of interest (Oct/N-1 to July/N)
df.columns = pd.to_datetime(df.columns)
date_mask = (df.columns >= str(year-1) + '-10-01') \
    & (df.columns < str(year)+'-08-01')
df = df.loc[:, date_mask]

# Missing data visualization
X_NDVI = df.to_numpy()
zeros_per_row = np.count_nonzero(X_NDVI==0, axis=1) +  np.count_nonzero(np.isnan(X_NDVI), axis=1)
zeros_per_column = np.count_nonzero(X_NDVI==0, axis=0) +  np.count_nonzero(np.isnan(X_NDVI), axis=0)
missing_ratio = zeros_per_row.sum()/X_NDVI.size
print(f'Data dimensions: {X_NDVI.shape}')
print(f'{missing_ratio*100:.2f}% missing data.')
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
    plt.figure()
    plt.imshow(X_NDVI == 0, aspect='auto',cmap='Greys')  # Visualize data
    plt.title('NDVI data presence (black corresponds to missing data)')
    plt.xlabel('Time index'), plt.ylabel('Sample index')

### INTERPOLATION ###
df = df.replace(0, np.nan)
# Remove all-zero rows (cannot be interpolated)
allNaN = df.isnull().all(1)
if any(allNaN):
    print(f'WEIRD! {allNaN.sum()} all-NaN rows. Removing...')
    df = df.loc[~allNaN]
else:
    print('GREAT! No all-NaN rows.')
# Remove all-zero columns (optional)
if NDVI_trim_columns:
    allNaN = df.isnull().all(0)
    if any(allNaN):
        print(f'(OPTIONAL) Removing {allNaN.sum()} all-NaN columns...')
        df = df.loc[:,~allNaN]
    else:
        print('GREAT! No all-NaN columns.')
# Interpolate
if interpolate:
    print('\nINTERPOLATING DATA THROUGH TIME')
    df = df.interpolate(method='time', axis=1)
    df = df.interpolate(method='time', axis=1,
                        limit_direction='backward')
missing_ratio = sum(df.isnull().sum())/df.size
print(f'{missing_ratio*100:.2f}% missing data remaining.')

NDVI_df = df
NDVI_index = df.index
print(f'\nNDVI data dimension: {NDVI_df.shape}')

### SAR DATA ###
print('\n===== ADDING SAR DATA =====')
X_SAR = []
default_idx = pd.DataFrame().index
SAR_index = pd.DataFrame().index
for polarization in ['VV', 'VH']:
    allOrbs_df = pd.DataFrame()
    if grid_mean: 
        allOrbs_df_grid = pd.DataFrame()

    print(f'\n----- TYPE {polarization} -----')

    for orb in [81, 8, 30]:

        print(f'\nORBIT {orb}:')

        ### LOAD DATA ###
        # Place your data at './Colza_DB/Plot/'
        # E.g.: './Colza_DB/Plot/2018/Orb30_2017_Colza2018/mean_VH_Orb30.csv'
        path = f'./Colza_DB/Plot/'
        path1 = f'{year}/Orb{orb}_{year-1}_{year}/' 
        path2 = f'{year}/Orb{orb}_{year}_{year}/' 
        filename = f'mean_{polarization}_ORB{orb}.csv'
        if orb == 30:
            filename = f'mean_{polarization}_Orb{orb}.csv'

        # Load csv from year and year-1
        df1 = pd.read_csv(path + path1 + filename, delimiter=',', index_col=0)
        df2 = pd.read_csv(path + path2 + filename, delimiter=',', index_col=0)
        print(f'\t{year-1} data dimensions {df1.shape}')
        print(f'\t{year} data dimensions {df2.shape}')

        ### SANITY CHECKS ###
        print('\n\tSanity checks:')
        # Check if indexes match
        # Compare year-1 to year
        if not df1.index.equals(df2.index):
            print(f'\tWEIRD! Indexes in {year-1} do not match those in {year}!')
            print(f'\t\tIntersection size is {df1.index.intersection(df2.index).shape}')
        df1 = check_duplicates(df1, year-1)
        df2 = check_duplicates(df2, year)


        #### GET GRID MEAN ###
        if grid_mean:
            print('\n\tGetting grid mean...')
            # Load grid mean data
            path = f'./Colza_DB/RPG_grid_mean/'
            filename = f'mean_{polarization}_Orb{orb}.csv'
            gridmean1 = pd.read_csv(path+path1+filename,
                                    delimiter=';', index_col=0)
            gridmean2 = pd.read_csv(path+path2+filename,
                                    delimiter=';', index_col=0)

            df1, df1_grid = get_grid_mean(df1,gridmean1,path1)
            df2, df2_grid = get_grid_mean(df2,gridmean2,path2)

        ### CONCATENATE AND TRIM MONTHS OF INTEREST ###
        print(f'\n\tConcatenating data from {year-1} and {year} ({join})...')
        # Concatenate year and year-1
        df = pd.concat([df1, df2], axis=1, join=join)
        # Sort by date (should already be sorted)
        df = df.sort_index(axis=1, ascending=True)
        # Keeping only months of interest (Oct/N-1 to July/N)
        df.columns = pd.to_datetime(df.columns)
        date_mask = (df.columns >= str(year-1) + '-10-01') \
            & (df.columns < str(year)+'-08-01')
        df = df.loc[:, date_mask]
        print(f'\tData dimensions: ', df.shape)

        if grid_mean:
            # Concatenate year and year-1
            df_grid = pd.concat([df1_grid, df2_grid], axis=1, join=join)
            # Sort by date (should already be sorted)
            df_grid = df_grid.sort_index(axis=1, ascending=True)
            # Keeping only months of interest (Oct/N-1 to July/N)
            df_grid.columns = pd.to_datetime(df_grid.columns)
            date_mask = (df_grid.columns >= str(year-1) + '-10-01') \
                & (df_grid.columns < str(year)+'-08-01')
            df_grid = df_grid.loc[:, date_mask]

        ### EVALUATE MISSING DATA (ZEROS) ###
        data = df.to_numpy()
        zeros_per_row = np.count_nonzero(data == 0, axis=1) + np.count_nonzero(np.isnan(data), axis=1)
        zeros_per_column = np.count_nonzero(data == 0, axis=0) + np.count_nonzero(np.isnan(data), axis=0)
        missing_ratio = zeros_per_row.sum()/data.size
        print(f'\t{missing_ratio*100:.2f}% missing data.')
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
        print('\n\tMerging orbits...')
        if allOrbs_df.empty:
            allOrbs_df = df
        else:
            if not df.index.equals(allOrbs_df.index):
                print(f'\tWEIRD! Indexes do not match among orbits! Joining ({join})')
            allOrbs_df = pd.concat([allOrbs_df, df], axis=1, join=join)
            allOrbs_df = allOrbs_df.sort_index(axis=1, ascending=True)

        if grid_mean:
            if allOrbs_df_grid.empty:
                allOrbs_df_grid = df_grid
            else:
                if not df_grid.index.equals(allOrbs_df_grid.index):
                    print(f'\tWEIRD! Indexes do not match for grid mean! Joining ({join})')
                allOrbs_df_grid = pd.concat([allOrbs_df_grid, df_grid], axis=1, join=join)
                allOrbs_df_grid = allOrbs_df_grid.sort_index(axis=1, ascending=True)

        print(f'\tAll-orbits dimensions: {allOrbs_df.shape}')

    # Harmonize indexes with NDVI data
    print(f'\nRestricting SAR data to NDVI indexes')
    if not all(NDVI_index.isin(allOrbs_df.index)):
        nb_missing = sum(~NDVI_index.isin(allOrbs_df.index))
        print(f'{nb_missing} NDVI index missing on SAR data. Taking intersection.')
    allOrbs_df = allOrbs_df.loc[NDVI_index.intersection(allOrbs_df.index)]
    if grid_mean:
        allOrbs_df_grid = allOrbs_df_grid.loc[NDVI_index.intersection(allOrbs_df_grid.index)]

    print(f'All-orbits dimensions: {allOrbs_df.shape}')

    ### INTERPOLATION ###
    allOrbs_df = allOrbs_df.replace(0, np.nan)
    missing_ratio = allOrbs_df.isnull().sum().sum()/allOrbs_df.size
    if interpolate:
        print('\nINTERPOLATING DATA THROUGH TIME')
        print(f'{missing_ratio*100:.2f}% missing data.')
        allOrbs_df = allOrbs_df.interpolate(method='time', axis=1)
        allOrbs_df = allOrbs_df.interpolate(method='time', axis=1,
                                            limit_direction='backward')

    if grid_mean:
        allOrbs_df_grid = allOrbs_df_grid.replace(0, np.nan)
        missing_ratio = allOrbs_df_grid.isnull().sum().sum()/allOrbs_df_grid.size
        if interpolate:
            print(f'grid mean: {missing_ratio*100:.2f}% missing data.')
            allOrbs_df_grid = allOrbs_df_grid.interpolate(method='time', axis=1)
            allOrbs_df_grid = allOrbs_df_grid.interpolate(method='time', axis=1,
                                                limit_direction='backward')

    # Missing data
    allNaN = allOrbs_df.isnull().all(1)
    if any(allNaN):
        print(f'WEIRD! {allNaN.sum()} all-NaN rows. Removing...')
        allOrbs_df = allOrbs_df.loc[~allNaN]
        if grid_mean:
            allOrbs_df_grid = allOrbs_df_grid.loc[~allNaN]
    else:
        print('GREAT! No all-NaN rows after interpolation.')
    missing_ratio = sum(allOrbs_df.isnull().sum())/allOrbs_df.size
    print(f'{missing_ratio*100:.2f}% missing data remaining.')
    print(f'All-orbits dimensions: {allOrbs_df.shape}')

    ### STACKING POLARIZATIONS ###
    if SAR_index.empty:
        SAR_index = allOrbs_df.index  # Default indexes for comparison
    else:
        print('\nStacking polarizations...')
        if not allOrbs_df.index.equals(SAR_index):
            print(f'WEIRD! Indexes do not match between polarizations !')
            print(f'Taking intersection of size {allOrbs_df.index.intersection(SAR_index).shape}')
            allOrbs_df = allOrbs_df.loc[allOrbs_df.index.intersection(SAR_index)]
            if grid_mean:
                allOrbs_df_grid = allOrbs_df_grid.loc[allOrbs_df_grid.index.intersection(SAR_index)]
            idx = SAR_index.isin(allOrbs_df.index.intersection(SAR_index))
            for i in range(len(X_SAR)): # applying intersection to previous polarizations
                X_SAR[i] = X_SAR[i][idx]
            SAR_index = allOrbs_df.index.intersection(SAR_index)
    X_SAR.append(allOrbs_df.to_numpy())
    if grid_mean:
        X_SAR.append(allOrbs_df.subtract(allOrbs_df_grid).to_numpy())

print(f'\n-------------------------------')

#### SAR DATA ###
X_SAR = np.stack(X_SAR, axis=-1)
# X_SAR = 10*np.log10(X_SAR) # Convert to dB
print(f'\nFinal SAR data dimension: {X_SAR.shape}')

#### NDVI DATA ###
# Harmonize indexes with SAR data
print(f'\nRestricting NDVI data to SAR indexes')
if all(SAR_index.isin(NDVI_index)):
    NDVI_df = NDVI_df.loc[SAR_index]
else:
    nb_missing = sum(~SAR_index.isin(NDVI_index))
    print(f'{nb_missing} SAR index missing on NDVI data. Taking intersection.')
    NDVI_df = NDVI_df.loc[SAR_index.intersects(NDVI_index)]
NDVI_index = NDVI_df.index
X_NDVI = NDVI_df.to_numpy()
print(f'\nFinal NDVI data dimension: {X_NDVI.shape}')

assert SAR_index.equals(NDVI_index), 'SAR and NDVI indexes should be equal'

### LABELS ###
print('\n===== TREATING LABELS =====\n')
# Place your data at './Colza_DB/ID/'
# E.g.: './Colza_DB/ID/gid_5000_2018.csv'
path = './Colza_DB/ID/'
filename = f'gid_5000_{year}.csv'
gid = pd.read_csv(path + filename, delimiter=',', index_col=1)
print(f'Total listed samples: {gid.shape[0]}')

# Check for duplicated indexes
gid = check_duplicates(gid, 'gid')

# Restric to default indexes (from SAR and NDVI data)
print(f'\nRestricting to data (SAR/NDVI) indexes')
gid = gid.loc[SAR_index]
print(f'Total available samples: {gid.shape[0]}')

# Class histograms
if show_plots:
    plt.figure()
    class_counts = gid["CODE_CULTU"].value_counts()
    class_counts[:20].plot.bar()
    plt.title(
        f'Histogram of more common classes (20 out of {len(class_counts)})')
    print(
        f'{class_counts["CZH"]} (winter) + {class_counts["CZP"]} (spring) colza samples.')

y = gid["CODE_CULTU"].to_numpy()

### SAVE DATA ###
np.savez(f'Colza_DB/Colza_data_{year}',
         X_SAR=X_SAR, X_NDVI=X_NDVI, y=y, id_parcels=SAR_index,
         dates_SAR=allOrbs_df.columns, dates_NDVI=NDVI_df.columns)


# if show_plots:
#     import seaborn as sns
#     sns.lineplot(x='variable', y='value',
#                  data=pd.melt(allOrbs_df.loc[y == "CZH"]))
#     allOrbs_df.loc[y == "CZH"].std() # Veeery small! Impressive
#
#     colza = allOrbs_df.loc[y == "CZH"]
#     sns.lineplot(data=colza.iloc[:10,:].T)
#     colza_np = colza.to_numpy()
#     colza_norm = colza_np/np.sqrt((colza_np * colza_np).sum(axis=1))[:,np.newaxis] # normalize rows
#     cov_mtx = colza_norm.dot(colza_norm.T)
