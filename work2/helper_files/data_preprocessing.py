
# Function to compute the percentage of rows with missing values
def percentage_missing_values(df):
    total_rows = len(df)

    # Rows with any missing values (returns a boolean Series)
    missing_rows_mask = df.isnull().any(axis=1)

    print("Rows with missing / empty / values:")
    print(f"\n{total_rows}")
    
    # Number of rows with missing values
    missing_rows_count = missing_rows_mask.sum()

    # TODO: Return a txt file containing all of the rows which have missing rows.
    # Inspect manually what is missing, and decide upon whether to remove the rows or 
    # add data (e.g.: median for that column).

    # If adding values via median, need to calculate the median values for each column
    # Perhaps in a new function, called at the beginning of this function.

    # Also need to decide upon a normalisation strategy, and conversion of data to 1 type (continuous or categorical)
    # I need to see what sort of data it is but I'm assuming it'll be continuous.
    # If so, need to investigate what normalisation technique is ideal for continuous data.
    # Also need to keep note in this notebook what exactly is being performed on the data (conversions, normalisations, etc...)
    
    return (missing_rows_count / total_rows) * 100 # Convert to percentage

# Function for data preprocessing
def preprocess_data(df):
    # Implement your data preprocessing steps here
    # Example: Dropping missing values
    df = df.dropna()
    return df

#def normalise_data(df):
    # Implement your data normalisation steps here
    # Example: Normalising data to a range of 0 to 1
#    df = (df - df.min()) / (df.max() - df.min())
#    return df

#def convert_to_categorical(df):
    # Implement your data conversion steps here
    # Example: Converting data to categorical
#    df = df.astype('category')
#    return df

