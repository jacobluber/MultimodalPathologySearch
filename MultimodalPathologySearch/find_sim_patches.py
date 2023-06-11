import pandas as pd 
from utils import get_sorted_df, get_votes, get_majority_counts


#provide the query filename that you want to calculate the majority voting for. File is expected to be in the data dir
filename = 'similarity_colon.csv'

df = pd.read_csv(f'data/{filename}')


print('Basic Stats')
print(df.describe())



sorted_df = get_sorted_df(df, save=True, file_to_save=filename)

patch_slide_channel_sim_dict = get_votes(sorted_df, threshold=0.7)
majority_vote_dict = get_majority_counts(patch_slide_channel_sim_dict)
