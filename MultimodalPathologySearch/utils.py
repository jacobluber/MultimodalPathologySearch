
#This file has soem util function to calculate the majority 


def update_codex_channel(row):
    # Condition to update the value
    if 'HOECHST' in row['Codex channel']:
        return 'HOECHST'
    return row['Codex channel'].split(' ')[0]


def get_column_mapping():
    column_mapping = {
        'name of the slide': 'slide_name',
        'patch coordination': 'codex_patch_coord',
        'input patch coordination': 'he_patch_coord',
    }
    return column_mapping


def get_sorted_df(df, save=True, file_to_save='temp'):
    #Standardize Channel Names
    df['channel_name'] = df.apply(update_codex_channel, axis = 1)


    #Standardize Column Names
    sorted_df = df.groupby(['input patch coordination']).apply(lambda x: x.sort_values('similarity', ascending=False)).reset_index(drop=True)


    # Update the column names based on the dictionary mapping
    sorted_df = sorted_df.rename(columns=get_column_mapping())

    #Drop some column that are not necessary now
    sorted_df.drop(['input slide', 'Codex channel', 'H&E'], axis = 1, inplace=True)
    
    if save:
        sorted_df.to_csv(f'data_sorted/sorted_{file_to_save}', index=False)
    return sorted_df

def get_votes(df, threshold):
    # Group by he_patch_coord and slide_name, calculate the sum of similarity values
    grouped_data = df.groupby(['he_patch_coord'])['slide_name', 'channel_name', 'similarity']

    patch_slide_channel_sim_dict = {}
    
    for name, group in grouped_data:
    #     print(name, group)
        he_patch = name
        patch_slide_channel_sim_dict[he_patch] = {}
        
        this_patch_slide_channel_sim = {}
        
        for index, row in group.iterrows():
            slide_name = row['slide_name']
            channel_name = row['channel_name']
            similarity = row['similarity']
            
            if similarity > threshold:
                
                if slide_name not in this_patch_slide_channel_sim.keys():
                    this_patch_slide_channel_sim[slide_name] = []
                
                this_patch_slide_channel_sim[slide_name].append({channel_name: similarity})
        if len(this_patch_slide_channel_sim) == 0:
            patch_slide_channel_sim_dict[he_patch] = {
                f'Below threshold {threshold}': 'Below threshold'
                
            }
        else:
            patch_slide_channel_sim_dict[he_patch] = this_patch_slide_channel_sim
            
    return patch_slide_channel_sim_dict 


def get_majority_counts(patch_slide_channel_sim_dict, verbose=False):
    patch_vote = {}
    for key in patch_slide_channel_sim_dict:
        max_voted_key_for_this_patch = max(patch_slide_channel_sim_dict[key], key=lambda k: len(patch_slide_channel_sim_dict[key][k]))
    #     print(max_voted_key_for_this_patch)
        
        patch_vote[max_voted_key_for_this_patch] = patch_vote.get(max_voted_key_for_this_patch, 0) + 1 
        

    sorted_dict = {k: v for k, v in sorted(patch_vote.items(), key=lambda item: item[1], reverse=True)}

    # Print the slide votes
    if verbose:
        for slide_name, votes in sorted_dict.items():
            
            print(f"Slide {slide_name} received {votes} vote(s)")
        
    return sorted_dict