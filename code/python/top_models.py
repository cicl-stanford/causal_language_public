from grid_search import *

trials = np.arange(30)
# testing ranges
# unoise_range = [0.9, 1.0, 1.1, 1.2]
# not_affect_param_range = [0.4, 0.5]
# stationary_softener_range = [0.4, 0.5]
# speaker_optimality_range = [1, 1.25]
# comparison_threshold_range = [0.4, 0.5]
# comparison_softener_range = [0.4, 0.5]

# actual ranges
unoise_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
not_affect_param_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
stationary_softener_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
speaker_optimality_range = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
comparison_threshold_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
comparison_softener_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
speaker_optimality_range = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
beta_range = np.arange(1,15,0.5)

############################## Find and save best models to file ###################################
print("***Save Top Models***")
print()

cc_full_model = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_version = "and_hm_or_ws_cc", enabled_version = "or_ws")

normal_full_model = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_version = "and_hm_or_ws", enabled_version = "or_ws")

cc_lesion_model = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_version="and_hm_or_ws_cc", enabled_version="or_ws")

normal_lesion_model = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_version="and_hm_or_ws", enabled_version="or_ws")

print("CC Full Model params")
print(cc_full_model)
print()
print("Full Model params")
print(normal_full_model)
print()
print("CC Lesion Model params")
print(cc_lesion_model)
print()
print("Lesion Model params")
print(normal_lesion_model)
print()

models = [{'params': cc_full_model, 'lesion_rsa': 0, 'caused_version': "and_hm_or_ws_cc", 'enabled_version': "or_ws"}, {'params': normal_full_model, 'lesion_rsa': 0, 'caused_version': "and_hm_or_ws", 'enabled_version': "or_ws"}, {'params': cc_lesion_model, 'lesion_rsa': 1, 'caused_version': "and_hm_or_ws_cc", "enabled_version": "or_ws"}, {'params': normal_lesion_model, 'lesion_rsa': 1, 'caused_version': "and_hm_or_ws", 'enabled_version': "or_ws"}]

save_model(models, output_file = "useful_csvs/top_models.csv")


############################## Run grid search on best full and no prag model #################################
print("***Run Cross Validation***")
print()

np.random.seed(1)

# Generate and save splits
splits = generate_splits(trials, 100)

df_splits = pd.DataFrame({"train": [spl[0] for spl in splits], "test": [spl[1] for spl in splits]})
df_splits.to_csv("useful_csvs/crossv_splits.csv")

# Run grid search for full model. Write opt model for each split to file
start_time = time.time()
full = cross_validation(splits, 0, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_version="and_hm_or_ws_cc", enabled_version="or_ws", save_models=True)
end_time = time.time()


print("Runtime:", end_time - start_time)
print()

# Run grid search for no pragmatics model. Write opt model for each split to file
start_time = time.time()
lesion = cross_validation(splits, 1, unoise_range, not_affect_param_range, stationary_softener_range, speaker_optimality_range, beta_range, caused_version="and_hm_or_ws", enabled_version="or_ws", save_models=True)
end_time = time.time()

print("Runtime:", end_time - start_time)
print()

######################## Finding enable and lesion best models ###############################

print("***Save Models for Enable Comparison***")
print()

old_enb_sem_prag = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_version = "and_hm_or_ws_cc", enabled_version = "or_ws")

new_enb_sem_prag = grid_search(trials, lesion_rsa=0, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=speaker_optimality_range, beta_range=None, caused_version = "and_hm_or_ws_cc", enabled_version = "or_ws_and_nh")

old_enb_sem_lesion = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_version = "and_hm_or_ws", enabled_version = "or_ws")

new_enb_sem_lesion = grid_search(trials, lesion_rsa=1, unoise_range=unoise_range, not_affect_param_range=not_affect_param_range, stationary_softener_range=stationary_softener_range, speaker_optimality_range=None, beta_range=beta_range, caused_version = "and_hm_or_ws", enabled_version = "or_ws_and_nh")

print("Old Enb prag model:")
print(old_enb_sem_prag)
print()
print("New Enb prag model:")
print(new_enb_sem_prag)
print()
print("Old Enb lesion model:")
print(old_enb_sem_lesion)
print()
print("New Enb lesion model:")
print(new_enb_sem_lesion)
print()

models = [{'params': old_enb_sem_prag, 'lesion_rsa': 0, 'caused_version': "and_hm_or_ws_cc", 'enabled_version': "or_ws"}, {'params': new_enb_sem_prag, 'lesion_rsa': 0, 'caused_version': "and_hm_or_ws_cc", 'enabled_version': "or_ws_and_nh"}, {'params': old_enb_sem_lesion, 'lesion_rsa': 1, 'caused_version': "and_hm_or_ws", "enabled_version": "or_ws"}, {'params': new_enb_sem_lesion, 'lesion_rsa': 1, 'caused_version': "and_hm_or_ws", 'enabled_version': "or_ws_and_nh"}]

save_model(models, output_file = "useful_csvs/enable_comparison.csv")