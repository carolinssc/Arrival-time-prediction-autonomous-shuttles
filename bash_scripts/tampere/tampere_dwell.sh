python src/train_model.py +data.site_name=TAMPERE_FINAL \
                          +data.time_kind=dwell_times \
                          +model=node_encoded_gcn_1l \
                          ++model.model_parameters.input_size=18 \
                          ++data.rf_remove_zero_obs=True \
                          ++model.model_parameters.empty_graph=False \
                          ++logging_parameters.project_name=TAMPERE_dwell_time_RF_GCN
